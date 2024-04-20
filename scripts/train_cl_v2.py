import logging
import os
import time

import hydra
import torch
import numpy as np
import wandb

from functorch import vmap
from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import SyncDataCollector, AgentSpec
from omni_drones.utils.torchrl.transforms import (
    LogOnEpisode, 
    FromMultiDiscreteAction, 
    FromDiscreteAction,
    ravel_composite,
    History
)
from omni_drones.utils.wandb import init_wandb
from omni_drones.learning import (
    MAPPOPolicy, 
    HAPPOPolicy,
    QMIXPolicy,
    DQNPolicy,
    SACPolicy,
    TD3Policy,
    MATD3Policy,
    TDMPCPolicy,
    Policy,
    PPOPolicy,
    PPOAdaptivePolicy, PPORNNPolicy
)

from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv, 
    InitTracker, 
    Compose,
)

from tqdm import tqdm

# Class for applying a function every specific steps
class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1

from typing import Sequence
from tensordict import TensorDictBase

# Class for storing statistics for every iteration
class EpisodeStats:
    def __init__(self, in_keys: Sequence[str] = None):
        self.in_keys = in_keys
        self._stats = []
        self._episodes = 0

    # when called, store new values into internal data
    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        done = tensordict.get(("next", "done"))
        truncated = tensordict.get(("next", "truncated"), None)
        done_or_truncated = (
            (done | truncated) if truncated is not None else done.clone()
        )
        if done_or_truncated.any():
            done_or_truncated = done_or_truncated.squeeze(-1) # [env_num, 1, 1]
            self._episodes += done_or_truncated.sum().item()
            self._stats.extend(
                # [env, n, 1]
                tensordict.select(*self.in_keys)[:, 1:][done_or_truncated[:, :-1]].clone().unbind(0)
            )
    
    # pop all the data out and clear internal data
    def pop(self):
        stats: TensorDictBase = torch.stack(self._stats).to_tensordict()
        self._stats.clear()
        return stats

    def __len__(self):
        return len(self._stats)

# import config file
@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    # seed
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(cfg.seed)
    
    # read config and init modules
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))
    
    # link RL algorithms
    from omni_drones.envs.isaac_env import IsaacEnv
    algos = {
        "ppo": PPOPolicy,
        "ppo_adaptive": PPOAdaptivePolicy,
        "ppo_rnn": PPORNNPolicy,
        "mappo": MAPPOPolicy, 
        "happo": HAPPOPolicy,
        "qmix": QMIXPolicy,
        "dqn": DQNPolicy,
        "sac": SACPolicy,
        "td3": TD3Policy,
        "matd3": MATD3Policy,
        "tdmpc": TDMPCPolicy,
        "test": Policy
    }

    # init customize env class
    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    # init tranforms applied to the original env
    transforms = [InitTracker()]

    # transform for observations (output of env)
    # a CompositeSpec is by default processed by a entity-based encoder
    # flatten it to use a MLP encoder instead
    if cfg.task.get("flatten_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("flatten_state", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "state")))
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
    ):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

    if cfg.task.get("history", False):
        transforms.append(History([("agents", "observation")]))
    
    # transform for actions (input of env)
    # optionally discretize the action space or use a controller
    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform == "velocity":
            from omni_drones.controllers import LeePositionController
            from omni_drones.utils.torchrl.transforms import VelController
            controller = LeePositionController(9.81, base_env.drone.params).to(base_env.device)
            transform = VelController(controller)
            transforms.append(transform)
        elif action_transform == "attitude":
            from omni_drones.controllers import AttitudeController as Controller
            from omni_drones.utils.torchrl.transforms import AttitudeController
            controller = Controller(9.81, base_env.drone.params).to(base_env.device)
            transform = AttitudeController(controller)
            transforms.append(transform)
        elif action_transform == "rate":
            from omni_drones.controllers import RateController as _RateController
            from omni_drones.utils.torchrl.transforms import RateController
            controller = _RateController(9.81, base_env.drone.params).to(base_env.device)
            transform = RateController(controller)
            transforms.append(transform)
        elif not action_transform.lower() == "none":
            raise NotImplementedError(f"Unknown action transform: {action_transform}")
    
    # apply the transform to original env and create wrapped env
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    # get parameters
    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device="cuda")

    if cfg.model_dir is not None:
        # torch.save(policy.state_dict(), ckpt_path)
        policy.load_state_dict(torch.load(cfg.model_dir))
        print("Successfully load model!")

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    eval_interval = cfg.get("eval_interval", -1)
    save_interval = cfg.get("save_interval", -1)

    # prepare the container to store statistics of each episode
    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True) 
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    # wrapper for env and policy
    # used to automatically perform policy in the env 
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def render(
        seed: int=0
    ):
        """
        Evaluate function called every certain steps. 
        Used to record statistics and videos.
        """
        frames = []

        # set env to rendering and evaluation mode
        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        # base_env.set_train = False
        env.set_seed(seed)

        from tqdm import tqdm
        t = tqdm(total=base_env.max_episode_length)
        
        def record_frame(*args, **kwargs):
            frame = env.base_env.render(mode="rgb_array")
            frames.append(frame)
            t.update(2)

        # get one episode rollout using current policy and form a trajectory
        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=lambda x: policy(x, deterministic=True),
            callback=Every(record_frame, 2),
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False
        )

        env.train() # set env back to training mode after evaluation
        base_env.set_train = True
        env.reset()

        # after rollout, set rendering mode to not headless and reset env
        base_env.enable_render(not cfg.headless)
        env.reset()

        # get first done index of each trajectory
        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.nanmean(v.float()).item() 
            for k, v in traj_stats.items()
        }
        
        # render video
        if len(frames):
            # video_array = torch.stack(frames)
            video_array = np.stack(frames).transpose(0, 3, 1, 2)
            frames.clear()
            info["recording"] = wandb.Video(
                video_array, fps=0.5 / cfg.sim.dt, format="mp4"
            )
                
        return info

    @torch.no_grad()
    def evaluate(
        seed: int=0
    ):
        """
        Evaluate function called every certain steps. 
        Used to record statistics and videos.
        """

        # set env to rendering and evaluation mode
        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        base_env.set_train = False
        env.set_seed(seed)

        from tqdm import tqdm
        t = tqdm(total=base_env.max_episode_length)

        def record(*args, **kwargs):
            t.update(2)

        # get one episode rollout using current policy and form a trajectory
        trajs = env.rollout(
            max_steps=base_env.max_episode_length,
            policy=lambda x: policy(x, deterministic=True),
            callback=Every(record, 2),
            auto_reset=True,
            break_when_any_done=False,
            return_contiguous=False
        )

        # manual cl evaluation
        eval_num_cylinders = np.arange(cfg.task.cylinder.min_active, cfg.task.cylinder.max_active + 1)
        capture_dict = dict()
        for idx in range(len(eval_num_cylinders)):
            num_cylinder = eval_num_cylinders[idx]
            capture_dict.update({'capture_{}'.format(num_cylinder): trajs['info']['capture_{}'.format(num_cylinder)][:, -1].mean().cpu().numpy()})
        # base_env.update_base_cl(capture_dict=capture_dict)

        # after rollout, set rendering mode to not headless and reset env
        base_env.enable_render(not cfg.headless)
        env.train() # set env back to training mode after evaluation
        base_env.set_train = True
        env.reset()

        # get first done index of each trajectory
        done = trajs.get(("next", "done"))
        first_done = torch.argmax(done.long(), dim=1).cpu()

        def take_first_episode(tensor: torch.Tensor):
            indices = first_done.reshape(first_done.shape+(1,)*(tensor.ndim-2))
            return torch.take_along_dim(tensor, indices, dim=1).reshape(-1)

        traj_stats = {
            k: take_first_episode(v)
            for k, v in trajs[("next", "stats")].cpu().items()
        }

        info = {
            "eval/stats." + k: torch.nanmean(v.float()).item() 
            for k, v in traj_stats.items()
        }
        
        info.update(capture_dict)
                        
        return info

    pbar = tqdm(collector)
    env.train() # set env into training mode
    base_env.set_train = True
    fps = []
    
    # mkdir for cl
    cl_model_dir = os.path.join(run.dir, 'tasks')
    if not os.path.exists(cl_model_dir):
        os.makedirs(cl_model_dir)
    
    # 代表细粒度的cl，metric: distance
    # 内层一直存中等难度的
    # 外层一直更新
    
    # for each iteration, the collector perform one step in the env
    # and get the result rollout as data
    for i, data in enumerate(pbar):
        # fps.append(collector._fps)
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}

        # store rollout data into the container
        episode_stats(data.to_tensordict())

        # if episode_stats is full (as long as the number of envs)
        # transfer all the statistics into info and clear the container
        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else k): torch.mean(v).item() 
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
        
        # update the policy using rollout data and store the training statistics
        info.update(policy.train_op(data.to_tensordict()))

        # update cl before sampling
        if eval_interval > 0 and i % eval_interval == 0:
            logging.info(f"Eval at {collector._frames} steps.")
            info.update(evaluate())
        
        if i % 100 == 0 and i > 0:
            base_env.outer_curriculum_module.save_task(cl_model_dir, i)
        
        # info.update(render())

        # save policy model every certain step
        if save_interval > 0 and i % save_interval == 0:
            
            if hasattr(policy, "state_dict"):
                ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                logging.info(f"Save checkpoint to {str(ckpt_path)}")
                torch.save(policy.state_dict(), ckpt_path)

        # log infos into wandb run
        run.log(info)
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames,
        })

        if max_iters > 0 and i >= max_iters - 1:
            break 
    
    # final evaluation after training
    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    info.update(evaluate())
    run.log(info)

    # final save
    if hasattr(policy, "state_dict"):
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        logging.info(f"Save checkpoint to {str(ckpt_path)}")
        torch.save(policy.state_dict(), ckpt_path)

    wandb.save(os.path.join(run.dir, "checkpoint*"))
    wandb.finish()
    
    simulation_app.close()


if __name__ == "__main__":
    main()
