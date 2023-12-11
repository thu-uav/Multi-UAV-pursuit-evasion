import logging
import os
import time
import hydra
import numpy as np

from tqdm import tqdm

from omegaconf import OmegaConf

from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.utils.torchrl import AgentSpec
from omni_drones.utils.set_transforms import set_transforms
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
from torchrl.envs.transforms import TransformedEnv, Compose


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

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    # read config and init simulation
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    
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
    env = TransformedEnv(base_env, Compose(*set_transforms(base_env, cfg)))
    env.set_seed(cfg.seed)

    # get policy
    agent_spec: AgentSpec = env.agent_spec["drone"]
    policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device="cuda")

    base_env.enable_render(True)
    base_env.eval()
    env.eval()
    env.set_seed(cfg.seed)

    # function for recording frames
    frames = []
    t = tqdm(total=base_env.max_episode_length)
    def record_frame(*args, **kwargs):
        frame = env.base_env.render(mode="rgb_array")
        frames.append(frame)
        t.update(2)

    # rollout
    env.rollout(
        max_steps=base_env.max_episode_length,
        policy=lambda x: policy(x, deterministic=True),
        callback=Every(record_frame, 2),
        auto_reset=True,
        break_when_any_done=False,
        return_contiguous=False
    )

    # render video
    if len(frames):
        from torchvision.io import write_video
        video_array = np.stack(frames)
        write_video(f"render.mp4", video_array, fps=1/cfg.sim.dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
