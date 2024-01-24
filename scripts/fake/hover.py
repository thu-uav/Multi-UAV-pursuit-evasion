import functorch
import torch
import torch.distributions as D

from .fake_env import AgentSpec, FakeEnv, FakeRobot
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, BinaryDiscreteTensorSpec

class FakeHover(FakeEnv):
    def __init__(self, cfg, headless):
        self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.time_encoding = cfg.task.time_encoding
        self.randomization = cfg.task.get("randomization", {})
        self.alpha = 0.8
        self.cfg = cfg
        self.drone = FakeRobot(self.cfg.task, self.cfg.task.drone_model, device = cfg.sim.device)

        super().__init__(cfg, headless)
        
        self.target_pos = torch.tensor([[0., 0., .5]])
        

    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        # observation_dim = drone_state_dim
        observation_dim = 3 + 3 + 4 + 3 + 3

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                #"observation": UnboundedContinuousTensorSpec((1, observation_dim-6), device=self.device),   remove throttle
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                # "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": self.drone.action_spec.unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done":  BinaryDiscreteTensorSpec(1, dtype=bool, device=self.device)
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "heading_alignment": UnboundedContinuousTensorSpec(1),
            "uprightness": UnboundedContinuousTensorSpec(1),
            "action_smoothness": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def _compute_state_and_obs(self) -> TensorDictBase:
        self.update_drone_state()
        self.rpos = self.target_pos - self.drone_state[..., :3]
        obs = [self.rpos, self.drone_state[..., 3:], torch.zeros((self.num_cf, 7))]
        obs = torch.concat(obs, dim=1).unsqueeze(0)
        
        return TensorDict({
            "agents": {
                "observation": obs,
            },
        }, self.num_envs)

    def _compute_reward_and_done(self) -> TensorDictBase:
        reward = torch.zeros((self.num_envs, 1, 1))
        done = torch.zeros((self.num_envs, 1, 1), dtype=torch.bool)
        return TensorDict(
            {
                "agents": {
                    "reward": reward,
                },
                "done": done,
            },
            self.num_envs,
        )