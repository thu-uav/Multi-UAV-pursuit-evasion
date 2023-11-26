import torch
import functorch
from torchrl.data import UnboundedContinuousTensorSpec
from tensordict.tensordict import TensorDict, TensorDictBase

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.objects import VisualSphere
import omni.isaac.core.objects as objects
import torch.distributions as D

from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import (
    Crazyflie, Firefly, Neo11, Hummingbird
)
from omni_drones.robots.drone import MultirotorBase, MultirotorCfg
import omni_drones.utils.kit as kit_utils
from omni_drones.views import RigidPrimView
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec


class PredatorPrey(IsaacEnv):
    
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        
        self.time_encoding = self.cfg.task.time_encoding
        
        self.drone.initialize()
        self.target_vis = RigidPrimView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)
        drone_state_dim = self.drone.state_spec.shape[-1]
        if self.time_encoding:
            self.time_encoding_dim = 4
            drone_state_dim += self.time_encoding_dim

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            1,
            UnboundedContinuousTensorSpec(drone_state_dim).to(self.device),
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )
        self.vels = self.drone.get_velocities()
        self.init_pos_scale = torch.tensor([2., 2., 0.6], device=self.device) 
        self.init_pos_offset = torch.tensor([-1., -1., 0.3], device=self.device)

        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        # self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        # self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        sphere = objects.DynamicSphere(
            "/World/envs/env_0/target",
            translation=(0., 0., 2.5),
            radius=0.05,
            color=torch.tensor([1., 0., 0.])
        )
        kit_utils.set_collision_properties(sphere.prim_path, collision_enabled=False)
        kit_utils.set_rigid_body_properties(sphere.prim_path, disable_gravity=True)

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.drone.spawn(translations=[(0., 0., 1.)])
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        _, rot = self.init_poses
        self.drone._reset_idx(env_ids, self.training)
        pos = torch.rand(len(env_ids), 1, 3, device=self.device) * self.init_pos_scale + self.init_pos_offset
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot[env_ids], env_ids
        )
        self.drone.set_velocities(torch.zeros_like(self.vels[env_ids]), env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]

        # relative position and heading
        # self.rpos = self.target_pos - self.root_state[..., :3]
        # self.rheading = self.target_heading - self.root_state[..., 13:16]
        # obs = [self.rpos, self.root_state[..., 3:], self.rheading,]
        
        obs = [self.root_state]
        
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        obs = torch.cat(obs, dim=-1)

        return TensorDict({
            "drone.obs": obs,
            "info": self.info
        }, self.batch_size)
    
    def _compute_reward_and_done(self):
        pos, rot, vels, heading, up = self.root_state[..., :19].split([3, 4, 6, 3, 3], dim=-1)
        # uprightness
        ups = functorch.vmap(torch_utils.quat_axis)(rot, axis=2)
        tiltage = torch.abs(1 - ups[..., 2])
        up_reward = 1.0 / (1.0 + torch.square(tiltage))
        # spin reward
        spin = torch.square(self.vels[..., -1])
        spin_reward = 1.0 / (1.0 + torch.square(spin))
        reward = (up_reward + spin_reward) # + effort_reward
        # self._tensordict["drone.return"] += reward.unsqueeze(-1)
        done  = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            | (pos[..., 2] < 0.1)
        )
        return TensorDict({
            "reward": {
                "drone.reward": reward.unsqueeze(-1)
            },
            "done": done
        }, self.batch_size)
