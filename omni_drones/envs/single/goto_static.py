import functorch
import torch
import torch.distributions as D

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.objects as objects
import numpy as np

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni_drones.utils.torch import euler_to_quaternion, quat_axis, quat_rotate_inverse

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
import collections
from ..utils import create_obstacle
from omni.isaac.core.prims import GeometryPrimView

class Goto_static(IsaacEnv):
    r"""
    A basic control task. The goal for the agent is to maintain a stable
    position and heading in mid-air without drifting. 

    Observation
    -----------
    The observation space consists of the following part:

    - `rpos` (3): The position relative to the target hovering position.
    - `root_state` (16 + num_rotors): The basic information of the drone (except its position), 
      containing its rotation (in quaternion), velocities (linear and angular), 
      heading and up vectors, and the current throttle.
    - `rheading` (3): The difference between the reference heading and the current heading.
    - *time_encoding*:

    Reward 
    ------
    - pos: 
    - heading_alignment:
    - up:
    - spin:

    The total reward is 

    .. math:: 
    
        r = r_\text{pos} + r_\text{pos} * (r_\text{up} + r_\text{heading})

    Episode End
    -----------
    - Termination: 

    Config
    ------

    """
    def __init__(self, cfg, headless):
        # self.reward_effort_weight = cfg.task.reward_effort_weight
        self.reward_action_smoothness_weight = cfg.task.reward_action_smoothness_weight
        self.reward_distance_scale = cfg.task.reward_distance_scale
        self.reward_collision = cfg.task.reward_collision
        self.time_encoding = cfg.task.time_encoding
        self.reach_threshold = cfg.task.reach_threshold
        self.reward_bonus_scale = cfg.task.reward_bonus_scale
        self.use_eval = cfg.task.use_eval
        
        self.randomization = cfg.task.get("randomization", {})

        super().__init__(cfg, headless)

        self.drone.initialize()
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
        
        self.target_vis = ArticulationView(
            "/World/envs/env_*/target",
            reset_xform_properties=False
        )
        self.target_vis.initialize()
        
        self.cylinder_pos = torch.zeros(self.num_envs, 2, 3, device=self.device)
        # 2 cylinders
        self.cylinder_pos[:, 0, 0] = - 0.5 * self.cylinder_radius
        self.cylinder_pos[:, 0, 1] = 1.5 * self.cylinder_radius
        self.cylinder_pos[:, 0, 2] = 0.5 * self.cylinder_height
        self.cylinder_pos[:, 1, 0] = 0.5 * self.cylinder_radius
        self.cylinder_pos[:, 1, 1] = - 1.5 * self.cylinder_radius
        self.cylinder_pos[:, 1, 2] = 0.5 * self.cylinder_height
        # wall
        self.wall_pos = torch.zeros(self.num_envs, 1, 2, device=self.device)
        self.wall_pos[..., 0] = self.narrow_width * 0.5
        self.wall_pos[..., 1] = - self.narrow_width * 0.5

        self.init_poses = self.drone.get_world_poses(clone=True)
        self.init_vels = torch.zeros_like(self.drone.get_velocities())

        self.init_drone_pos_dist = D.Uniform(
            torch.tensor([-self.cylinder_radius, 0.8, 0.2], device=self.device),
            torch.tensor([self.cylinder_radius, 1.2, 2.0], device=self.device)
        )
        self.init_target_pos_dist = D.Uniform(
            torch.tensor([-self.cylinder_radius, - 1.2, 0.2], device=self.device),
            torch.tensor([self.cylinder_radius, - 0.8, 2.0], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.2, -0.2, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 0.2], device=self.device) * torch.pi
        )
        
        if self.use_eval:
            self.init_drone_pos_dist = D.Uniform(
                torch.tensor([0.0, 1.0, 1.0], device=self.device),
                torch.tensor([0.0, 1.0, 1.0], device=self.device)
            )
            self.init_target_pos_dist = D.Uniform(
                torch.tensor([0.0, -1.0, 1.0], device=self.device),
                torch.tensor([0.0, -1.0, 1.0], device=self.device)
            )
            self.init_rpy_dist = D.Uniform(
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi,
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi
            )

        self.alpha = 0.8
        self.all_cylinder_height = torch.ones(self.num_envs, 1, 1, device=self.device) * self.cylinder_height
        self.all_cylinder_radius = torch.ones(self.num_envs, 1, 1, device=self.device) * self.cylinder_radius

        self.last_linear_v = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_v = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_linear_a = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_a = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_linear_jerk = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_jerk = torch.zeros(self.num_envs, 1, device=self.device)

        self.last_actions = torch.zeros(self.num_envs, 1, 4, device=self.device)

    def _design_scene(self):
        import omni_drones.utils.kit as kit_utils
        import omni.isaac.core.utils.prims as prim_utils

        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        drone_prim = self.drone.spawn(translations=[(0.0, 0.0, 0.05)])[0]

        target_vis_prim = prim_utils.create_prim(
            prim_path="/World/envs/env_0/target",
            usd_path=self.drone.usd_path,
            translation=(0.0, 0.0, 1.),
        )

        kit_utils.set_nested_collision_properties(
            target_vis_prim.GetPath(), 
            collision_enabled=False
        )
        kit_utils.set_nested_rigid_body_properties(
            target_vis_prim.GetPath(),
            disable_gravity=True
        )

        self.cylinder_height = 2.0
        self.cylinder_radius = 0.2
        self.narrow_width = 0.8
        objects.VisualCylinder(
            prim_path="/World/envs/env_0/Cylinder0",
            name="cylinder0",
            translation= torch.tensor([-0.2, 0.5, self.cylinder_height / 2.0], device=self.device),
            radius=self.cylinder_radius,
            height=self.cylinder_height,
        )
        objects.VisualCylinder(
            prim_path="/World/envs/env_0/Cylinder1",
            name="cylinder1",
            translation= torch.tensor([0.2, -0.5, self.cylinder_height / 2.0], device=self.device),
            radius=self.cylinder_radius,
            height=self.cylinder_height,
        )
        objects.DynamicCuboid(
            prim_path="/World/envs/env_0/wall0",
            name="wall0",
            translation= torch.tensor([0.5 * self.narrow_width, 0.0, self.cylinder_height / 2.0], device=self.device),
            scale=[0.01, 5.0, self.cylinder_height],
            mass=1000000.0
        )
        objects.DynamicCuboid(
            prim_path="/World/envs/env_0/wall1",
            name="wall1",
            translation= torch.tensor([-0.5 * self.narrow_width, 0.0, self.cylinder_height / 2.0], device=self.device),
            scale=[0.01, 5.0, self.cylinder_height],
            mass=1000000.0
        )

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 3 + 3 + 4 + 3 + 3 # position, velocity, quaternion, heading, up

        observation_dim += 3 * 2 + 1 + 1 # cylinder pos, radius and height
        
        observation_dim += 2 # wall

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.latency = 2 if self.cfg.task.latency else 0
        self.obs_buffer = collections.deque(maxlen=self.latency)

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                #"observation": UnboundedContinuousTensorSpec((1, observation_dim-6), device=self.device),   remove throttle
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
                "intrinsics": self.drone.intrinsics_spec.unsqueeze(0).to(self.device)
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
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics")
        )

        stats_spec = CompositeSpec({
            "return": UnboundedContinuousTensorSpec(1),
            "pos_bonus": UnboundedContinuousTensorSpec(1),
            "head_bonus": UnboundedContinuousTensorSpec(1),
            "reward_pos": UnboundedContinuousTensorSpec(1),
            "reward_pos_bonus": UnboundedContinuousTensorSpec(1),
            "reward_spin": UnboundedContinuousTensorSpec(1),
            "reward_up": UnboundedContinuousTensorSpec(1),
            "reward_collision": UnboundedContinuousTensorSpec(1),
            "reward_collision_wall": UnboundedContinuousTensorSpec(1),
            "reach_time": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "pos_error": UnboundedContinuousTensorSpec(1),
            "action_error_mean": UnboundedContinuousTensorSpec(1),
            "action_error_max": UnboundedContinuousTensorSpec(1),
            "action_smoothness_mean": UnboundedContinuousTensorSpec(1),
            "action_smoothness_max": UnboundedContinuousTensorSpec(1),
            "linear_v_max": UnboundedContinuousTensorSpec(1),
            "angular_v_max": UnboundedContinuousTensorSpec(1),
            "linear_a_max": UnboundedContinuousTensorSpec(1),
            "angular_a_max": UnboundedContinuousTensorSpec(1),
            "linear_jerk_max": UnboundedContinuousTensorSpec(1),
            "angular_jerk_max": UnboundedContinuousTensorSpec(1),
            "linear_v_mean": UnboundedContinuousTensorSpec(1),
            "angular_v_mean": UnboundedContinuousTensorSpec(1),
            "linear_a_mean": UnboundedContinuousTensorSpec(1),
            "angular_a_mean": UnboundedContinuousTensorSpec(1),
            "linear_jerk_mean": UnboundedContinuousTensorSpec(1),
            "angular_jerk_mean": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids, self.training)
        
        drone_pos = self.init_drone_pos_dist.sample((*env_ids.shape, 1))
        rpy = self.init_rpy_dist.sample((*env_ids.shape, 1))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(self.init_vels[env_ids], env_ids)
        
        self.last_actions[env_ids] = 2.0 * torch.square(self.drone.throttle) - 1.0

        self.target_pos = self.init_target_pos_dist.sample((*env_ids.shape, 1))
        # self.target_pos = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.target_vis.set_world_poses(positions=self.target_pos + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids)

        # # cylinder
        # self.cylinder.set_world_poses(
        #     self.cylinders_pos[env_ids] + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids
        # )

        # set last values
        self.last_linear_v[env_ids] = torch.norm(self.init_vels[env_ids][..., :3], dim=-1)
        self.last_angular_v[env_ids] = torch.norm(self.init_vels[env_ids][..., 3:], dim=-1)
        self.last_linear_a[env_ids] = torch.zeros_like(self.last_linear_v[env_ids])
        self.last_angular_a[env_ids] = torch.zeros_like(self.last_angular_v[env_ids])
        self.last_linear_jerk[env_ids] = torch.zeros_like(self.last_linear_a[env_ids])
        self.last_angular_jerk[env_ids] = torch.zeros_like(self.last_angular_a[env_ids])

        self.stats[env_ids] = 0.
        self.stats['reach_time'][env_ids] = self.max_episode_length
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        if self.cfg.task.action_noise:
            actions *= torch.randn(actions.shape, device=self.device) * 0.1 + 1
        
        self.effort = self.drone.apply_action(actions)
        
        # action difference
        action_error = torch.norm(actions - self.last_actions, dim=-1)
        self.stats['action_error_mean'].add_(action_error)
        self.stats['action_error_max'].set_(torch.max(action_error, self.stats['action_error_max']))
        self.last_actions = actions.clone()
        
    def _compute_state_and_obs(self):
        self.root_state = self.drone.get_state()
        self.info["drone_state"][:] = self.root_state[..., :13]

        # relative position and heading
        self.rpos = self.target_pos - self.root_state[..., :3]
        
        obs = [self.rpos, self.root_state[..., 3:10], self.root_state[..., 13:19],]  # (relative) position, velocity, quaternion, heading, up
        
        # cylinder_pos, _ = self.get_env_poses(self.cylinder.get_world_poses())
        self.rpos_cylinder = self.cylinder_pos - self.root_state[..., :3]
        obs.append(self.rpos_cylinder.reshape(self.num_envs, -1).unsqueeze(1))
        obs.append(self.all_cylinder_height)
        obs.append(self.all_cylinder_radius)
        self.rpos_wall = self.wall_pos - self.root_state[..., 0].unsqueeze(-1)
        obs.append(self.rpos_wall)
        
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))
        
        self.stats["action_smoothness_mean"].add_(self.drone.throttle_difference)
        self.stats["action_smoothness_max"].set_(torch.max(self.drone.throttle_difference, self.stats["action_smoothness_max"]))
        # linear_v, angular_v
        self.linear_v = torch.norm(self.root_state[..., 7:10], dim=-1)
        self.angular_v = torch.norm(self.root_state[..., 10:13], dim=-1)
        self.stats["linear_v_max"].set_(torch.max(self.stats["linear_v_max"], torch.abs(self.linear_v)))
        self.stats["linear_v_mean"].add_(self.linear_v)
        self.stats["angular_v_max"].set_(torch.max(self.stats["angular_v_max"], torch.abs(self.angular_v)))
        self.stats["angular_v_mean"].add_(self.angular_v)
        # linear_a, angular_a
        self.linear_a = torch.abs(self.linear_v - self.last_linear_v) / self.dt
        self.angular_a = torch.abs(self.angular_v - self.last_angular_v) / self.dt
        self.stats["linear_a_max"].set_(torch.max(self.stats["linear_a_max"], torch.abs(self.linear_a)))
        self.stats["linear_a_mean"].add_(self.linear_a)
        self.stats["angular_a_max"].set_(torch.max(self.stats["angular_a_max"], torch.abs(self.angular_a)))
        self.stats["angular_a_mean"].add_(self.angular_a)
        # linear_jerk, angular_jerk
        self.linear_jerk = torch.abs(self.linear_a - self.last_linear_a) / self.dt
        self.angular_jerk = torch.abs(self.angular_a - self.last_angular_a) / self.dt
        self.stats["linear_jerk_max"].set_(torch.max(self.stats["linear_jerk_max"], torch.abs(self.linear_jerk)))
        self.stats["linear_jerk_mean"].add_(self.linear_jerk)
        self.stats["angular_jerk_max"].set_(torch.max(self.stats["angular_jerk_max"], torch.abs(self.angular_jerk)))
        self.stats["angular_jerk_mean"].add_(self.angular_jerk)
        
        # set last
        self.last_linear_v = self.linear_v.clone()
        self.last_angular_v = self.angular_v.clone()
        self.last_linear_a = self.linear_a.clone()
        self.last_angular_a = self.angular_a.clone()
        self.last_linear_jerk = self.linear_jerk.clone()
        self.last_angular_jerk = self.angular_jerk.clone()
        
        obs = torch.cat(obs, dim=-1)

        if self.cfg.task.add_noise:
            obs *= torch.randn(obs.shape, device=self.device) * 0.1 + 1 # add a gaussian noise of mean 0 and variance 0.01
        
        if self.latency:
            self.obs_buffer.append(obs)
            obs = self.obs_buffer[0]

        return TensorDict({
            "agents": {
                "observation": obs,
                "intrinsics": self.drone.intrinsics
            },
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        # pose reward
        pos_error = torch.norm(self.rpos, dim=-1)
        
        self.stats["pos_error"].add_(pos_error)
        
        cylinder_pos_error = torch.norm(self.rpos_cylinder[..., :2], dim=-1)

        reward_pos = - pos_error * self.reward_distance_scale
        reward_pos_bonus = ((pos_error <= self.reach_threshold) * self.reward_bonus_scale).float()
        
        reward_collision = - self.reward_collision * (cylinder_pos_error <= self.cylinder_radius + 0.05).float().sum(-1).unsqueeze(1)
        
        reward_collision_wall = - self.reward_collision * ((self.rpos_wall[..., 0] <= 0.05).float() + (self.rpos_wall[..., 1] >= -0.05).float())
        
        reward_up = torch.square((self.drone.up[..., 2] + 1) / 2)

        spin = torch.square(self.drone.vel[..., -1])
        reward_spin = 0.5 / (1.0 + torch.square(spin))
        
        reward = (
            reward_pos
            + reward_pos_bonus
            + reward_spin
            + reward_up
            + reward_collision
            + reward_collision_wall
        )

        self.stats['reward_pos'].add_(reward_pos)
        self.stats['reward_pos_bonus'].add_(reward_pos_bonus)
        self.stats['reward_spin'].add_(reward_spin)
        self.stats['reward_up'].add_(reward_up)
        self.stats['reward_collision'].add_(reward_collision)
        self.stats['reward_collision_wall'].add_(reward_collision_wall)
        reach_flag = (reward_pos_bonus > 0).float()
        current_reach = self.progress_buf.unsqueeze(1) * reach_flag + self.max_episode_length * (1.0 - reach_flag)
        self.stats['reach_time'].set_(torch.min(self.stats['reach_time'], current_reach))
        
        # done_misbehave = (distance > 0.5)

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
            # | done_misbehave
        )

        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["pos_error"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['action_error_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["linear_v_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["angular_v_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["linear_a_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["angular_a_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["linear_jerk_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["angular_jerk_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['action_smoothness_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        
        self.stats['reward_pos'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_spin'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_pos_bonus'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_up'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_collision'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['reward_collision_wall'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        
        self.stats["return"].add_(reward)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        return TensorDict(
            {
                "agents": {
                    "reward": reward.unsqueeze(-1)
                },
                "done": done,
            },
            self.batch_size,
        )
