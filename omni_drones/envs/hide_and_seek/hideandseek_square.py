import torch
import numpy as np
import functorch
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from tensordict.tensordict import TensorDict, TensorDictBase
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import wandb
import time
from functorch import vmap
from omni_drones.utils.torch import cpos, off_diag, quat_axis, others
import torch.distributions as D
from torch.masked import masked_tensor, as_masked_tensor

import omni.isaac.core.objects as objects
# from omni.isaac.core.objects import VisualSphere, DynamicSphere, FixedCuboid, VisualCylinder, FixedCylinder, DynamicCylinder
# from omni.isaac.core.prims import RigidPrimView, GeometryPrimView
import omni.isaac.core.prims as prims
from omni_drones.views import RigidPrimView
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.kit as kit_utils
# import omni_drones.utils.restart_sampling as rsp
from pxr import UsdGeom, Usd, UsdPhysics
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
from omni_drones.utils.scene import design_scene
from ..utils import create_obstacle
import pdb
import copy
from omni_drones.utils.torch import euler_to_quaternion

from omni.isaac.debug_draw import _debug_draw

from .placement import rejection_sampling_with_validation_large_cylinder_cl, generate_outside_cylinders_x_y
from .draw import draw_traj, draw_detection, draw_court
from .draw_circle import Float3, _COLOR_ACCENT, _carb_float3_add


# drones on land by default
# only cubes are available as walls

def is_line_blocked_by_cylinder(drone_pos, target_pos, cylinder_pos, cylinder_size):
    '''
        # 1. compute_reward: for catch reward, not blocked
        # 2. compute_obs: for drones' state, mask the target state in the shadow
        # 3. dummy_prey_policy: if not blocked, the target gets force from the drone
    '''
    # drone_pos: [num_envs, num_agents, 3]
    # target_pos: [num_envs, 1, 3]
    # cylinder_pos: [num_envs, num_cylinders, 3]
    # consider the x-y plane, the distance of c to the line ab
    # d = abs((x2 - x1)(y3 - y1) - (y2 - y1)(x3 - x1)) / sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    diff = drone_pos - target_pos
    diff2 = cylinder_pos - target_pos
    # numerator: [num_envs, num_agents, num_cylinders]
    numerator = torch.abs(torch.matmul(diff[..., 0].unsqueeze(-1), diff2[..., 1].unsqueeze(1)) - torch.matmul(diff[..., 1].unsqueeze(-1), diff2[..., 0].unsqueeze(1)))
    # denominator: [num_envs, num_agents, 1]
    denominator = torch.sqrt(diff[..., 0].unsqueeze(-1) ** 2 + diff[..., 1].unsqueeze(-1) ** 2)
    dist_to_line = numerator / (denominator + 1e-5)

    # which cylinder blocks the line between the ith drone and the target
    # blocked: [num_envs, num_agents, num_cylinders]
    blocked = dist_to_line <= cylinder_size

    return blocked.any(dim=(-1))

class HideAndSeek_square(IsaacEnv): 
    """
    HideAndSeek environment designed for curriculum learning.

    Internal functions:

        _set_specs(self): 
            Set environment specifications for observations, states, actions, 
            rewards, statistics, infos, and initialize agent specifications

        _design_scenes(self): 
            Generate simulation scene and initialize all required objects
            
        _reset_idx(self, env_ids: torch.Tensor): 
            Reset poses of all objects, statistics and infos

        _pre_sim_step(self, tensordict: TensorDictBase):
            Process need to be completed before each step of simulation, 
            including setting the velocities and poses of target and obstacles

        _compute_state_and_obs(self):
            Obtain the observations and states tensor from drone state data
            Observations:   ==> torch.Size([num_envs, num_drone, *, *]) ==> each of dim1 is sent to a separate drone
                state_self:     [relative position of target,       ==> torch.Size([num_envs, num_drone, 1, obs_dim(35)])
                                 absolute velocity of target (expanded to n),
                                 states of all drones,
                                 identity matrix]                   
                state_others:   [relative positions of drones]      ==> torch.Size([num_envs, num_drone, num_drone-1, pos_dim(3)])
                state_frame:    [absolute position of target,       ==> torch.Size([num_envs, num_drone, 1, frame_dim(13)])
                                 absolute velocity of target,
                                 time progress] (expanded to n)     
                obstacles:      [relative position of obstacles,    ==> torch.Size([num_envs, num_drone, num_obstacles, posvel_dim(6)])
                                 absolute velocity of obstacles (expanded to n)]
            States:         ==> torch.Size([num_envs, *, *])
                state_drones:   "state_self" in Obs                 ==> torch.Size([num_envs, num_drone, obs_dim(35)])
                state_frame:    "state_frame" in Obs (unexpanded)   ==> torch.Size([num_envs, 1, frame_dim(13)])
                obstacles:      [absolute position of obstacles,    ==> torch.Size([num_envs, num_obstacles, posvel_dim(6)])
                                 absolute velocity of obstacles]

        _compute_reward_and_done(self):
            Obtain the reward value and done flag from the position of drones, target and obstacles
            Reward = speed_rw + catch_rw + distance_rw + collision_rw
                speed:      if use speed penalty, then punish speed exceeding cfg.task.v_drone
                catch:      reward distance within capture radius 
                distance:   punish distance outside capture radius by the minimum distance
                collision:  if use collision penalty, then punish distance to obstacles within collision radius
            Done = whether or not progress_buf (increases 1 every step) reaches max_episode_length

        _get_dummy_policy_prey(self):
            Get forces (in 3 directions) for the target to move
            Force = f_from_predators + f_from_arena_edge

    """
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()

        self.target = RigidPrimView(
            "/World/envs/env_*/target", 
            reset_xform_properties=False,
            shape=[self.num_envs, -1],
        )
        self.target.initialize()

        self.cylinders = RigidPrimView(
            "/World/envs/env_*/cylinder_*",
            reset_xform_properties=False,
            track_contact_forces=False,
            shape=[self.num_envs, -1],
        )
        self.cylinders.initialize()
        
        self.time_encoding = self.cfg.task.time_encoding

        self.target_init_vel = self.target.get_velocities(clone=True)
        self.env_ids = torch.from_numpy(np.arange(0, cfg.env.num_envs))
        self.arena_size = self.cfg.task.arena_size
        self.returns = self.progress_buf * 0
        self.collision_radius = self.cfg.task.collision_radius
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.v_prey = self.cfg.task.v_drone * self.cfg.task.v_prey
        self.catch_reward_coef = self.cfg.task.catch_reward_coef
        self.detect_reward_coef = self.cfg.task.detect_reward_coef
        self.collision_coef = self.cfg.task.collision_coef
        self.speed_coef = self.cfg.task.speed_coef
        self.use_eval = self.cfg.use_eval
        
        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )

        self.init_drone_pos_dist = D.Uniform(
            torch.tensor([-1.0, -1.0, 0.05], device=self.device),
            torch.tensor([-0.3, -0.3, self.max_height], device=self.device)
        )
        self.init_target_pos_dist = D.Uniform(
            torch.tensor([0.3, 0.3, 0.05], device=self.device),
            torch.tensor([1.0, 1.0, self.max_height], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi
        )

        self.mask_value = -5.0
        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _set_specs(self):        
        drone_state_dim = self.drone.state_spec.shape.numel()
        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4       

        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, 3 + self.time_encoding_dim + 13)),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, 3)), # pos
            "cylinders": UnboundedContinuousTensorSpec((1, 5)), # pos + radius + height
        }).to(self.device)
        state_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, 3 + drone_state_dim)),
            "cylinders": UnboundedContinuousTensorSpec((1, 5)), # pos + radius + height
        }).to(self.device)
        
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": observation_spec.expand(self.drone.n),
                "state": state_spec,
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action": torch.stack([self.drone.action_spec]*self.drone.n, dim=0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((self.drone.n, 1)),                
            })
        }).expand(self.num_envs).to(self.device)

        self.agent_spec["drone"] = AgentSpec(
            "drone", self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state")
        )

        # stats and infos
        stats_spec = CompositeSpec({
            "capture": UnboundedContinuousTensorSpec(1),
            "distance_reward": UnboundedContinuousTensorSpec(1),
            "speed_reward": UnboundedContinuousTensorSpec(1),
            "collision_reward": UnboundedContinuousTensorSpec(1),
            "detect_reward": UnboundedContinuousTensorSpec(1),
            "catch_reward": UnboundedContinuousTensorSpec(1),
            "first_capture_step": UnboundedContinuousTensorSpec(1),
            "sum_detect_step": UnboundedContinuousTensorSpec(1),
            "return": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()
        
    def _design_scene(self): # for render
        self.num_agents = self.cfg.task.num_agents
        self.num_cylinders = self.cfg.task.cylinder.num
        self.drone_detect_radius = self.cfg.task.drone_detect_radius
        self.target_detect_radius = self.cfg.task.target_detect_radius
        self.catch_radius = self.cfg.task.catch_radius
        self.arena_size = self.cfg.task.arena_size
        self.max_height = self.cfg.task.arena_size
        self.cylinder_size = self.cfg.task.cylinder.size
        self.cylinder_height = self.max_height

        # init drone
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        cfg.rigid_props.max_linear_velocity = self.cfg.task.v_drone
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        drone_pos = torch.tensor([
                            [-0.7, -0.7, 0.5],
                            [-0.4, -0.4, 0.5],
                            [-0.7, -0.4, 0.5],
                            [-0.4, -0.7, 0.5],
                        ], device=self.device)
        self.drone.spawn(drone_pos)
        
        # init prey
        objects.DynamicSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=(0.5, 0.5, 0.5),
            radius=0.05,
            color=torch.tensor([1., 0., 0.]),
            mass=1.0
        )

        # cylinders with physcical properties
        self.cylinders_size = []
        cylinders_pos = torch.tensor([
                            [0.0, self.cylinder_size, 0.5 * self.cylinder_height],
                            [0.0, - self.cylinder_size, 0.5 * self.cylinder_height],
                        ], device=self.device)
        for idx in range(self.num_cylinders):
            # orientation = None
            self.cylinders_size.append(self.cylinder_size)
            objects.DynamicCylinder(
                prim_path="/World/envs/env_0/cylinder_{}".format(idx),
                name="cylinder_{}".format(idx),
                translation=cylinders_pos[idx],
                radius=self.cylinder_size,
                height=self.cylinder_height,
                mass=1000000.0
            )

        # self.cylinders_prims = [None] * self.num_cylinders
        # self.cylinders_size = []
        # for idx in range(self.num_cylinders):
        #     self.cylinders_size.append(self.cylinder_size)
        #     attributes = {'axis': 'Z', 'radius': self.cylinder_size, 'height': self.cylinder_height}
        #     self.cylinders_prims[idx] = create_obstacle(
        #         "/World/envs/env_0/cylinder_{}".format(idx), 
        #         prim_type="Cylinder",
        #         translation=cylinders_pos[idx],
        #         attributes=attributes
        #     ) # Use 'self.cylinders_prims[0].GetAttribute('radius').Get()' to get attributes

        objects.DynamicCuboid(
            prim_path="/World/envs/env_0/wall0",
            name="wall0",
            translation= torch.tensor([0.0, 0.5 * self.arena_size, 0.5 * self.cylinder_height], device=self.device),
            scale=[self.arena_size - 0.02, 0.01, self.cylinder_height],
            mass=1000000.0
        )
        objects.DynamicCuboid(
            prim_path="/World/envs/env_0/wall1",
            name="wall1",
            translation= torch.tensor([0.0, -0.5 * self.arena_size, 0.5 * self.cylinder_height], device=self.device),
            scale=[self.arena_size - 0.02, 0.01, self.cylinder_height],
            mass=1000000.0
        )
        objects.DynamicCuboid(
            prim_path="/World/envs/env_0/wall2",
            name="wall2",
            translation= torch.tensor([0.5 * self.arena_size, 0.0, 0.5 * self.cylinder_height], device=self.device),
            scale=[0.01, self.arena_size - 0.02, self.cylinder_height],
            mass=1000000.0
        )
        objects.DynamicCuboid(
            prim_path="/World/envs/env_0/wall3",
            name="wall3",
            translation= torch.tensor([-0.5 * self.arena_size, 0.0, 0.5 * self.cylinder_height], device=self.device),
            scale=[0.01, self.arena_size - 0.02, self.cylinder_height],
            mass=1000000.0
        )
    
        kit_utils.set_rigid_body_properties(
            prim_path="/World/envs/env_0/target",
            disable_gravity=True
        )        

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
    
        drone_pos = self.init_drone_pos_dist.sample((*env_ids.shape, self.num_agents))
        rpy = self.init_rpy_dist.sample((*env_ids.shape, self.num_agents))
        rot = euler_to_quaternion(rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        drone_init_velocities = torch.zeros_like(self.drone.get_velocities())
        self.drone.set_velocities(drone_init_velocities, env_ids)

        self.target_pos = self.init_target_pos_dist.sample((*env_ids.shape, 1))
        self.target.set_world_poses(positions=self.target_pos + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids)

        # reset stats
        self.stats[env_ids] = 0.
        self.stats['first_capture_step'].set_(torch.ones_like(self.stats['first_capture_step']) * self.max_episode_length)

        # for substep in range(1):
        #     self.sim.step(self._should_render(substep))

    def _pre_sim_step(self, tensordict: TensorDictBase):   
        actions = tensordict[("agents", "action")]
        
        self.effort = self.drone.apply_action(actions)
        
        target_vel = self.target.get_velocities()
        forces_target = self._get_dummy_policy_prey()
        
        # fixed velocity
        target_vel[...,:3] = self.v_prey * forces_target / (torch.norm(forces_target, dim=1).unsqueeze(1) + 1e-5)
        
        self.target.set_velocities(target_vel.type(torch.float32), self.env_ids)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        self.info["drone_state"][:] = self.drone_states[..., :13]
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses())
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)
        
        obs = TensorDict({}, [self.num_envs, self.drone.n])

        # cylinders
        # get masked cylinder relative position
        cylinders_pos, _ = self.get_env_poses(self.cylinders.get_world_poses())
        cylinders_rpos = vmap(cpos)(drone_pos, cylinders_pos) # [num_envs, num_agents, num_cylinders, 3]
        # TODO: random height
        self.cylinders_state = torch.concat([
            cylinders_rpos,
            self.cylinder_height * torch.ones(self.num_envs, self.num_agents, self.num_cylinders, 1, device=self.device),
            self.cylinder_size * torch.ones(self.num_envs, self.num_agents, self.num_cylinders, 1, device=self.device),
        ], dim=-1)
        
        cylinders_mdist_z = torch.abs(cylinders_rpos[..., 2]) - 0.5 * self.cylinder_height
        cylinders_mdist_xy = torch.norm(cylinders_rpos[..., :2], dim=-1) - self.cylinder_size
        cylinders_mdist = torch.stack([torch.max(cylinders_mdist_xy, torch.zeros_like(cylinders_mdist_xy)), 
                                       torch.max(cylinders_mdist_z, torch.zeros_like(cylinders_mdist_z))
                                       ], dim=-1)
        # only use the nearest cylinder
        min_distance_idx = torch.argmin(torch.norm(cylinders_mdist, dim=-1), dim=-1)
        self.min_distance_idx_expanded = min_distance_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.cylinders_state.shape[-1])
        # cylinders: the nearest cylinder
        obs["cylinders"] = self.cylinders_state.gather(2, self.min_distance_idx_expanded)

        # state_self
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        target_rpos = vmap(cpos)(drone_pos, target_pos) # [num_envs, num_agents, 1, 3]
        # self.blocked use in the _compute_reward_and_done
        # _get_dummy_policy_prey: recompute
        self.blocked = is_line_blocked_by_cylinder(drone_pos, target_pos, cylinders_pos, self.cylinder_size)
        in_detection_range = (torch.norm(target_rpos, dim=-1) < self.drone_detect_radius)
        # detect: [num_envs, num_agents, 1]
        detect = in_detection_range * (~ self.blocked.unsqueeze(-1))
        # broadcast the detect info to all drones
        broadcast_detect = torch.any(detect, dim=1).unsqueeze(1).expand(-1, self.num_agents, -1)
        target_mask = (~ broadcast_detect).unsqueeze(-1).expand_as(target_rpos) # [num_envs, num_agents, 1, 3]
        target_rpos_masked = target_rpos.clone()
        target_rpos_masked.masked_fill_(target_mask, self.mask_value)

        t = (self.progress_buf / self.max_episode_length).unsqueeze(-1).unsqueeze(-1)

        obs["state_self"] = torch.cat(
            [target_rpos_masked.reshape(self.num_envs, self.num_agents, -1),
             self.drone_states[..., 3:10],
             self.drone_states[..., 13:19],
             t.expand(-1, self.num_agents, self.time_encoding_dim),
             ], dim=-1
        ).unsqueeze(2)
                         
        # state_others
        obs["state_others"] = self.drone_rpos
        
        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = torch.cat(
            [target_rpos.reshape(self.num_envs, self.num_agents, -1),
             self.drone_states, 
             ], dim=-1
        )   # [num_envs, drone.n, drone_state_dim]
        state["cylinders"] = self.cylinders_state.gather(2, self.min_distance_idx_expanded)

        # draw drone trajectory and detection range
        if self._should_render(0) and self.use_eval:
            self._draw_traj()
            if self.drone_detect_radius > 0.0:
                self._draw_detection()   

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "state": state,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        # TODO: check step by step
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses())
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        
        # [num_envs, num_agents]
        target_dist = torch.norm(target_pos - drone_pos, dim=-1)

        # guidance, share distance reward
        min_dist = torch.min(target_dist, dim=-1).values.unsqueeze(-1).expand_as(target_dist)
        active_distance_reward = (min_dist > self.catch_radius).float()
        # share distance reward
        distance_reward = 1.0 / torch.exp(min_dist * active_distance_reward)
        self.stats['distance_reward'].add_(distance_reward.mean(-1).unsqueeze(-1))
        
        # detect
        detect = (target_dist < self.drone_detect_radius)
        masked_detect = detect * (~ self.blocked).float()
        broadcast_detect = torch.any(masked_detect, dim=-1).unsqueeze(-1).expand_as(masked_detect) # cooperative reward
        detect_reward = self.detect_reward_coef * broadcast_detect
        # if detect, current_capture_step = progress_buf
        # else, current_capture_step = max_episode_length
        detect_flag = torch.any(detect_reward, dim=1)
        self.stats['sum_detect_step'] += 1.0 * detect_flag.unsqueeze(1)
        self.stats['detect_reward'].add_(detect_reward.mean(-1).unsqueeze(-1))
        
        # capture
        capture = (target_dist < self.catch_radius)
        masked_capture = capture * (~ self.blocked).float()
        broadcast_capture = torch.any(masked_capture, dim=-1).unsqueeze(-1).expand_as(masked_capture) # cooperative reward
        catch_reward = self.catch_reward_coef * broadcast_capture
        # if capture, current_capture_step = progress_buf
        # else, current_capture_step = max_episode_length
        capture_flag = torch.any(catch_reward, dim=1)
        current_capture_step = capture_flag.float() * self.progress_buf + (~capture_flag).float() * self.max_episode_length
        self.stats['first_capture_step'] = torch.min(self.stats['first_capture_step'], current_capture_step.unsqueeze(1))
        self.stats['catch_reward'].add_(catch_reward.mean(-1).unsqueeze(-1))

        # speed penalty
        drone_vel = self.drone.get_velocities()
        drone_speed_norm = torch.norm(drone_vel[..., :3], dim=-1)
        speed_reward = - self.speed_coef * (drone_speed_norm > self.cfg.task.v_drone)
        self.stats['speed_reward'].add_(speed_reward.mean(-1).unsqueeze(-1))

        # collison with cylinders, drones and walls
        # for cylinders, only consider the nearest cylinder in x-y plane
        # TODO: consider the z-axis
        nearest_cylinder_state = self.cylinders_state.gather(2, self.min_distance_idx_expanded)
        cylinder_pos_dist = torch.norm(nearest_cylinder_state[..., :2], dim= -1).squeeze(-1)
        collision_reward = - self.collision_coef * (cylinder_pos_dist - self.cylinder_size < self.collision_radius).float()
        # for drones
        drone_pos_dist = torch.norm(self.drone_rpos, dim=-1)
        collision_reward += - self.collision_coef * (drone_pos_dist < 2.0 * self.collision_radius).float().sum(-1)
        # for wall
        collision_reward += - self.collision_coef * (drone_pos[..., 0] > 0.5 * self.arena_size).type(torch.float32)
        collision_reward += - self.collision_coef * (drone_pos[..., 0] < - 0.5 * self.arena_size).type(torch.float32)
        collision_reward += - self.collision_coef * (drone_pos[..., 1] > 0.5 * self.arena_size).type(torch.float32)
        collision_reward += - self.collision_coef * (drone_pos[..., 1] < - 0.5 * self.arena_size).type(torch.float32)
        collision_reward += - self.collision_coef * (drone_pos[..., 2] > self.arena_size).type(torch.float32)
        collision_reward += - self.collision_coef * (drone_pos[..., 2] < 0.0).type(torch.float32)

        self.stats['collision_reward'].add_(torch.abs(collision_reward).mean(-1).unsqueeze(-1))
        
        reward = (
            distance_reward
            + detect_reward
            + catch_reward
            + collision_reward
            + speed_reward
        )

        done  = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        )

        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["distance_reward"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["detect_reward"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["catch_reward"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["collision_reward"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["speed_reward"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        
        self.stats["return"] += reward.mean(-1).unsqueeze(-1)
        
        return TensorDict({
            "agents": {
                "reward": reward.unsqueeze(-1)
            },
            "done": done,
        }, self.batch_size)
        
    def _get_dummy_policy_prey(self):
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses(False))
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        cylinders_pos, _ = self.get_env_poses(self.cylinders.get_world_poses())
        
        target_rpos = vmap(cpos)(drone_pos, target_pos)
        target_cylinders_rpos = vmap(cpos)(target_pos, cylinders_pos)
        
        force = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # pursuers
        dist_pos = torch.norm(target_rpos, dim=-1).squeeze(1).unsqueeze(-1)
        blocked = is_line_blocked_by_cylinder(drone_pos, target_pos, cylinders_pos, self.cylinder_size)
        detect_drone = (dist_pos < self.target_detect_radius).squeeze(-1)

        # active_drone: if drone is in th detect range, get force from it
        active_drone = detect_drone * (~blocked).unsqueeze(-1) # [num_envs, num_agents, 1]      
        force_p = -target_rpos.squeeze(1) * (1 / (dist_pos**2 + 1e-5)) * active_drone.unsqueeze(-1)
        force += torch.sum(force_p, dim=1)

        # arena
        # 3D
        force_r = torch.zeros_like(force)
        # right
        force_r[..., 0] = - (0.5 * self.arena_size - target_pos[..., 0]) / ((0.5 * self.arena_size - target_pos[..., 0])**2 + 1e-5)
        # left
        force_r[..., 0] += - (-0.5 * self.arena_size - target_pos[..., 0]) / ((-0.5 * self.arena_size - target_pos[..., 0])**2 + 1e-5)
        # front
        force_r[..., 1] = - (0.5 * self.arena_size - target_pos[..., 1]) / ((0.5 * self.arena_size - target_pos[..., 1])**2 + 1e-5)
        # back
        force_r[..., 1] += - (-0.5 * self.arena_size - target_pos[..., 1]) / ((-0.5 * self.arena_size - target_pos[..., 1])**2 + 1e-5)
        # up
        force_r[...,2] = - (self.arena_size - target_pos[..., 2]) / ((self.arena_size - target_pos[..., 2])**2 + 1e-5)
        # down
        force_r[...,2] += - (0.0 - target_pos[..., 2]) / ((0.0 - target_pos[..., 2])**2 + 1e-5)
        force += force_r
        
        # only get force from the nearest cylinder to the target
        # get the nearest cylinder to the target
        target_cylinders_mdist_z = torch.abs(target_cylinders_rpos[..., 2]) - 0.5 * self.cylinder_height
        target_cylinders_mdist_xy = torch.norm(target_cylinders_rpos[..., :2], dim=-1) - self.cylinder_size
        target_cylinders_mdist = torch.stack([torch.max(target_cylinders_mdist_xy, torch.zeros_like(target_cylinders_mdist_xy)), 
                                       torch.max(target_cylinders_mdist_z, torch.zeros_like(target_cylinders_mdist_z))
                                       ], dim=-1)
        target_min_distance_idx = torch.argmin(torch.norm(target_cylinders_mdist, dim=-1), dim=-1)
        target_min_distance_idx_expanded = target_min_distance_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, target_cylinders_rpos.shape[-1])
        nearest_cylinder_to_target = target_cylinders_rpos.gather(2, target_min_distance_idx_expanded)
        
        force_c = torch.zeros_like(force)
        dist_target_cylinder = torch.norm(nearest_cylinder_to_target[..., :2], dim=-1)
        detect_cylinder = (dist_target_cylinder < self.target_detect_radius)
        force_c[..., :2] = detect_cylinder * nearest_cylinder_to_target[..., :2].squeeze(2) / (dist_target_cylinder**2 + 1e-5)
        force += force_c

        return force.type(torch.float32)

    # def obs_repel(self, pos):
    #     # drone or prey
    #     shape_pos = pos.shape
    #     # cylinders
    #     # cylinder_mask = self.cylinders_mask.reshape(self.num_envs, 1, -1, 1)
    #     # TODO: get prey cylinder_mask
    #     cylinder_mask = torch.zeros()
    #     force = torch.zeros_like(pos)
    #     cylinders_pos, _ = self.get_env_poses(self.cylinders.get_world_poses())
    #     # cylinders_pos, cylinders_height = refresh_cylinder_pos_height(max_cylinder_height=self.cylinder_height,
    #     #                                                                   origin_cylinder_pos=cylinders_pos,
    #     #                                                                   device=self.device)
    #     xy_dist = (torch.norm(vmap(cpos)(pos[..., :2], cylinders_pos[..., :2]), dim=-1) - self.cylinder_size).unsqueeze(-1)
    #     z_dist = vmap(cpos)(pos[..., 2].unsqueeze(-1), self.cylinders_height.unsqueeze(-1))
    #     xy_mask = (xy_dist > 0) * (z_dist < 0) * 1.0
    #     z_mask = (xy_dist < 0) * (z_dist > 0) * 1.0
    #     # xy
    #     drone_to_cy = vmap(cpos)(pos[..., :2], cylinders_pos[..., :2])
    #     dist_drone_cy = torch.norm(drone_to_cy, dim=-1, keepdim=True)
    #     p_drone_cy = drone_to_cy / (dist_drone_cy + 1e-9)
    #     force[..., :2] = torch.sum(p_drone_cy / (torch.relu(dist_drone_cy - self.cylinder_size - 0.05) + 1e-9) * xy_mask * cylinder_mask, dim=-2) # 0.05 also for ball
    #     force[..., 2] = torch.sum(1 / (torch.relu(z_dist - 0.05) + 1e-9) * z_mask * cylinder_mask, dim=-2).squeeze(-1)
        
    #     # if xy_dist>0 and z_dist>0
    #     p_circle = torch.zeros(self.num_envs, shape_pos[1], self.num_cylinders, 3, device=self.device)
    #     p_circle[..., :2] = p_drone_cy * xy_dist
    #     p_circle[..., 2] = z_dist[..., 0]
    #     p_force = torch.sum(self._norm(p_circle, p=1) * (xy_dist > 0) * (z_dist > 0) * cylinder_mask, dim=-2)
    #     force += p_force

    #     return force

    def _norm(self, x, p=0):
        y = x / ((torch.norm(x, dim=-1, keepdim=True)).expand_as(x) + 1e-9)**(p+1)
        return y

    # visualize functions
    def _draw_court(self, size, height):
        self.draw.clear_lines()

        point_list_1, point_list_2, colors, sizes = draw_court(
            2*size, 2*size, height, line_size=5.0
        )
        point_list_1 = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list_1
        ]
        point_list_2 = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list_2
        ]
        self.draw.draw_lines(point_list_1, point_list_2, colors, sizes)   

    def _draw_traj(self):
        drone_pos = self.drone_states[..., :3]
        drone_vel = self.drone.get_velocities()[..., :3]
        point_list1, point_list2, colors, sizes = draw_traj(
            drone_pos[self.central_env_idx, :], drone_vel[self.central_env_idx, :], dt=0.02, size=4.0
        )
        point_list1 = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list1
        ]
        point_list2 = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list2
        ]
        self.draw.draw_lines(point_list1, point_list2, colors, sizes)   
    
    def _draw_detection(self):
        self.draw.clear_points()

        drone_pos = self.drone_states[..., :3]
        drone_ori = self.drone_states[..., 3:7]
        drone_xaxis = quat_axis(drone_ori, 0)
        drone_yaxis = quat_axis(drone_ori, 1)
        drone_zaxis = quat_axis(drone_ori, 2)
        point_list, colors, sizes = draw_detection(
            pos=drone_pos[self.central_env_idx, :],
            xaxis=drone_xaxis[self.central_env_idx, 0, :],
            yaxis=drone_yaxis[self.central_env_idx, 0, :],
            zaxis=drone_zaxis[self.central_env_idx, 0, :],
            drange=self.drone_detect_radius,
        )
        point_list = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list
        ]
        self.draw.draw_points(point_list, colors, sizes)
    