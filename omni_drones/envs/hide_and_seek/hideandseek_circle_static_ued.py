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

from omni.isaac.debug_draw import _debug_draw

from .placement import plot_heatmap, rejection_sampling_all_obj_xy, generate_outside_cylinders_x_y
from .draw import draw_traj, draw_detection
from .draw_circle import Float3, _COLOR_ACCENT, _carb_float3_add, draw_court_circle


# drones on land by default
# only cubes are available as walls

class InnerCurriculum(object):
    """
    Naive CL, use [catch_radius, speed_ratio, num_agents]
    """
    def __init__(self) -> None:
        self.start_catch_radius = 0.5
        self.start_speed = 1.3
        # self.start_num_agents = 4
        
    def set_target_task(self, **kwargs):
        self.end_catch_radius = kwargs['catch_radius']
        self.end_speed = kwargs['speed']
        num_axis = 10
        self.training_order = []
        if self.start_catch_radius == self.end_catch_radius:
            training_catch_radius = np.array([self.start_catch_radius])
        else:
            training_catch_radius = np.linspace(self.start_catch_radius, \
                                            self.end_catch_radius, \
                                            num_axis)
        if self.start_speed == self.end_speed:
            training_speed = np.array([self.start_speed])
        else:
            training_speed = np.linspace(self.start_speed, \
                                        self.end_speed, \
                                        num_axis)
        catch_idx = 0
        speed_idx = 0
        if training_speed.shape[0] == 1 and training_catch_radius.shape[0] == 1:
            self.training_order.append([self.start_catch_radius, self.start_speed])
        else:
            while (speed_idx < training_speed.shape[0]):
                while (catch_idx < training_catch_radius.shape[0]):
                    self.training_order.append([training_catch_radius[catch_idx], \
                                                training_speed[speed_idx]
                                                ])
                    catch_idx += 1
                catch_idx = training_catch_radius.shape[0] - 1
                self.training_order.append([training_catch_radius[catch_idx], \
                                training_speed[speed_idx]
                                ])
                speed_idx += 1
 
    def get_training_task(self):
        return self.training_order[0]
    
    def update_curriculum_queue(self):
        if len(self.training_order) > 1:
            self.training_order.pop(0)

class OuterCurriculum(object):
    '''
    propose new environments
    '''
    def __init__(self, cfg, device) -> None:
        self.OOD_num_cylinders = cfg.task.cylinder.num
        self.max_num_cylinders = cfg.task.cylinder.max_active # training dist: 0 ~ max_num_cylinders
        self.num_drones = cfg.task.num_agents
        self.max_num_obj = self.num_drones + self.OOD_num_cylinders + 1 # drone + target + cylinders
        self.cylinder_size = cfg.task.cylinder.size
        self.arena_size = cfg.task.arena_size
        self.device = device
        self.omega_min = 0.5
        self.omega_max = 0.9
        self.moderate_archive = [] # save tasks in [omega_min, omega_max]
        np.random.seed(cfg.seed)
    
    def generate_envs(self, num_envs):
        '''
        randomly generate envs without evaluation
        '''
        new_envs = [] # env_one: [drone_pos, target_pos, num_cylinder, cylinders_pos]
        num_active_cylinders = []
        occupancy_matrix_all = []
        while len(new_envs) < num_envs:
            current_num_cylinder = np.random.randint(0, self.max_num_cylinders + 1, 1, dtype=int)[0]
            num_active_cylinders.append(current_num_cylinder)
            obj_pos, occupancy_matrix = rejection_sampling_all_obj_xy(
                arena_size=self.arena_size,
                cylinder_size=self.cylinder_size,
                num_drones=self.num_drones,
                num_cylinders=self.max_num_cylinders,
                device=self.device
            )
            new_envs.append(obj_pos)
            occupancy_matrix_all.append(occupancy_matrix)
        return torch.stack(new_envs), num_active_cylinders, np.array(occupancy_matrix_all)

    def collect_moderate_envs(self, envs, captures):
        '''
        save tasks into moderate_archive
        envs: the same as the output of generate_envs()
        captures: obtain from the rollout procedure
        '''
        for env, capture in zip(envs, captures):
            if capture >= self.omega_min and capture <= self.omega_max:
                self.moderate_archive.append(env)
  
class HideAndSeek_circle_static_UED(IsaacEnv): 
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
            reset_xform_properties=False
        )
        self.target.initialize()

        self.cylinders = RigidPrimView(
            "/World/envs/env_*/cylinder_*",
            reset_xform_properties=False,
            track_contact_forces=True,
            shape=[self.num_envs, -1],
        )
        self.cylinders.initialize()
        
        self.time_encoding = self.cfg.task.time_encoding

        self.target_init_vel = self.target.get_velocities(clone=True)
        self.env_ids = torch.from_numpy(np.arange(0, cfg.env.num_envs))
        self.arena_size = self.cfg.task.arena_size
        self.returns = self.progress_buf * 0
        self.catch_radius = self.cfg.task.catch_radius
        self.collision_radius = self.cfg.task.collision_radius
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.v_prey = self.cfg.task.v_drone * self.cfg.task.v_prey
        
        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )

        self.mask_value = -1.0
        
        # outer CL
        self.use_outer_cl = self.cfg.task.use_outer_cl
        if self.use_outer_cl:
            self.outer_curriculum_module = OuterCurriculum(cfg=self.cfg, device=self.device)
        
        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _set_specs(self):        
        drone_state_dim = self.drone.state_spec.shape.numel()
        frame_state_dim = 9 # target_pos_dim + target_vel
        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            frame_state_dim += self.time_encoding_dim        

        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, 3 + 6 + drone_state_dim)),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, 3)), # pos
            "state_frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
            "cylinders": UnboundedContinuousTensorSpec((self.num_cylinders, 5)), # pos + radius + height
        }).to(self.device)
        state_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, 3 + 6 + drone_state_dim)),
            "state_frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
            "cylinders": UnboundedContinuousTensorSpec((self.num_cylinders, 5)), # pos + radius + height
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
            "capture_episode": UnboundedContinuousTensorSpec(1),
            "capture_per_step": UnboundedContinuousTensorSpec(1),
            "first_capture_step": UnboundedContinuousTensorSpec(1),
            # "cover_rate": UnboundedContinuousTensorSpec(1),
            "catch_radius": UnboundedContinuousTensorSpec(1),
            "v_prey": UnboundedContinuousTensorSpec(1),
            "return": UnboundedContinuousTensorSpec(1),
            "drone1_speed_per_step": UnboundedContinuousTensorSpec(1),
            "drone2_speed_per_step": UnboundedContinuousTensorSpec(1),
            "drone3_speed_per_step": UnboundedContinuousTensorSpec(1),
            "drone1_max_speed": UnboundedContinuousTensorSpec(1),
            "drone2_max_speed": UnboundedContinuousTensorSpec(1),
            "drone3_max_speed": UnboundedContinuousTensorSpec(1),
            "prey_speed": UnboundedContinuousTensorSpec(1)
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
        self.max_active_cylinders = self.cfg.task.cylinder.max_active
        self.random_active = self.cfg.task.cylinder.random_active
        self.cylinder_size = self.cfg.task.cylinder.size
        self.detect_range = self.cfg.task.detect_range
        self.arena_size = self.cfg.task.arena_size
        size = self.arena_size
        self.cylinder_height = 2 * size

        obj_pos, _ = rejection_sampling_all_obj_xy(
            arena_size=self.arena_size, 
            cylinder_size=self.cylinder_size, 
            num_drones=self.num_agents, 
            num_cylinders=self.max_active_cylinders, 
            device=self.device)

        drone_z = D.Uniform(
                torch.tensor([0.1], device=self.device),
                torch.tensor([2 * size], device=self.device)
            ).sample((1, self.num_agents)).squeeze(0)
        target_z = D.Uniform(
            torch.tensor([0.1], device=self.device),
            torch.tensor([2 * size], device=self.device)
        ).sample()
        
        drone_x_y = obj_pos[:self.num_agents].clone()
        target_x_y = obj_pos[self.num_agents].clone()
        drone_pos = torch.concat([drone_x_y, drone_z], dim=-1)
        target_pos = torch.concat([target_x_y, target_z], dim=-1)
        
        if self.random_active:
            num_active_cylinder = np.random.randint(0, self.max_active_cylinders + 1, 1, dtype=int)[0]
        else:
            num_active_cylinder = self.max_active_cylinders
        num_inactive = self.num_cylinders - num_active_cylinder
        active_cylinder_x_y = obj_pos[self.num_agents + 1:].clone()[:num_active_cylinder]
        inactive_cylinders_x_y = generate_outside_cylinders_x_y(arena_size=self.arena_size, 
                                                                num_envs=1, 
                                                                device=self.device)[:num_inactive]
        cylinder_x_y = torch.concat([active_cylinder_x_y, inactive_cylinders_x_y], dim=0)
        cylinders_z = torch.ones(self.num_cylinders, device=self.device).unsqueeze(-1)
        cylinders_pos = torch.concat([cylinder_x_y, cylinders_z], dim=-1)

        # init drone
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        cfg.rigid_props.max_linear_velocity = self.cfg.task.v_drone
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        self.drone.spawn(drone_pos)
        
        # init prey
        objects.DynamicSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=target_pos.unsqueeze(0),
            radius=0.05,
            color=torch.tensor([1., 0., 0.]),
            mass=1.0
        )

        # cylinders with physcical properties
        self.cylinders_size = []
        for idx in range(self.num_cylinders):
            # orientation = None
            self.cylinders_size.append(self.cylinder_size)
            objects.DynamicCylinder(
                prim_path="/World/envs/env_0/cylinder_{}".format(idx),
                name="cylinder_{}".format(idx),
                translation=cylinders_pos[idx],
                radius=self.cylinder_size,
                height=self.cylinder_height,
                mass=1000.0
            )

        objects.VisualCylinder(
            prim_path="/World/envs/env_0/Cylinder",
            name="ground",
            translation= torch.tensor([0., 0., 0.], device=self.device),
            radius=size,
            height=0.001,
            color=np.array([0.0, 0.0, 0.0]),
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
        n = self.num_agents
        init_pos, rot = self.init_poses
        self.drone._reset_idx(env_ids)

        if self.use_outer_cl:
            # TODO, generate after eval, rejection sampling for max_active
            # TODO, need to implement randomization for num_inactive
            # current_tasks contains x and y of drones, target and active_cylinders
            current_tasks, num_active_cyliders_list, _ = self.outer_curriculum_module.generate_envs(num_envs=len(env_ids))
                
        n_envs = len(env_ids)
        drone_pos = []
        cylinders_pos = []
        target_pos = []
        self.cylinders_mask = []
        occupation_matrix = []
        
        # start_time = time.time()
        for idx in range(n_envs):
            size = self.arena_size
            # random set z
            drone_z = D.Uniform(
                    torch.tensor([0.1], device=self.device),
                    torch.tensor([2 * size], device=self.device)
                ).sample((1, n)).squeeze(0)
            target_z = D.Uniform(
                torch.tensor([0.1], device=self.device),
                torch.tensor([2 * size], device=self.device)
            ).sample()
            
            # set x and y by cl or domain randomization
            if self.use_outer_cl:
                # TODO: following else to debug
                # drone pos
                drone_x_y = current_tasks[idx][:self.num_agents]
                drone_pos.append(torch.concat([drone_x_y, drone_z], dim=-1))
                # target pos
                target_x_y = current_tasks[idx][self.num_agents]
                target_pos.append(torch.concat([target_x_y, target_z], dim=-1))
                # cylinders pos
                num_active = num_active_cyliders_list[idx]
                cylinders_pos_one = current_tasks[idx][self.num_agents + 1:]
                cylinders_pos_one = torch.concat([cylinders_pos_one, cylinders_z], dim=-1)
                num_inactive = self.num_cylinders - num_active
                for inactive_idx in range(num_inactive):
                    cylinders_pos_one[inactive_idx + num_active, 2] = -1.0 * self.cylinder_height
                cylinders_pos.append(cylinders_pos_one)
                cylinder_mask_one = torch.ones(self.num_cylinders, device=self.device)
                cylinder_mask_one[num_active:] = 0.0
                # inactive_indices = torch.randperm(self.num_cylinders)[:num_inactive]
                # cylinder_mask_one[inactive_indices] = 0.0
                self.cylinders_mask.append(cylinder_mask_one)
            else:
                obj_pos, occupation_matrix_one = rejection_sampling_all_obj_xy(
                    arena_size=self.arena_size, 
                    cylinder_size=self.cylinder_size, 
                    num_drones=self.num_agents, 
                    num_cylinders=self.max_active_cylinders, 
                    device=self.device)
                occupation_matrix.append(occupation_matrix_one)
                
                drone_x_y = obj_pos[:self.num_agents].clone()
                target_x_y = obj_pos[self.num_agents].clone()
                drone_pos.append(torch.concat([drone_x_y, drone_z], dim=-1))
                target_pos.append(torch.concat([target_x_y, target_z], dim=-1))
                
                if self.random_active:
                    num_active_cylinder = np.random.randint(0, self.max_active_cylinders + 1, 1, dtype=int)[0]
                else:
                    num_active_cylinder = self.max_active_cylinders
                num_inactive = self.num_cylinders - num_active_cylinder
                active_cylinder_x_y = obj_pos[self.num_agents + 1:].clone()[:num_active_cylinder]
                inactive_cylinders_x_y = generate_outside_cylinders_x_y(arena_size=self.arena_size, 
                                                                       num_envs=1, 
                                                                       device=self.device)[:num_inactive]
                cylinder_x_y = torch.concat([active_cylinder_x_y, inactive_cylinders_x_y], dim=0)
                cylinders_z = torch.ones(self.num_cylinders, device=self.device).unsqueeze(-1)
                cylinders_pos_one = torch.concat([cylinder_x_y, cylinders_z], dim=-1)

                cylinders_pos.append(cylinders_pos_one)
                cylinder_mask_one = torch.ones(self.num_cylinders, device=self.device)
                cylinder_mask_one[num_active_cylinder:] = 0.0
                # inactive_indices = torch.randperm(self.num_cylinders)[:num_inactive]
                # cylinder_mask_one[inactive_indices] = 0.0
                self.cylinders_mask.append(cylinder_mask_one)

            if idx == self.central_env_idx and self._should_render(0):
                self._draw_court_circle(size)
        
        # end_time = time.time()
        # print('reset time', end_time - start_time)
        
        drone_pos = torch.stack(drone_pos, dim=0).type(torch.float32)
        target_pos = torch.stack(target_pos, dim=0).type(torch.float32)
        cylinders_pos = torch.stack(cylinders_pos, dim=0).type(torch.float32)
        self.cylinders_mask = torch.stack(self.cylinders_mask, dim=0).type(torch.float32) # 1 means active, 0 means inactive
           
        # set position and velocity
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), rot[env_ids], env_ids
        )
        drone_init_velocities = torch.zeros_like(self.drone.get_velocities())
        self.drone.set_velocities(torch.zeros_like(drone_init_velocities), env_ids)
        
        self.drone_sum_speed = drone_init_velocities[...,0].squeeze(-1)
        self.drone_max_speed = drone_init_velocities[...,0].squeeze(-1)

        # set target
        self.target.set_world_poses((self.envs_positions + target_pos)[env_ids], env_indices=env_ids)
        target_vel = self.target.get_velocities()
        self.target.set_velocities(2 * torch.rand_like(target_vel) - 1, self.env_ids)

        # cylinders
        self.cylinders.set_world_poses(
            (cylinders_pos + self.envs_positions[env_ids].unsqueeze(1))[env_ids], env_indices=env_ids
        )
        
        self.step_spec = 0

        # reset stats
        self.stats[env_ids] = 0.
        self.stats['catch_radius'].set_(torch.ones_like(self.stats['catch_radius'], device=self.device) * self.catch_radius)
        self.stats['v_prey'].set_(torch.ones_like(self.stats['v_prey'], device=self.device) * self.v_prey)
        self.stats['first_capture_step'].set_(torch.ones_like(self.stats['first_capture_step']) * self.max_episode_length)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        self.step_spec += 1
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)
        
        target_vel = self.target.get_velocities()
        forces_target = self._get_dummy_policy_prey()
        
        # fixed velocity
        target_vel[:,:3] = self.v_prey * forces_target / (torch.norm(forces_target, dim=1).unsqueeze(1) + 1e-5)
        
        self.target.set_velocities(target_vel.type(torch.float32), self.env_ids)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        self.info["drone_state"][:] = self.drone_states[..., :13]
        drone_pos = self.drone_states[..., :3]
        drone_vel = self.drone.get_velocities()
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)
        
        # draw drone trajectory and detection range
        if self._should_render(0):
            self._draw_traj()
            if self.detect_range > 0.0:
                self._draw_detection()       

        drone_speed_norm = torch.norm(drone_vel[..., :3], dim=-1)
        self.drone_sum_speed += drone_speed_norm
        self.drone_max_speed = torch.max(torch.stack([self.drone_max_speed, drone_speed_norm], dim=-1), dim=-1).values

        # record stats
        self.stats['drone1_speed_per_step'].set_(self.drone_sum_speed[:,0].unsqueeze(-1) / self.step_spec)
        self.stats['drone2_speed_per_step'].set_(self.drone_sum_speed[:,1].unsqueeze(-1) / self.step_spec)
        self.stats['drone3_speed_per_step'].set_(self.drone_sum_speed[:,2].unsqueeze(-1) / self.step_spec)
        self.stats['drone1_max_speed'].set_(self.drone_max_speed[:,0].unsqueeze(-1))
        self.stats['drone2_max_speed'].set_(self.drone_max_speed[:,1].unsqueeze(-1))
        self.stats['drone3_max_speed'].set_(self.drone_max_speed[:,2].unsqueeze(-1))

        # get target position and velocity        
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        target_pos = target_pos.unsqueeze(1) # [N, 1, 3]
        target_vel = self.target.get_velocities()
        target_vel = target_vel.unsqueeze(1) # [N, 1, 6]
        self.stats["prey_speed"].set_(torch.norm(target_vel.squeeze(1)[:, :3], dim=-1).unsqueeze(-1))

        # get masked target relative position and velocity
        target_rpos = target_pos - drone_pos # [N, n, 3]
        target_rvel = target_vel - drone_vel # [N, n, 6]
        if self.detect_range < 0.0:
            target_mask = torch.zeros_like(target_rpos[...,0], dtype=bool)
        else:
            target_mask = torch.norm(target_rpos, dim=-1) > self.detect_range # [N, n]
            target_mask = target_mask.all(1).unsqueeze(1).expand_as(target_mask) # [N, n]

        target_pmask = target_mask.unsqueeze(-1).expand_as(target_rpos) # [N, n, 3]
        target_vmask = target_mask.unsqueeze(-1).expand_as(target_rvel) # [N, n, 6]
        target_rpos_masked = target_rpos.clone()
        target_rpos_masked.masked_fill_(target_pmask, self.mask_value)
        target_rvel_masked = target_rvel.clone()
        target_rvel_masked.masked_fill_(target_vmask, self.mask_value)

        # get full target state
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            target_state = torch.cat([
                target_pos,
                target_vel,
                t.expand(-1, self.time_encoding_dim).unsqueeze(1)
            ], dim=-1) # [num_envs, 1, 9+time_encoding_dim]
        else:
            target_state = torch.cat([
                target_pos,
                target_vel
            ], dim=-1) # [num_envs, 1, 9]

        # identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        obs["state_self"] = torch.cat(
            [-target_rpos_masked,
             -target_rvel_masked,
             self.drone_states, 
            #  identity
             ], dim=-1
        ).unsqueeze(2)

        obs["state_others"] = self.drone_rpos

        frame_state = target_state.unsqueeze(1).expand(-1, self.drone.n, -1, -1)
        obs["state_frame"] = frame_state

        # get masked cylinder relative position
        cylinders_pos, _ = self.get_env_poses(self.cylinders.get_world_poses())
        cylinders_rpos = vmap(cpos)(drone_pos, cylinders_pos) # [N, n, num_cylinders, 3]
        cylinders_height = torch.tensor([self.cylinder_height for _ in range(self.num_cylinders)], 
                                        device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(
                                            self.num_envs, self.drone.n, -1, -1)
        cylinders_radius = torch.tensor([self.cylinder_size for _ in range(self.num_cylinders)],
                                        device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(
                                            self.num_envs, self.drone.n, -1, -1)
        cylinders_state = torch.concat([
            cylinders_rpos,
            cylinders_height,
            cylinders_radius
        ], dim=-1)
        
        cylinders_mdist_z = torch.abs(cylinders_rpos[..., 2]) - cylinders_height.squeeze(-1) / 2
        cylinders_mdist_xy = torch.norm(cylinders_rpos[..., :2], dim=-1) - cylinders_radius.squeeze(-1)
        cylinders_mdist = torch.stack([torch.max(cylinders_mdist_xy, torch.zeros_like(cylinders_mdist_xy)), 
                                       torch.max(cylinders_mdist_z, torch.zeros_like(cylinders_mdist_z))
                                       ], dim=-1)
        if self.detect_range < 0.0:
            cylinders_mask = torch.zeros_like(cylinders_mdist[...,0], dtype=bool)
        else:
            cylinders_mask = torch.norm(cylinders_mdist, dim=-1) > self.detect_range
            cylinders_mask = cylinders_mask.all(1).unsqueeze(1).expand_as(cylinders_mask)
        # add physical mask, for inactive cylinders
        cylinders_inactive_mask = ~ self.cylinders_mask.unsqueeze(1).expand(-1, self.num_agents, -1).type(torch.bool)
        cylinders_mask = cylinders_mask + cylinders_inactive_mask
        cylinders_smask = cylinders_mask.unsqueeze(-1).expand(-1, -1, -1, 5)
        cylinders_state_masked = cylinders_state.clone()
        cylinders_state_masked.masked_fill_(cylinders_smask, self.mask_value)
        obs["cylinders"] = cylinders_state_masked

        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = torch.cat(
            [-target_rpos,
             -target_rvel,
             self.drone_states, 
            #  identity
             ], dim=-1
        )   # [num_envs, drone.n, drone_state_dim]
        state["state_frame"] = target_state                # [num_envs, 1, target_rpos_dim]
        state["cylinders"] = cylinders_state
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
        drone_pos, _ = self.drone.get_world_poses()
        target_pos, _ = self.target.get_world_poses()
        target_pos = target_pos.unsqueeze(1)

        target_dist = torch.norm(target_pos - drone_pos, dim=-1)

        capture_flag = (target_dist < self.catch_radius)
        self.stats['capture_episode'].add_(torch.sum(capture_flag, dim=1).unsqueeze(-1))
        self.stats['capture'].set_(torch.from_numpy(self.stats['capture_episode'].to('cpu').numpy() > 0.0).type(torch.float32).to(self.device))
        # capture_eval = self.stats['capture'].reshape(self.task_space_len, self.eval_num_envs, -1)
        # capture_eval = capture_eval.mean(dim=1)
        # self.stats['cover_rate'].set_((torch.sum(capture_eval >= 0.95) / self.task_space_len).unsqueeze(-1).expand_as(self.stats['capture']))
        self.stats['capture_per_step'].set_(self.stats['capture_episode'] / self.step_spec)
        # catch_reward = 10 * capture_flag.sum(-1).unsqueeze(-1).expand_as(capture_flag)
        catch_reward = 10 * capture_flag.type(torch.float32)
        catch_flag = torch.any(catch_reward, dim=1).unsqueeze(-1)
        self.stats['first_capture_step'][catch_flag * (self.stats['first_capture_step'] >= self.step_spec)] = self.step_spec

        # speed penalty
        if self.cfg.task.use_speed_penalty:
            drone_vel = self.drone.get_velocities()
            drone_speed_norm = torch.norm(drone_vel[..., :3], dim=-1)
            speed_reward = - 100 * (drone_speed_norm > self.cfg.task.v_drone)
        else:
            speed_reward = 0.0

        # collison with cylinders
        coll_reward = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        
        cylinders_pos, _ = self.cylinders.get_world_poses()
        for i in range(self.num_cylinders):
            relative_pos = drone_pos[..., :2] - cylinders_pos[:, i, :2].unsqueeze(-2)
            norm_r = torch.norm(relative_pos, dim=-1)
            if_coll = (norm_r < (self.collision_radius + self.cylinders_size[i])).type(torch.float32)
            tmp_cylinder_mask = self.cylinders_mask[:, i].unsqueeze(-1).expand(-1, self.num_agents)
            coll_reward -= if_coll * tmp_cylinder_mask # sparse

        # distance reward
        min_dist = target_dist
        dist_reward_mask = (min_dist > self.catch_radius)
        distance_reward = - 1.0 * min_dist * dist_reward_mask
        if self.cfg.task.use_collision:
            reward = speed_reward + 1.0 * catch_reward + 1.0 * distance_reward + 5 * coll_reward
        else:
            reward = speed_reward + 1.0 * catch_reward + 1.0 * distance_reward
        
        self._tensordict["return"] += reward.unsqueeze(-1)
        self.returns = self._tensordict["return"].sum(1)
        self.stats["return"].set_(self.returns)

        done  = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        )
        
        self.progress_std = torch.std(self.progress_buf)

        return TensorDict({
            "agents": {
                "reward": reward.unsqueeze(-1)
            },
            "done": done,
        }, self.batch_size)
        
    def _get_dummy_policy_prey(self):
        pos, _ = self.drone.get_world_poses(False)
        prey_pos, _ = self.target.get_world_poses()
        prey_pos = prey_pos.unsqueeze(1)
        
        force = torch.zeros(self.num_envs, 3, device=self.device)

        # predators
        # active mask : if drone is failed, do not get force from it
        drone_vel = self.drone.get_velocities()
        active_mask = (torch.norm(drone_vel[...,:3],dim=-1) > 1e-5).unsqueeze(-1).expand(-1,-1,3)
        prey_pos_all = prey_pos.expand(-1,self.num_agents,-1)
        dist_pos = torch.norm(prey_pos_all - pos,dim=-1).unsqueeze(-1).expand(-1,-1,3)
        direction_p = (prey_pos_all - pos) / (dist_pos + 1e-5)
        # force_p = direction_p * (1 / (dist_pos + 1e-5)) * active_mask
        force_p = direction_p * (1 / (dist_pos + 1e-5))
        force += torch.sum(force_p, dim=1)

        # arena
        # 3D
        prey_env_pos, _ = self.get_env_poses(self.target.get_world_poses())
        force_r = torch.zeros_like(force)
        prey_origin_dist = torch.norm(prey_env_pos[:, :2],dim=-1)
        force_r[..., 0] = - prey_env_pos[:,0] / ((self.arena_size - prey_origin_dist)**2 + 1e-5)
        force_r[..., 1] = - prey_env_pos[:,1] / ((self.arena_size - prey_origin_dist)**2 + 1e-5)
        force_r[...,2] += 1 / (prey_env_pos[:,2] - 0 + 1e-5) - 1 / (2 * self.arena_size - prey_env_pos[:,2] + 1e-5)
        force += force_r
        
        # cylinders
        cylinders_pos, _ = self.cylinders.get_world_poses()
        dist_pos = torch.norm(prey_pos[..., :3] - cylinders_pos[..., :3],dim=-1).unsqueeze(-1).expand(-1, -1, 3) # expand to 3-D
        direction_c = (prey_pos[..., :3] - cylinders_pos[..., :3]) / (dist_pos + 1e-5)
        force_c = direction_c * (1 / (dist_pos + 1e-5))
        cylinder_force_mask = self.cylinders_mask.unsqueeze(-1).expand(-1, -1, 3)
        force_c = force_c * cylinder_force_mask
        force[..., :3] += torch.sum(force_c, dim=1)

        # set force_z to 0
        return force.type(torch.float32)
    
    # visualize functions
    def _draw_court_circle(self, size):
        self.draw.clear_lines()

        point_list_1, point_list_2, colors, sizes = draw_court_circle(
            size, 2 * size, line_size=5.0
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
            drange=self.detect_range,
        )
        point_list = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list
        ]
        self.draw.draw_points(point_list, colors, sizes)
    