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

from .draw import draw_traj, draw_catch
from .draw_circle import Float3, _COLOR_ACCENT, _carb_float3_add, draw_court_circle


# drones on land by default
# only cubes are available as walls

class CurriculumBuffer(object):
    def __init__(self,):
        self.eps = 1e-10
        self.random_task_space = True

        self._state_buffer = np.zeros((0, 1), dtype=np.float32)
        self._weight_buffer = np.zeros((0, 1), dtype=np.float32)
        self._task_space = dict(size=0, speed=0)
        self._naive_task_space = []
        self._temp_state_buffer = []
        self._moving_max = 0.0
        self._moving_min = 0.0

    def get_eval_task(self, eval_num_envs):
        task_space = []
        speed_len = len(self._task_space['speed'])
        size_len = len(self._task_space['size'])
        if speed_len == 1 or size_len == 1:
            return np.stack([np.array([self._task_space['speed'][0], self._task_space['size'][0]])] * eval_num_envs, axis=0)
        for speed_idx in range(speed_len - 1):
            for size_idx in range(size_len - 1):
                speed = np.random.uniform(self._task_space['speed'][speed_idx], self._task_space['speed'][speed_idx + 1])
                size = np.random.uniform(self._task_space['size'][size_idx], self._task_space['size'][size_idx + 1])
                task_space.append(np.repeat(np.expand_dims(np.array([speed, size]), axis=0), eval_num_envs, axis=0))
        task_space = np.concatenate(task_space, axis=0)
        return task_space

    def insert(self, states):
        """
        input:
            states: list of np.array(size=(state_dim, ))
            weight: list of np.array(size=(1, ))
        """
        self._temp_state_buffer.append(copy.deepcopy(states))

    def update_states(self):
        # concatenate to get all states
        all_states = []
        if len(self._temp_state_buffer) > 0:
            all_states = np.concatenate(self._temp_state_buffer, axis=0)

        # update
        if len(all_states) > 0:
            self._state_buffer = copy.deepcopy(all_states)
        # reset temp state and weight buffer
        self._temp_state_buffer = []

        return self._state_buffer.copy()

    def update_weights(self, weights):
        self._weight_buffer = weights.copy()

    def sample(self, num_samples):
        """
        return list of np.array
        """
        if self._state_buffer.shape[0] == 0:  # state buffer is empty
            initial_states = [None for _ in range(num_samples)]
        else:
            weights = self._weight_buffer / np.mean(self._weight_buffer)
            probs = weights / np.sum(weights)
            sample_idx = np.random.choice(self._state_buffer.shape[0], num_samples, replace=True, p=probs)
            initial_states = [self._state_buffer[idx] for idx in sample_idx]
        return initial_states
    
    def save_task(self, model_dir, episode):
        np.save('{}/tasks_{}.npy'.format(model_dir,episode), self._state_buffer)
        np.save('{}/scores_{}.npy'.format(model_dir,episode), self._weight_buffer)

    # for update_metric = greedy
    def _buffer_sort(self, list1, *args): # sort by list1, ascending order
        zipped = zip(list1,*args)
        sort_zipped = sorted(zipped,key=lambda x:(x[0],np.mean(x[1])))
        result = zip(*sort_zipped)
        return [list(x) for x in result]

class HideAndSeek_circle_static(IsaacEnv): 
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

        _update_curriculum(self, capture):
            Determine whether the capture result matches requirements and 
            insert corresponding states and weights into curriculum buffer

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
            shape=[self.num_envs, -1],
            # track_contact_forces=True
        )
        self.cylinders.initialize()
        # breakpoint()
        
        self.time_encoding = self.cfg.task.time_encoding

        self.target_init_vel = self.target.get_velocities(clone=True)
        self.env_ids = torch.from_numpy(np.arange(0, cfg.env.num_envs))
        self.size_min = self.cfg.task.size_min
        self.size_max = self.cfg.task.size_max
        self.size_dist = D.Uniform(
            torch.tensor([self.size_min], device=self.device),
            torch.tensor([self.size_max], device=self.device)
        )
        self.returns = self.progress_buf * 0
        self.catch_radius = self.cfg.task.catch_radius
        self.collision_radius = self.cfg.task.collision_radius
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.v_low = self.cfg.task.v_drone * self.cfg.task.v_low
        self.v_high = self.cfg.task.v_drone * self.cfg.task.v_high
        
        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )

        self.mask_value = -1.0

        # CL
        self.curriculum_buffer = CurriculumBuffer()
        self.sigma_min = 0.5
        self.sigma_max = 0.95
        self.prob_curriculum = self.cfg.prob_curriculum
        self.prob_decay = self.cfg.prob_decay
        self.set_train = True
        self.update_iter = 0
        self.max_iters = self.cfg.max_iters
        self.use_dynamic = self.cfg.task.use_dynamic
        self.fixed_config = self.cfg.task.fixed_config
        
        # init task_space
        speed_num = 1 if self.v_low == self.v_high else 5
        task_speed = np.linspace(start = self.v_low, stop = self.v_high, num = speed_num, endpoint=True)
        self.curriculum_buffer._task_space['speed'] = task_speed
        
        size_num = 1 if self.size_min == self.size_max else 5
        task_size = np.linspace(start = self.size_min, stop = self.size_max, num = size_num, endpoint=True)
        self.curriculum_buffer._task_space['size'] = task_size
 
        self.task_space_len = (speed_num - 1) * (size_num - 1)
        if self.task_space_len == 0:
            self.task_space_len = 1
                    
        # init phase
        start_speed, start_size = self.v_low, self.size_min
        self.curriculum_buffer.insert(np.array([[start_speed, start_size]]))
        self.curriculum_buffer.update_states()
        weights = np.ones(len(self.curriculum_buffer._state_buffer), dtype=np.float32)
        self.curriculum_buffer.update_weights(weights)
        
        self.draw = _debug_draw.acquire_debug_draw_interface()

    def _set_specs(self):        
        drone_state_dim = self.drone.state_spec.shape.numel()
        frame_state_dim = 9 # target_pos_dim + target_vel
        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            frame_state_dim += self.time_encoding_dim        

        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, 3 + 6 + drone_state_dim + self.drone.n)),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, 3)), # pos
            "state_frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
            "cylinders": UnboundedContinuousTensorSpec((self.num_cylinders, 5)), # pos + radius + height
        }).to(self.device)
        state_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, 3 + 6 + drone_state_dim + self.drone.n)),
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
            "cover_rate": UnboundedContinuousTensorSpec(1),
            "p": UnboundedContinuousTensorSpec(1),
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
        
    def _design_scene(self):
        self.num_agents = self.cfg.task.num_agents
        self.num_cylinders = self.cfg.task.cylinder.num
        self.size_min = self.cfg.task.size_min
        self.size_max = self.cfg.task.size_max

        # init drone
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        cfg.rigid_props.max_linear_velocity = self.cfg.task.v_drone
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        translation = torch.zeros(self.num_agents, 3)
        translation[:, 0] = torch.arange(self.num_agents)
        translation[:, 1] = torch.arange(self.num_agents)
        translation[:, 2] = 0.5
        self.drone.spawn(translation)
        
        # init prey
        self.target_pos = torch.tensor([[0., 0.05, 0.5]], device=self.device)
        objects.DynamicSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=self.target_pos,
            radius=0.05,
            # height=0.1,
            color=torch.tensor([1., 0., 0.]),
            mass=1.0
        )

        size_dist = D.Uniform(
            torch.tensor([self.size_min], device=self.device),
            torch.tensor([self.size_max], device=self.device)
        )
        size = size_dist.sample().item()
        random_pos_dist = D.Uniform(
            torch.tensor([-size, -size, 0.0], device=self.device),
            torch.tensor([size, size, 0.0], device=self.device)
        )

        self.cylinders_prims = [None] * self.num_cylinders
        self.cylinders_size = []
        for idx in range(self.num_cylinders):
            cyl = self.cfg.task.cylinder['cyl{}'.format(idx)]
            translation = orientation = None
            if 'translation' in cyl:
                translation = (cyl.translation.x, cyl.translation.y, cyl.translation.z)
            if 'orientation' in cyl:
                orientation = (cyl.orientation.qw, cyl.orientation.qx, cyl.orientation.qy, cyl.orientation.qz)
            attributes = {'axis': cyl.axis, 'radius': cyl.radius, 'height': cyl.height}
            self.cylinders_size.append(cyl.radius)
            self.cylinders_prims[idx] = create_obstacle(
                "/World/envs/env_0/cylinder_{}".format(idx), 
                prim_type="Cylinder",
                translation=translation,
                orientation=orientation,
                attributes=attributes
            ) # Use 'self.cylinders_prims[0].GetAttribute('radius').Get()' to get attributes
            
        objects.VisualCuboid(
            prim_path="/World/envs/env_0/ground",
            name="ground",
            translation= torch.tensor([0., 0., 0.], device=self.device),
            scale=torch.tensor([size * 2, size * 2, 0.001], device=self.device),
            color=torch.tensor([0., 0., 0.]),
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

        # CL : training distribution
        if self.set_train:
            sample_p = max(0.0, self.prob_curriculum * (1 - self.prob_decay * self.update_iter / self.max_iters))
            # sample_p = self.prob_curriculum
            self.stats['p'] = sample_p * torch.ones(size=(self.num_envs,1)).to(self.device)
            use_curriculum = (np.random.uniform(size=self.num_envs) < sample_p)
            set_idx = np.arange(self.num_envs)[use_curriculum]
            initial_states = self.curriculum_buffer.sample(len(set_idx))
            initial_states += [None] * (self.num_envs - len(set_idx))
            self.eval_num_envs = self.num_envs // self.task_space_len if self.task_space_len > 0 else self.num_envs
        else:
            if self.use_dynamic:
                self.eval_num_envs = self.num_envs // self.task_space_len if self.task_space_len > 0 else self.num_envs
                initial_states = [np.array([self.v_low, self.size_min])] * self.num_envs
            else:
                if self.fixed_config:
                    self.eval_num_envs = self.num_envs // self.task_space_len if self.task_space_len > 0 else self.num_envs
                    initial_states = [np.array([self.v_high, self.size_max])] * self.num_envs
                else:
                    self.eval_num_envs = self.num_envs // self.task_space_len if self.task_space_len > 0 else self.num_envs
                    self.initial_states = self.curriculum_buffer.get_eval_task(eval_num_envs=self.eval_num_envs)
                    initial_states = self.initial_states.tolist()
                    initial_states += [None] * (self.num_envs - len(initial_states))

        n_envs = len(env_ids)
        drone_pos = []
        cylinder_pos = []
        target_pos = []
        self.v_prey = []
        self.size_list = []
        # reset size
        for idx in range(n_envs):            
            if initial_states[idx] is not None:
                prey_speed = initial_states[idx][0]
                size = initial_states[idx][1]
            else:
                if self.fixed_config:
                    prey_speed = self.v_high
                    size = self.size_max
                else:
                    prey_speed = np.random.uniform(self.v_low, self.v_high)
                    size = self.size_dist.sample().item()
            
            self.v_prey.append(prey_speed)
            self.size_list.append(size)
        
            drone_pos_dist = D.Uniform(
                torch.tensor([-size, -size, 0.0], device=self.device),
                torch.tensor([size, size, 2 * size], device=self.device)
            )
            drone_pos.append(drone_pos_dist.sample((1,n)))

            target_pos_dist = D.Uniform(
                torch.tensor([-size, -size, 0.0], device=self.device),
                torch.tensor([size, size, 2 * size], device=self.device)
            )
            target_pos.append(target_pos_dist.sample())

            cylinder_pos_temp = torch.tensor([], device=self.device)
            for cyl_idx in range(self.num_cylinders):
                cyl = self.cfg.task.cylinder['cyl{}'.format(cyl_idx)]
                translation = [[[cyl.translation.x, cyl.translation.y, cyl.translation.z]]]
                cylinder_pos_temp = torch.concat(
                    (cylinder_pos_temp, torch.tensor(translation, device=self.device)), dim=1)
            cylinder_pos.append(cylinder_pos_temp)

            if idx == self.central_env_idx and self._should_render(0):
                self._draw_court_circle(size)

        drone_pos = torch.concat(drone_pos, dim=0).type(torch.float32)
        target_pos = torch.stack(target_pos, dim=0).type(torch.float32)
        cylinder_pos = torch.concat(cylinder_pos, dim=0).type(torch.float32)
        self.v_prey = torch.Tensor(np.array(self.v_prey)).to(self.device)
        self.size_list = torch.Tensor(np.array(self.size_list)).to(self.device)
        # set position and velocity
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), rot[env_ids], env_ids
        )
        drone_init_velocities = torch.zeros_like(self.drone.get_velocities())
        self.drone.set_velocities(torch.zeros_like(drone_init_velocities), env_ids)
        
        self.drone_sum_speed = drone_init_velocities[...,0].squeeze(-1)
        self.drone_max_speed = drone_init_velocities[...,0].squeeze(-1)
        self.eval_drone_sum_speed = drone_init_velocities[...,0].squeeze(-1)
        self.eval_drone_max_speed = drone_init_velocities[...,0].squeeze(-1)

        # set target
        self.target.set_world_poses((self.envs_positions + target_pos)[env_ids], env_indices=env_ids)
        target_vel = self.target.get_velocities()
        self.target.set_velocities(2 * torch.rand_like(target_vel) - 1, self.env_ids)

        # cylinders
        self.cylinders.set_world_poses(
            (cylinder_pos + self.envs_positions[env_ids].unsqueeze(1))[env_ids], env_indices=env_ids
        )
        
        self.step_spec = 0

        # reset stats
        self.stats[env_ids] = 0.   

    def _update_curriculum(self, capture):
        capture = capture.reshape(self.task_space_len,-1) # [eval_num_envs, task_space_len]
        capture = np.mean(capture, axis=1) # [task_space_len]
        eval_tasks = self.initial_states.reshape(self.task_space_len, self.eval_num_envs, -1)
        for capture_idx in range(len(capture)):
            print('task', eval_tasks[capture_idx][0], 'capture', capture[capture_idx])
            if capture[capture_idx] >= self.sigma_min and capture[capture_idx] <= self.sigma_max:
                self.curriculum_buffer.insert(np.array(eval_tasks[capture_idx][0])[np.newaxis,:])
        # eval and save tasks into CL buffer
        self.curriculum_buffer.update_states()
        weights = np.ones(len(self.curriculum_buffer._state_buffer), dtype=np.float32)
        self.curriculum_buffer.update_weights(np.array(weights))

    def _pre_sim_step(self, tensordict: TensorDictBase):
        if self.use_dynamic:
            if self.step_spec == self.max_episode_length // 2:
                self.v_prey = [self.v_high] * self.num_envs
                self.size_list = [self.size_max] * self.num_envs
                self.v_prey = torch.Tensor(np.array(self.v_prey)).to(self.device)
                self.size_list = torch.Tensor(np.array(self.size_list)).to(self.device)
        
        self.step_spec += 1
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)
        
        target_vel = self.target.get_velocities()
        forces_target = self._get_dummy_policy_prey()
        
        # fixed velocity
        target_vel[:,:3] = self.v_prey.unsqueeze(1) * forces_target / (torch.norm(forces_target, dim=1).unsqueeze(1) + 1e-5)
        
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
            self._draw_catch()            

        drone_speed_norm = torch.norm(drone_vel[..., :3], dim=-1)
        if self.set_train:
            self.drone_sum_speed += drone_speed_norm
            self.drone_max_speed = torch.max(torch.stack([self.drone_max_speed, drone_speed_norm], dim=-1), dim=-1).values
        else:
            self.eval_drone_sum_speed += drone_speed_norm
            self.eval_drone_max_speed = torch.max(torch.stack([self.eval_drone_max_speed, drone_speed_norm], dim=-1), dim=-1).values
        
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

        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        obs["state_self"] = torch.cat(
            [-target_rpos,
             -target_rvel,
             self.drone_states, 
             identity], dim=-1
        ).unsqueeze(2)

        obs["state_others"] = self.drone_rpos

        frame_state = target_state.unsqueeze(1).expand(-1, self.drone.n, -1, -1)
        obs["state_frame"] = frame_state

        # get masked cylinder relative position
        cylinders_pos, _ = self.get_env_poses(self.cylinders.get_world_poses())
        cylinders_rpos = vmap(cpos)(drone_pos, cylinders_pos) # [N, n, num_cylinders, 3]
        cylinders_height = torch.tensor([prim.GetAttribute('height').Get() for prim in self.cylinders_prims], 
                                        device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(
                                            self.num_envs, self.drone.n, -1, -1)
        cylinders_radius = torch.tensor([prim.GetAttribute('radius').Get() for prim in self.cylinders_prims],
                                        device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(
                                            self.num_envs, self.drone.n, -1, -1)
        cylinders_state = torch.concat([
            cylinders_rpos,
            cylinders_height,
            cylinders_radius
        ], dim=-1)
        
        obs["cylinders"] = cylinders_state
        # breakpoint()

        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = torch.cat(
            [-target_rpos,
             -target_rvel,
             self.drone_states, 
             identity], dim=-1
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
        capture_eval = self.stats['capture'].reshape(self.task_space_len, self.eval_num_envs, -1)
        capture_eval = capture_eval.mean(dim=1)
        self.stats['cover_rate'].set_((torch.sum(capture_eval >= 0.95) / self.task_space_len).unsqueeze(-1).expand_as(self.stats['capture']))
        self.stats['capture_per_step'].set_(self.stats['capture_episode'] / self.step_spec)
        # catch_reward = 10 * capture_flag.sum(-1).unsqueeze(-1).expand_as(capture_flag)
        catch_reward = 10 * capture_flag.type(torch.float32)

        # speed penalty
        if self.cfg.task.use_speed_penalty:
            drone_vel = self.drone.get_velocities()
            drone_speed_norm = torch.norm(drone_vel[..., :3], dim=-1)
            speed_reward = - 100 * (drone_speed_norm > self.cfg.task.v_drone)
        else:
            speed_reward = 0.0

        # collison with cylinders
        coll_reward = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        
        cylinder_pos, _ = self.cylinders.get_world_poses()
        for i in range(self.num_cylinders):
            relative_pos = drone_pos[..., :2] - cylinder_pos[:, i, :2].unsqueeze(-2)
            norm_r = torch.norm(relative_pos, dim=-1)
            if_coll = (norm_r < (self.collision_radius + self.cylinders_size[i])).type(torch.float32)
            coll_reward -= if_coll # sparse

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
        force_r[..., 0] = - prey_env_pos[:,0] / ((self.size_list - prey_origin_dist)**2 + 1e-5)
        force_r[..., 1] = - prey_env_pos[:,1] / ((self.size_list - prey_origin_dist)**2 + 1e-5)
        # force_r[...,0] = 1 / (prey_env_pos[:,0] - (- self.size_list) + 1e-5) - 1 / (self.size_list - prey_env_pos[:,0] + 1e-5)
        # force_r[...,1] = 1 / (prey_env_pos[:,1] - (- self.size_list) + 1e-5) - 1 / (self.size_list - prey_env_pos[:,1] + 1e-5)
        force_r[...,2] += 1 / (prey_env_pos[:,2] - 0 + 1e-5) - 1 / (2 * self.size_list - prey_env_pos[:,2] + 1e-5)
        force += force_r
        
        # set force_z to 0
        return force.type(torch.float32)
    
    # visualize functions
    def _draw_court_circle(self, size):
        self.draw.clear_lines()

        point_list_1, point_list_2, colors, sizes = draw_court_circle(
            size, 2*size, line_size=5.0
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
    
    def _draw_catch(self):
        self.draw.clear_points()

        drone_pos = self.drone_states[..., :3]
        drone_ori = self.drone_states[..., 3:7]
        drone_xaxis = quat_axis(drone_ori, 0)
        drone_yaxis = quat_axis(drone_ori, 1)
        drone_zaxis = quat_axis(drone_ori, 2)
        point_list, colors, sizes = draw_catch(
            pos=drone_pos[self.central_env_idx, :],
            xaxis=drone_xaxis[self.central_env_idx, 0, :],
            yaxis=drone_yaxis[self.central_env_idx, 0, :],
            zaxis=drone_zaxis[self.central_env_idx, 0, :],
            drange=self.catch_radius,
        )
        point_list = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list
        ]
        self.draw.draw_points(point_list, colors, sizes)
    