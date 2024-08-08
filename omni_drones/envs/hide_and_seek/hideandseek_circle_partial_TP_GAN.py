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
from .draw import draw_traj, draw_detection, draw_catch, draw_court
from .draw_circle import Float3, _COLOR_ACCENT, _carb_float3_add, draw_court_circle
import time
import collections
from omni_drones.learning import TP_net
from omni_drones.envs.hide_and_seek.gan.lsgan import LSGAN
import math

# *********check whether the capture is blocked***************
def is_perpendicular_line_intersecting_segment(a, b, c):
    # a: drones, b: target, c: cylinders
    
    # the direction of ab
    dx = b[:, :, 0] - a[:, :, 0]  # [batch, num_drones]
    dy = b[:, :, 1] - a[:, :, 1]  # [batch, num_drones]
    
    # c to ab, cd is perpendicular to ab
    num = (c[:, :, 0].unsqueeze(1) - a[:, :, 0].unsqueeze(2)) * dx.unsqueeze(2) + \
          (c[:, :, 1].unsqueeze(1) - a[:, :, 1].unsqueeze(2)) * dy.unsqueeze(2)  # [batch, num_drones, num_cylinders]
    
    denom = dx.unsqueeze(2)**2 + dy.unsqueeze(2)**2  # [batch, num_drones, 1]
    
    t = num / denom  # [batch, num_drones, num_cylinders]
    
    # check d in or not in ab
    is_on_segment = (t >= 0) & (t <= 1)  # [batch, num_drones, num_cylinders]
    
    return is_on_segment

def is_line_blocked_by_cylinder(drone_pos, target_pos, cylinder_pos, cylinder_size):
    '''
        # only consider cylinders on the ground
        # 1. compute_reward: for catch reward, not blocked
        # 2. compute_obs: for drones' state, mask the target state in the shadow
        # 3. dummy_prey_policy: if not blocked, the target gets force from the drone
    '''
    # drone_pos: [num_envs, num_agents, 3]
    # target_pos: [num_envs, 1, 3]
    # cylinder_pos: [num_envs, num_cylinders, 3]
    # consider the x-y plane, the distance of c to the line ab
    # d = abs((x2 - x1)(y3 - y1) - (y2 - y1)(x3 - x1)) / sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    batch, num_agents, _ = drone_pos.shape
    _, num_cylinders, _ = cylinder_pos.shape
    
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
    
    # whether the cylinder between the drone and the target
    flag = is_perpendicular_line_intersecting_segment(drone_pos, target_pos, cylinder_pos)
    
    # cylinders on the ground
    on_ground = (cylinder_pos[..., -1] > 0.0).unsqueeze(1).expand(-1, num_agents, num_cylinders)
    
    blocked = blocked * flag * on_ground

    return blocked.any(dim=(-1))

# *************grid initialization****************
def select_unoccupied_positions(occupancy_matrix, num_objects):
    # num_obstacles: max_num, include the inactive cylinders
    batch_size, height, width = occupancy_matrix.shape
    all_chosen_coords = []
    for i in range(batch_size):
        available_coords = torch.nonzero(occupancy_matrix[i] == 0, as_tuple=False)
        if available_coords.size(0) < num_objects:
            raise ValueError(f"Not enough available coordinates in batch {i} to choose from")

        chosen_coords = available_coords[torch.randperm(available_coords.size(0))[:num_objects]]
        
        all_chosen_coords.append(chosen_coords)
        
    return torch.stack(all_chosen_coords) 

def select_unoccupied_positions_GAN(occupancy_matrix, candidate_cylinder_grid):
    # cylinder_grid: all cylidners generated by GAN, cylinders can cover each other
    batch_size, _, _ = occupancy_matrix.shape
    all_chosen_coords = []
    active_cylinders_list = []
    for i in range(batch_size):
        wall_occupancy_matrix = occupancy_matrix[i].clone()
        # flatten index_matrix
        flat_indices = candidate_cylinder_grid[i].view(-1, 2).to(torch.long)
        # set occupied grid by cylinders
        if len(flat_indices) > 0:
            occupancy_matrix[i][flat_indices[:, 0], flat_indices[:, 1]] = 1
        # diff means real cylinders, no duplication, not out of the circle, not on the drones and target
        cylinders_coords = torch.nonzero((occupancy_matrix[i] - wall_occupancy_matrix) == 1, as_tuple=False)
        active_cylinders = cylinders_coords.shape[0]
        active_cylinders_list.append(active_cylinders)
        # reset real cylinders, only use grid[0:active_cylinders]
        candidate_cylinder_grid[i][:active_cylinders] = cylinders_coords
        all_chosen_coords.append(candidate_cylinder_grid[i])
    
    active_cylinders_list = torch.tensor(active_cylinders_list)

    return torch.stack(all_chosen_coords), active_cylinders_list

def grid_to_continuous(grid_coords, boundary, grid_size, center_pos, center_grid):
    """
    Convert grid coordinates to continuous coordinates.
    
    Args:
    grid_center (torch.Tensor): A 2D tensor of shape (2,) representing the center coordinates of the grid.
    grid_size (float): The size of each grid cell.
    grid_coords (torch.Tensor): A 2D tensor of shape (num_agents, 2) containing the grid coordinates.
    
    Returns:
    torch.Tensor: A 2D tensor of shape (num_agents, 2) containing the continuous coordinates.
    """
    # Calculate the offset from the center of the grid
    offset = (grid_coords - center_grid) * grid_size
    
    # Add the offset to the center coordinates to get the continuous coordinates
    continuous_coords = center_pos + offset
    
    # sanity check, inside
    continuous_coords = torch.clamp(continuous_coords, -boundary, boundary)
    
    return continuous_coords

def continuous_to_grid(continuous_coords, num_grid, grid_size, center_pos, center_grid):
    """
    Convert continuous coordinates to grid coordinates.
    
    Args:
    continuous_coords (torch.Tensor): A 2D tensor of shape (num_agents, 2) containing the continuous coordinates.
    grid_size (float): The size of each grid cell.
    grid_center (torch.Tensor): A 2D tensor of shape (2,) representing the center coordinates of the grid.
    
    Returns:
    torch.Tensor: A 2D tensor of shape (num_agents, 2) containing the grid coordinates.
    """
    # Calculate the offset from the center of the grid
    offset = continuous_coords - center_pos
    
    # Convert the offset to grid coordinates
    grid_coords = torch.round(offset / grid_size).int() + center_grid
    
    # sanity check
    grid_coords = torch.clamp(grid_coords, 0, num_grid - 1)
    
    return grid_coords

# *****************set grid_map = 1*****************
def set_outside_circle_to_one(grid_map):
    n = grid_map.shape[-1]
    
    radius = n // 2
    center = (radius, radius)
    
    for i in range(n):
        for j in range(n):
            distance = np.sqrt((i - center[0]) ** 2 + (j - center[1]) ** 2)
            
            if distance >= radius:
                grid_map[:, i, j] = 1
    
    return grid_map

class StateGenerator(object):
    """A base class for state generation."""

    def pretrain_uniform(self):
        """Pretrain the generator distribution to uniform distribution in the limit."""
        raise NotImplementedError

    def pretrain(self, states):
        """Pretrain with state distribution in the states list."""
        raise NotImplementedError

    def sample_states(self, size):
        """Sample states with given size."""
        raise NotImplementedError

    def sample_states_with_noise(self, size):
        """Sample states with noise."""
        raise NotImplementedError

    def train(self, states, labels):
        """Train with respect to given states and labels."""
        raise NotImplementedError

class StateGAN(StateGenerator):
    """A GAN for generating states. """
    def __init__(self, gan_configs, state_range = None, state_bounds = None):
        self.gan = LSGAN(gan_configs)
        self.gan_configs = gan_configs
        self.state_size = gan_configs['goal_size']
        self.evaluater_size = gan_configs['num_labels']
        self.state_center = np.array(gan_configs['goal_center'])
        if state_range is not None:
            self.state_bounds = state_range
        elif state_bounds is not None:
            self.state_bounds = np.array(state_bounds)
            self.state_range = self.state_bounds[1] - self.state_bounds[0]

        self.state_noise_level = gan_configs['goal_noise_level']

    def pretrain_uniform(self, size=10000, outer_iters=500):
        """
        :param size: number of uniformly sampled states (that we will try to fit as output of the GAN)
        :param outer_iters: of the GAN
        """
        states = np.random.uniform(
            self.state_center + self.state_bounds[0], self.state_center + self.state_bounds[1], size=(size, self.state_size)
        )
        return self.pretrain(states, outer_iters)

    def pretrain_uniform_rejection(self, states, outer_iters=500):
        """
        :param size: number of uniformly sampled states (that we will try to fit as output of the GAN)
        :param outer_iters: of the GAN
        """
        return self.pretrain(states, outer_iters)

    def pretrain(self, states, outer_iters=500, generator_iters=None, discriminator_iters=None):
        """
        Pretrain the state GAN to match the distribution of given states.
        :param states: the state distribution to match
        :param outer_iters: of the GAN
        """
        labels = np.ones((states.shape[0], self.evaluater_size))  # all state same label --> uniform
        return self.train(states, labels, outer_iters)

    def _add_noise_to_states(self, states):
        noise = np.random.randn(*states.shape) * self.state_noise_level
        states += noise
        return np.clip(states, self.state_center + self.state_bounds[0], self.state_center + self.state_bounds[1])

    def sample_states(self, size):  # un-normalizes the states
        normalized_states, noise = self.gan.sample_generator(size)
        tmp = normalized_states.cpu().detach().numpy()
        states = self.state_center + tmp * self.state_bounds[1]
        return states, noise

    def sample_states_with_noise(self, size):
        states, noise = self.sample_states(size)
        states = self._add_noise_to_states(states)
        return states, noise

    def train(self, states, labels, outer_iters=None, generator_iters=None, discriminator_iters=None):
        # normalize 0~1
        normalized_states = (states - self.state_center) / self.state_bounds[1][-1]
        return self.gan.train(normalized_states, labels, self.gan_configs)

    def discriminator_predict(self, states):
        return self.gan.discriminator_predict(states)

class GANBuffer(object):
    def __init__(self):
        self._state_buffer = np.zeros((0, 1), dtype=np.float32)
        self._history_buffer = np.zeros((0, 20), dtype=np.float32) # task_dim = 20
        self._weight_buffer = np.zeros((0, 1), dtype=np.float32)
        self.buffer_length = 10000
        self._temp_state_buffer = []
        self._temp_weight_buffer = []
    
    def insert(self, states):
        """
        input:
            states: list of np.array(size=(state_dim, ))
        """
        self._temp_state_buffer.extend(copy.deepcopy(states))

    def insert_weights(self, weights):
        self._temp_weight_buffer.append(weights.to('cpu').numpy())

    def insert_history(self, states):
        if len(states) > 0:
            self._history_buffer = np.concatenate([self._history_buffer, states])[-self.buffer_length:]

    def update(self):
        self._state_buffer = np.array(self._temp_state_buffer)
        self._weight_buffer = np.stack(self._temp_weight_buffer, axis=-1).mean(-1)

        # reset temp state and weight buffer
        self._temp_state_buffer = []
        self._temp_weight_buffer = []

    def sample(self, num_tasks):
        indices = np.random.choice(self._history_buffer.shape[0], num_tasks, replace=False)
        return self._history_buffer[indices]

    def save_task(self, model_dir, episode):
        np.save('{}/tasks_{}.npy'.format(model_dir,episode), self._state_buffer)
        np.save('{}/weights_{}.npy'.format(model_dir,episode), self._weight_buffer)

class HideAndSeek_circle_partial_TP_GAN(IsaacEnv): 
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
        self.dist_reward_coef = self.cfg.task.dist_reward_coef
        self.smoothness_coef = self.cfg.task.smoothness_coef
        self.use_eval = self.cfg.task.use_eval
        self.use_partial_obs = self.cfg.task.use_partial_obs
        self.capture = torch.zeros(self.num_envs, 3, device=self.device)
        self.min_dist = torch.ones(self.num_envs, 1, device=self.device) * float(torch.inf) # for teacher evaluation
        
        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )

        self.init_drone_pos_dist = D.Uniform(
            torch.tensor([0.1, -self.arena_size / math.sqrt(2.0)], device=self.device),
            torch.tensor([self.arena_size / math.sqrt(2.0), self.arena_size / math.sqrt(2.0)], device=self.device)
        )
        self.init_target_pos_dist = D.Uniform(
            torch.tensor([-self.arena_size / math.sqrt(2.0), -self.arena_size / math.sqrt(2.0)], device=self.device),
            torch.tensor([-0.1, self.arena_size / math.sqrt(2.0)], device=self.device)
        )

        self.init_drone_pos_dist_z = D.Uniform(
            torch.tensor([self.max_height / 2 - 0.1], device=self.device),
            torch.tensor([self.max_height / 2 + 0.1], device=self.device)
        )
        self.init_target_pos_dist_z = D.Uniform(
            torch.tensor([self.max_height / 2 - 0.1], device=self.device),
            torch.tensor([self.max_height / 2 + 0.1], device=self.device)
        )

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.2, -0.2, 0.0], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, 0.2], device=self.device) * torch.pi
        )

        if self.use_eval:
            self.init_rpy_dist = D.Uniform(
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi,
                torch.tensor([0.0, 0.0, 0.0], device=self.device) * torch.pi
            )

        self.mask_value = -5
        self.draw = _debug_draw.acquire_debug_draw_interface()

        # TP net
        # t, target pos masked, target vel masked, drone_pos
        self.TP = TP_net(input_dim = 1 + 3 + 3 + 3 * self.num_agents, output_dim = 3 * self.future_predcition_step, future_predcition_step = self.future_predcition_step, window_step=self.window_step).to(self.device)
        self.history_step = self.cfg.task.history_step
        self.history_data = collections.deque(maxlen=self.history_step)

        # for deployment
        self.prev_actions = torch.zeros(self.num_envs, self.num_agents, 4, device=self.device)

        # GAN proposer
        self.gan_buffer = GANBuffer()
        self.create_goalproposal_gan()
        self.update_iter = 0 # multiple initialization for agents and target
        self.eval_iter = 5

    def _set_specs(self):        
        drone_state_dim = self.drone.state_spec.shape.numel()
        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
        self.obs_max_cylinder = self.cfg.task.cylinder.obs_max_cylinder
        self.future_predcition_step = self.cfg.task.future_predcition_step
        self.history_step = self.cfg.task.history_step
        self.window_step = self.cfg.task.window_step

        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, 3 + 3 * self.future_predcition_step + self.time_encoding_dim + 13)),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, 3)), # pos
            "cylinders": UnboundedContinuousTensorSpec((self.obs_max_cylinder, 5)), # pos + radius + height
        }).to(self.device)
        state_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, 3 + 3 * self.future_predcition_step + self.time_encoding_dim + 13)),
            "cylinders": UnboundedContinuousTensorSpec((self.obs_max_cylinder, 5)), # pos + radius + height
        }).to(self.device)
        
        # TP network
        TP_spec = CompositeSpec({
            "TP_input": UnboundedContinuousTensorSpec((self.history_step, 1 + 3 + 3 + self.num_agents * 3)),
            # "TP_output": UnboundedContinuousTensorSpec((self.future_predcition_step, 3)),
            "TP_groundtruth": UnboundedContinuousTensorSpec((1, 3)),
            "TP_done": UnboundedContinuousTensorSpec((1, 3)),
        }).to(self.device)
        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": observation_spec.expand(self.drone.n),
                "state": state_spec,
                "TP": TP_spec
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
            state_key=("agents", "state"),
        )

        # stats and infos
        stats_spec = CompositeSpec({
            "success": UnboundedContinuousTensorSpec(1),
            "collision": UnboundedContinuousTensorSpec(1),
            "distance_reward": UnboundedContinuousTensorSpec(1),
            "distance_predicted_reward": UnboundedContinuousTensorSpec(1),
            "speed_reward": UnboundedContinuousTensorSpec(1),
            "collision_reward": UnboundedContinuousTensorSpec(1),
            "collision_wall": UnboundedContinuousTensorSpec(1),
            "collision_cylinder": UnboundedContinuousTensorSpec(1),
            "collision_drone": UnboundedContinuousTensorSpec(1),
            "detect_reward": UnboundedContinuousTensorSpec(1),
            "catch_reward": UnboundedContinuousTensorSpec(1),
            "smoothness_reward": UnboundedContinuousTensorSpec(1),
            "smoothness_mean": UnboundedContinuousTensorSpec(1),
            "smoothness_max": UnboundedContinuousTensorSpec(1),
            "first_capture_step": UnboundedContinuousTensorSpec(1),
            "sum_detect_step": UnboundedContinuousTensorSpec(1),
            "return": UnboundedContinuousTensorSpec(1),
            "action_error_order1_mean": UnboundedContinuousTensorSpec(1),
            "action_error_order1_max": UnboundedContinuousTensorSpec(1),
            "target_predicted_error": UnboundedContinuousTensorSpec(1),
            "distance_threshold_L": UnboundedContinuousTensorSpec(1),
            "dis_loss": UnboundedContinuousTensorSpec(1),
            "gen_loss": UnboundedContinuousTensorSpec(1),
        })
        # }).expand(self.num_envs).to(self.device)
        # add success and number for all cylinders
        for i in range(self.num_cylinders + 1):
            stats_spec['ratio_cylinders_{}'.format(i)] = UnboundedContinuousTensorSpec(1)
            stats_spec['success_cylinders_{}'.format(i)] = UnboundedContinuousTensorSpec(1)
        stats_spec = stats_spec.expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13), device=self.device),
            "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()
        
    def _design_scene(self): # for render
        self.num_agents = self.cfg.task.num_agents
        self.max_cylinders = self.cfg.task.cylinder.max_num
        self.drone_detect_radius = self.cfg.task.drone_detect_radius
        self.target_detect_radius = self.cfg.task.target_detect_radius
        self.catch_radius = self.cfg.task.catch_radius
        self.arena_size = self.cfg.task.arena_size
        self.max_height = self.cfg.task.max_height
        self.cylinder_size = self.cfg.task.cylinder.size
        self.cylinder_height = self.max_height
        self.scenario_flag = self.cfg.task.scenario_flag
        self.use_random_cylinder = self.cfg.task.use_random_cylinder
        self.use_GAN = self.cfg.task.use_GAN
        self.num_cylinders = self.max_cylinders
        self.invalid_z = -20.0 # for invalid cylinders_z, far enough
        # init grid map
        self.grid_size = 2 * self.cylinder_size
        self.num_grid = int(self.arena_size * 2 / self.grid_size)
        self.boundary = self.arena_size - 0.1
        self.use_TP_net = self.cfg.algo.use_TP_net
        
        # set all_cylinders under the ground
        all_cylinders_x = torch.arange(self.num_cylinders) * 2 * self.cylinder_size
        all_cylinders_pos = torch.zeros(self.num_cylinders, 3, device=self.device)
        all_cylinders_pos[:, 0] = all_cylinders_x
        all_cylinders_pos[:, 1] = 0.0
        all_cylinders_pos[:, 2] = self.invalid_z

        # init
        drone_pos = torch.tensor([
                            [0.6000,  0.0000, 0.5],
                            [0.8000,  0.0000, 0.5],
                            [0.8000, -0.2000, 0.5],
                            [0.8000,  0.2000, 0.5],
                        ], device=self.device)
        target_pos = torch.tensor([
                            [-0.8000,  0.0000, 0.5],
                        ], device=self.device)
        
        if self.use_random_cylinder:
            cylinders_pos_xy = self.rejection_sampling_random_cylinder(1, drone_pos, target_pos)
            cylinder_pos_z = torch.ones(1, self.num_cylinders, 1, device=self.device) * 0.5 * self.cylinder_height
            # set inactive cylinders under the ground
            cylinder_pos_z[self.inactive_mask] = self.invalid_z
            all_cylinders_pos = torch.concat([cylinders_pos_xy, cylinder_pos_z], dim=-1).squeeze(0)
        else:
            if self.scenario_flag == 'empty':
                num_fixed_cylinders = 0
            elif self.scenario_flag == '2cylinders':
                num_fixed_cylinders = 2
                all_cylinders_pos[:num_fixed_cylinders] = torch.tensor([
                                    [0.0, 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [0.0, - 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                ], device=self.device)
            elif self.scenario_flag == 'center':
                num_fixed_cylinders = 9
                all_cylinders_pos[:num_fixed_cylinders] = torch.tensor([
                                    [0.0, 0.0, 0.5 * self.cylinder_height],
                                    [0.0, 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [0.0, - 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [2 * self.cylinder_size, 0.0, 0.5 * self.cylinder_height],
                                    [- 2 * self.cylinder_size, 0.0, 0.5 * self.cylinder_height],
                                    [2 * self.cylinder_size, 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [- 2 * self.cylinder_size, 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [2 * self.cylinder_size, -2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [- 2 * self.cylinder_size, -2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                ], device=self.device)
            elif self.scenario_flag == 'corner':
                # init
                drone_pos = torch.tensor([
                                    [-0.6000,  0.0000, 0.5],
                                    [-0.8000,  0.0000, 0.5],
                                    [-0.8000, -0.2000, 0.5],
                                    [-0.8000,  0.2000, 0.5],
                                ], device=self.device)
                target_pos = torch.tensor([
                                    [0.8000,  0.8000, 0.5],
                                ], device=self.device)
                num_fixed_cylinders = 3
                all_cylinders_pos[:num_fixed_cylinders] = torch.tensor([
                                    [2 * self.cylinder_size, 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [2 * self.cylinder_size, 4 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [4 * self.cylinder_size, 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                ], device=self.device)
            elif self.scenario_flag == 'wall':
                num_fixed_cylinders = 4
                all_cylinders_pos[:num_fixed_cylinders] = torch.tensor([
                                    [0.0, 0.0, 0.5 * self.cylinder_height],
                                    [0.0, 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [0.0, 4 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [0.0, 8 * self.cylinder_size, 0.5 * self.cylinder_height],
                                ], device=self.device)
            elif self.scenario_flag == '2line':
                num_fixed_cylinders = 6
                all_cylinders_pos[:num_fixed_cylinders] = torch.tensor([
                                    [2 * self.cylinder_size, 0.0, 0.5 * self.cylinder_height],
                                    [-2 * self.cylinder_size, 0.0, 0.5 * self.cylinder_height],
                                    [2 * self.cylinder_size, 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [2 * self.cylinder_size, -2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [-2 * self.cylinder_size, 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [-2 * self.cylinder_size, -2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                ], device=self.device)
            elif self.scenario_flag == '3line':
                num_fixed_cylinders = 7
                all_cylinders_pos[:num_fixed_cylinders] = torch.tensor([
                                    [2 * self.cylinder_size, 0.0, 0.5 * self.cylinder_height],
                                    [2 * self.cylinder_size, -2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [2 * self.cylinder_size, 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [-2 * self.cylinder_size, -2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [-2 * self.cylinder_size, -4 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [-2 * self.cylinder_size, 2 * self.cylinder_size, 0.5 * self.cylinder_height],
                                    [-2 * self.cylinder_size, 4 * self.cylinder_size, 0.5 * self.cylinder_height],
                                ], device=self.device)

        if not self.use_random_cylinder:
            self.active_cylinders = torch.ones(self.num_envs, 1, device=self.device) * num_fixed_cylinders

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
            translation=target_pos,
            radius=0.05,
            color=torch.tensor([1., 0., 0.]),
            mass=1.0
        )

        for idx in range(self.num_cylinders):
            attributes = {'axis': 'Z', 'radius': self.cylinder_size, 'height': self.cylinder_height}
            create_obstacle(
                "/World/envs/env_0/cylinder_{}".format(idx), 
                prim_type="Cylinder",
                translation=all_cylinders_pos[idx],
                attributes=attributes
            ) # Use 'self.cylinders_prims[0].GetAttribute('radius').Get()' to get attributes
    
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

    def rejection_sampling_random_cylinder(self, num_tasks, drone_pos: torch.Tensor, target_pos: torch.Tensor):
        # init for drones and target
        grid_map = torch.zeros((num_tasks, self.num_grid, self.num_grid), device=self.device, dtype=torch.int)
        center_pos = torch.zeros((num_tasks, 1, 2), device=self.device)
        center_grid = torch.ones((num_tasks, 1, 2), device=self.device, dtype=torch.int) * int(self.num_grid / 2)
        grid_map = set_outside_circle_to_one(grid_map)
        # setup drone and target
        drone_grid = continuous_to_grid(drone_pos[..., :2], self.num_grid, self.grid_size, center_pos, center_grid)
        target_grid = continuous_to_grid(target_pos[..., :2], self.num_grid, self.grid_size, center_pos, center_grid)
        batch_indices = torch.arange(num_tasks).unsqueeze(1)
        x_indices = drone_grid[:, :, 0].flatten().long()
        y_indices = drone_grid[:, :, 1].flatten().long()
        grid_map[batch_indices.expand(-1, self.num_agents).flatten(), x_indices, y_indices] = 1
        x_indices = target_grid[:, :, 0].flatten().long()
        y_indices = target_grid[:, :, 1].flatten().long()
        grid_map[batch_indices.expand(-1, 1).flatten(), x_indices, y_indices] = 1
     
        # randomize number of activate cylinders, use it later
        self.active_cylinders = torch.randint(low=0, high=self.num_cylinders + 1, size=(num_tasks, 1), device=self.device)
        self.inactive_mask = torch.arange(self.num_cylinders, device=self.device).unsqueeze(0).expand(num_tasks, -1)
        # inactive = True, [envs, self.num_cylinders]
        self.inactive_mask = self.inactive_mask >= self.active_cylinders

        cylinders_grid = select_unoccupied_positions(grid_map, self.num_cylinders)
        
        objects_pos = grid_to_continuous(cylinders_grid, self.boundary, self.grid_size, center_pos, center_grid)
        return objects_pos
    
    def ignore_duplicated_cylinders(self, env_ids: torch.Tensor, drone_pos: torch.Tensor, target_pos: torch.Tensor):
        grid_map = torch.zeros((len(env_ids), self.num_grid, self.num_grid), device=self.device, dtype=torch.int)
        grid_map = set_outside_circle_to_one(grid_map)
        center_pos = torch.zeros((len(env_ids), 1, 2), device=self.device)
        center_grid = torch.ones((len(env_ids), 1, 2), device=self.device, dtype=torch.int) * int(self.num_grid / 2)
        # setup drone and target
        drone_grid = continuous_to_grid(drone_pos[..., :2], self.num_grid, self.grid_size, center_pos, center_grid)
        target_grid = continuous_to_grid(target_pos[..., :2], self.num_grid, self.grid_size, center_pos, center_grid)
        batch_indices = torch.arange(len(env_ids)).unsqueeze(1)
        x_indices = drone_grid[:, :, 0].flatten().long()
        y_indices = drone_grid[:, :, 1].flatten().long()
        grid_map[batch_indices.expand(-1, self.num_agents).flatten(), x_indices, y_indices] = 1
        x_indices = target_grid[:, :, 0].flatten().long()
        y_indices = target_grid[:, :, 1].flatten().long()
        grid_map[batch_indices.expand(-1, 1).flatten(), x_indices, y_indices] = 1

        # cylinders maybe in the same grid
        candidate_cylinders_grid = continuous_to_grid(self.candidate_cylinders_pos, self.num_grid, self.grid_size, center_pos, center_grid)
        # ignore duplication of cylinders and illegal cylinders
        cylinders_grid, self.active_cylinders = select_unoccupied_positions_GAN(grid_map, candidate_cylinders_grid)
        self.active_cylinders = self.active_cylinders.unsqueeze(-1).to(self.device)

        self.inactive_mask = torch.arange(self.num_cylinders, device=self.device).unsqueeze(0).expand(len(env_ids), -1)
        # inactive = True, [envs, self.num_cylinders]
        self.inactive_mask = self.inactive_mask >= self.active_cylinders
        cylinders_pos = grid_to_continuous(cylinders_grid, self.boundary, self.grid_size, center_pos, center_grid)
        
        return cylinders_pos

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        # init, fixed xy and randomize z
        drone_pos_z = self.init_drone_pos_dist_z.sample((*env_ids.shape, self.num_agents))
        target_pos_z = self.init_target_pos_dist_z.sample((*env_ids.shape, 1))

        if self.use_random_cylinder:
            if self.update_iter == 0: # fixed cylinders for eval_iter
                num_buffer = min(self.gan_buffer._history_buffer.shape[0], int(len(env_ids) * (1 - self.goal_configs['num_GAN'])))
                num_GAN = len(env_ids) - num_buffer
                # sample tasks by GAN
                tasks_GAN, _ = self.gan.sample_states_with_noise(num_GAN)
                if num_buffer > 0:
                    # sample from GAN_buffer
                    tasks_buffer = self.gan_buffer.sample(num_buffer)
                    self.candidate_tasks = np.concatenate([tasks_buffer, tasks_GAN])
                else:
                    self.candidate_tasks = tasks_GAN
                self.gan_buffer.insert(self.candidate_tasks)
                self.candidate_cylinders_pos = torch.from_numpy(self.candidate_tasks[..., 2 * self.num_agents + 2:]).to(self.device).reshape(len(env_ids), -1, 2).float()
            drone_pos = torch.from_numpy(self.candidate_tasks[..., :2 * self.num_agents]).to(self.device).reshape(len(env_ids), -1, 2).float()
            target_pos = torch.from_numpy(self.candidate_tasks[..., 2 * self.num_agents: 2 * self.num_agents + 2]).to(self.device).reshape(len(env_ids), -1, 2).float()
            cylinders_pos_xy = self.ignore_duplicated_cylinders(env_ids, drone_pos, target_pos)

            drone_pos = torch.concat([drone_pos, drone_pos_z], dim=-1)
            target_pos = torch.concat([target_pos, target_pos_z], dim=-1)
            cylinder_pos_z = torch.ones(len(env_ids), self.num_cylinders, 1, device=self.device) * 0.5 * self.cylinder_height
            # set inactive cylinders under the ground
            cylinder_pos_z[self.inactive_mask] = self.invalid_z
            cylinders_pos = torch.concat([cylinders_pos_xy, cylinder_pos_z], dim=-1)
        else:
            self.inactive_mask = torch.arange(self.num_cylinders, device=self.device).unsqueeze(0).expand(len(env_ids), -1)
            # inactive = True, [envs, self.num_cylinders]
            self.inactive_mask = self.inactive_mask >= self.active_cylinders
        
        if not self.use_random_cylinder:
            if self.scenario_flag == 'empty':
                drone_pos = torch.tensor([
                                    [0.6000,  0.0000, 0.5],
                                    [0.8000,  0.0000, 0.5],
                                    [0.8000, -0.2000, 0.5],
                                    [0.8000,  0.2000, 0.5],
                                ], device=self.device)
                target_pos = torch.tensor([
                                    [-0.8000,  0.0000, 0.5],
                                ], device=self.device)
            elif self.scenario_flag == 'corner':
                # init
                drone_pos = torch.tensor([
                                    [-0.6000,  0.0000, 0.5],
                                    [-0.8000,  0.0000, 0.5],
                                    [-0.8000, -0.2000, 0.5],
                                    [-0.8000,  0.2000, 0.5],
                                ], device=self.device).unsqueeze(0).expand(len(env_ids), -1, -1)
                target_pos = torch.tensor([
                                    [0.8000,  0.8000, 0.5],
                                ], device=self.device).unsqueeze(0).expand(len(env_ids), -1, -1)
            elif self.scenario_flag == 'wall':
                drone_pos = torch.tensor([
                                    [0.6000,  0.0000, 0.5],
                                    [0.8000,  0.0000, 0.5],
                                    [0.8000, -0.2000, 0.5],
                                    [0.8000,  0.2000, 0.5],
                                ], device=self.device)
                target_pos = torch.tensor([
                                    [-0.8000,  0.0000, 0.5],
                                ], device=self.device)
            elif self.scenario_flag == 'center':
                drone_pos = torch.tensor([
                                    [0.6000,  0.0000, 0.5],
                                    [0.8000,  0.0000, 0.5],
                                    [0.8000, -0.2000, 0.5],
                                    [0.8000,  0.2000, 0.5],
                                ], device=self.device)
                target_pos = torch.tensor([
                                    [-0.8000,  0.0000, 0.5],
                                ], device=self.device)
            elif self.scenario_flag == '2line':
                drone_pos = torch.tensor([
                                    [0.6000,  0.0000, 0.5],
                                    [0.8000,  0.0000, 0.5],
                                    [0.8000, -0.2000, 0.5],
                                    [0.8000,  0.2000, 0.5],
                                ], device=self.device)
                target_pos = torch.tensor([
                                    [0.0000,  0.0000, 0.5],
                                ], device=self.device)
            elif self.scenario_flag == '3line':
                drone_pos = torch.tensor([
                                    [0.6000,  0.0000, 0.5],
                                    [0.8000,  0.0000, 0.5],
                                    [0.8000, -0.2000, 0.5],
                                    [0.8000,  0.2000, 0.5],
                                ], device=self.device)
                target_pos = torch.tensor([
                                    [0.0000,  0.0000, 0.5],
                                ], device=self.device)
                       
        # drone_pos = self.init_drone_pos_dist.sample((*env_ids.shape, self.num_agents))
        rpy = self.init_rpy_dist.sample((*env_ids.shape, self.num_agents))
        rot = euler_to_quaternion(rpy)

        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        drone_init_velocities = torch.zeros_like(self.drone.get_velocities())
        self.drone.set_velocities(drone_init_velocities, env_ids)

        # target_pos = self.init_target_pos_dist.sample((*env_ids.shape, 1))
        self.target.set_world_poses(positions=target_pos + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids)
        
        # set cylinders
        if self.use_random_cylinder:
            self.cylinders.set_world_poses(positions=cylinders_pos + self.envs_positions[env_ids].unsqueeze(1), env_indices=env_ids)

        # reset stats
        self.stats[env_ids] = 0.
        self.stats['first_capture_step'].set_(torch.ones_like(self.stats['first_capture_step']) * self.max_episode_length)

        cmd_init = 2.0 * (self.drone.throttle[env_ids]) ** 2 - 1.0
        max_thrust_ratio = self.drone.params['max_thrust_ratio']
        self.info['prev_action'][env_ids, :, 3] = (0.5 * (max_thrust_ratio + cmd_init)).mean(dim=-1)
        self.prev_actions[env_ids] = self.info['prev_action'][env_ids]

        if self.use_eval and self._should_render(0):
            self._draw_court_circle()

        for substep in range(1):
            self.sim.step(self._should_render(substep))

    def _pre_sim_step(self, tensordict: TensorDictBase):   
        actions = tensordict[("agents", "action")]
        
        # for deployment
        self.info["prev_action"] = tensordict[("info", "prev_action")]
        self.prev_actions = self.info["prev_action"].clone()
        self.action_error_order1 = tensordict[("stats", "action_error_order1")].clone()
        self.stats["action_error_order1_mean"].add_(self.action_error_order1.mean(dim=-1).unsqueeze(-1))
        self.stats["action_error_order1_max"].set_(torch.max(self.stats["action_error_order1_max"], self.action_error_order1.mean(dim=-1).unsqueeze(-1)))

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
        # mask inactive cylinders(underground)
        self.cylinders_mask = (cylinders_pos[..., 2] < 0.0) # [num_envs, self.num_cylinders]
        cylinders_rpos = vmap(cpos)(drone_pos, cylinders_pos) # [num_envs, num_agents, num_cylinders, 3]
        self.cylinders_state = torch.concat([
            cylinders_rpos,
            self.cylinder_height * torch.ones(self.num_envs, self.num_agents, self.num_cylinders, 1, device=self.device),
            self.cylinder_size * torch.ones(self.num_envs, self.num_agents, self.num_cylinders, 1, device=self.device),
        ], dim=-1)
        # cylinders_mdist_z = torch.abs(cylinders_rpos[..., 2]) - 0.5 * self.cylinder_height
        cylinders_mdist = torch.norm(cylinders_rpos, dim=-1) - self.cylinder_size

        # use the kth nearest cylinders
        _, sorted_indices = torch.sort(cylinders_mdist, dim=-1)
        min_k_indices = sorted_indices[..., :self.obs_max_cylinder]
        self.min_distance_idx_expanded = min_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.cylinders_state.shape[-1])
        self.k_nearest_cylinders = self.cylinders_state.gather(2, self.min_distance_idx_expanded)
        # mask invalid cylinders
        self.k_nearest_cylinders_mask = self.cylinders_mask.unsqueeze(1).expand(-1, self.num_agents, -1).gather(2, min_k_indices)
        self.k_nearest_cylinders_masked = self.k_nearest_cylinders.clone()
        self.k_nearest_cylinders_masked.masked_fill_(self.k_nearest_cylinders_mask.unsqueeze(-1).expand_as(self.k_nearest_cylinders_masked), self.mask_value)
        obs["cylinders"] = self.k_nearest_cylinders_masked

        # state_self
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        target_vel = self.target.get_velocities()
        target_rpos = vmap(cpos)(drone_pos, target_pos) # [num_envs, num_agents, 1, 3]
        # self.blocked use in the _compute_reward_and_done
        # _get_dummy_policy_prey: recompute the blocked
        self.blocked = is_line_blocked_by_cylinder(drone_pos, target_pos, cylinders_pos, self.cylinder_size)
        in_detection_range = (torch.norm(target_rpos, dim=-1) < self.drone_detect_radius)
        # detect: [num_envs, num_agents, 1]
        detect = in_detection_range * (~ self.blocked.unsqueeze(-1))
        # broadcast the detect info to all drones
        self.broadcast_detect = torch.any(detect, dim=1)
        target_rpos_mask = (~ self.broadcast_detect).unsqueeze(-1).unsqueeze(-1).expand_as(target_rpos) # [num_envs, num_agents, 1, 3]
        target_rpos_masked = target_rpos.clone()
        target_rpos_masked.masked_fill_(target_rpos_mask, self.mask_value)
        
        t = (self.progress_buf / self.max_episode_length).unsqueeze(-1).unsqueeze(-1)

        # TP input
        target_mask = (~ self.broadcast_detect).unsqueeze(-1).expand_as(target_pos)
        target_pos_masked = target_pos.clone()
        target_pos_masked.masked_fill_(target_mask, self.mask_value)   
        target_vel_masked = target_vel[..., :3].clone()
        target_vel_masked.masked_fill_(target_mask, self.mask_value)

        # use the real target pos to supervise the TP network
        TP = TensorDict({}, [self.num_envs])
        frame_state = torch.concat([
            self.progress_buf.unsqueeze(-1),
            target_pos_masked.reshape(self.num_envs, -1),
            target_vel_masked.squeeze(1),
            drone_pos.reshape(self.num_envs, -1)
        ], dim=-1)
        if len(self.history_data) < self.history_step:
            # init history data
            for i in range(self.history_step):
                self.history_data.append(frame_state)
        else:
            self.history_data.append(frame_state)
        TP['TP_input'] = torch.stack(list(self.history_data), dim=1).to(self.device)
        # target_pos_predicted, x, y -> [-0.5 * self.arena_size, 0.5 * self.arena_size]
        # z -> [0, self.max_height]
        self.target_pos_predicted = self.TP(TP['TP_input']).reshape(self.num_envs, self.future_predcition_step, -1) # [num_envs, 3 * future_step]
        self.target_pos_predicted[..., :2] = self.target_pos_predicted[..., :2] * 0.5 * self.arena_size
        self.target_pos_predicted[..., 2] = (self.target_pos_predicted[..., 2] + 1.0) / 2.0 * self.max_height
        # TP["TP_output"] = self.target_pos_predicted
        TP["TP_done"] = (self.progress_buf <= (self.max_episode_length - self.future_predcition_step)).unsqueeze(-1)
        # TP_groundtruth: clip to (-1.0, 1.0)
        TP["TP_groundtruth"] = target_pos.squeeze(1).clone()
        TP["TP_groundtruth"][..., :2] = TP["TP_groundtruth"][..., :2] / (0.5 * self.arena_size)
        TP["TP_groundtruth"][..., 2] = TP["TP_groundtruth"][..., 2] / self.max_height * 2.0 - 1.0     

        target_rpos_predicted = (drone_pos.unsqueeze(2) - self.target_pos_predicted.unsqueeze(1)).view(self.num_envs, self.num_agents, -1)

        obs["state_self"] = torch.cat(
            [
            target_rpos_masked.reshape(self.num_envs, self.num_agents, -1),
            target_rpos_predicted,
            self.drone_states[..., 3:10],
            self.drone_states[..., 13:19],
            t.expand(-1, self.num_agents, self.time_encoding_dim),
            ], dim=-1
        ).unsqueeze(2)
                         
        # state_others
        if self.drone.n > 1:
            obs["state_others"] = self.drone_rpos
        
        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = torch.cat(
            [target_rpos.reshape(self.num_envs, self.num_agents, -1),
            target_rpos_predicted,
            self.drone_states[..., 3:10],
            self.drone_states[..., 13:19],
            t.expand(-1, self.num_agents, self.time_encoding_dim),
            ], dim=-1
        )   # [num_envs, drone.n, drone_state_dim]
        state["cylinders"] = self.k_nearest_cylinders_masked

        # draw drone trajectory and detection range
        if self._should_render(0) and self.use_eval:
            self._draw_catch()

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "state": state,
                    "TP": TP,
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        drone_pos, _ = self.get_env_poses(self.drone.get_world_poses())
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        
        # [num_envs, num_agents]
        target_dist = torch.norm(target_pos - drone_pos, dim=-1)

        # guidance, individual distance reward
        min_dist = torch.min(target_dist, dim=-1).values.unsqueeze(-1)
        active_distance_reward = (min_dist.expand_as(target_dist) > self.catch_radius).float()
        expanded_broadcast_detect = self.broadcast_detect.expand(-1, self.num_agents)
        distance_reward = - self.dist_reward_coef * target_dist * active_distance_reward
        self.stats['distance_reward'].add_(distance_reward.mean(-1).unsqueeze(-1))
        
        # detect
        detect_reward = self.detect_reward_coef * self.broadcast_detect.expand(-1, self.num_agents)
        # if detect, current_capture_step = progress_buf
        # else, current_capture_step = max_episode_length
        detect_flag = torch.any(self.broadcast_detect.expand(-1, self.num_agents), dim=1)
        self.stats['sum_detect_step'] += 1.0 * detect_flag.unsqueeze(1)
        self.stats['detect_reward'].add_(detect_reward.mean(-1).unsqueeze(-1))
        
        # capture
        self.capture = (target_dist < self.catch_radius)
        masked_capture = self.capture * (~ self.blocked).float()
        broadcast_capture = torch.any(masked_capture, dim=-1).unsqueeze(-1).expand_as(masked_capture) # cooperative reward
        catch_reward = self.catch_reward_coef * broadcast_capture
        # if capture, current_capture_step = progress_buf
        # else, current_capture_step = max_episode_length
        capture_flag = torch.any(catch_reward, dim=1)
        self.stats["success"] = torch.logical_or(capture_flag.unsqueeze(1), self.stats["success"]).float()
        current_capture_step = capture_flag.float() * self.progress_buf + (~capture_flag).float() * self.max_episode_length
        self.stats['first_capture_step'] = torch.min(self.stats['first_capture_step'], current_capture_step.unsqueeze(1))
        self.stats['catch_reward'].add_(catch_reward.mean(-1).unsqueeze(-1))

        # speed penalty
        drone_vel = self.drone.get_velocities()
        drone_speed_norm = torch.norm(drone_vel[..., :3], dim=-1)
        speed_reward = - self.speed_coef * (drone_speed_norm > self.cfg.task.v_drone)
        self.stats['speed_reward'].add_(speed_reward.mean(-1).unsqueeze(-1))

        # collison with cylinders, drones and walls
        # self.k_nearest_cylinders: [num_envs, num_agents, k, 5]
        # self.k_nearest_cylinders_mask: : [num_envs, num_agents, k]
        cylinder_pos_dist = torch.norm(self.k_nearest_cylinders[..., :2], dim= -1)
        collision_cylinder = (cylinder_pos_dist - self.cylinder_size < self.collision_radius).float() # [num_envs, num_agents, k]
        # mask inactive cylinders
        collision_cylinder.masked_fill_(self.k_nearest_cylinders_mask, 0.0)
        # sum all cylinders
        collision_cylinder = collision_cylinder.sum(-1)
        collision_reward = - self.collision_coef * collision_cylinder
        self.stats['collision_cylinder'].add_(collision_cylinder.mean(-1).unsqueeze(-1))
        # for drones
        drone_pos_dist = torch.norm(self.drone_rpos, dim=-1)
        collision_drone = (drone_pos_dist < 2.0 * self.collision_radius).float().sum(-1)
        collision_reward += - self.collision_coef * collision_drone
        self.stats['collision_drone'].add_(collision_drone.mean(-1).unsqueeze(-1))
        # for wall
        collision_wall = ((drone_pos[..., -1] > self.max_height).type(torch.float32) + ((drone_pos[..., 0]**2 + drone_pos[..., 1]**2) > self.arena_size**2).type(torch.float32))
        collision_reward += - self.collision_coef * collision_wall
        
        collision_flag = torch.any(collision_reward < 0, dim=1)
        self.stats["collision"].add_(collision_flag.unsqueeze(1))
        
        self.stats['collision_wall'].add_(collision_wall.mean(-1).unsqueeze(-1))
        self.stats['collision_reward'].add_(collision_reward.mean(-1).unsqueeze(-1))
        
        # smoothness
        smoothness_reward = self.smoothness_coef * torch.exp(-self.action_error_order1)
        self.stats['smoothness_reward'].add_(smoothness_reward.mean(-1).unsqueeze(-1))
        self.stats["smoothness_mean"].add_(self.drone.throttle_difference.mean(-1).unsqueeze(-1))
        self.stats["smoothness_max"].set_(torch.max(self.drone.throttle_difference.max(-1).values.unsqueeze(-1), self.stats["smoothness_max"]))
        
        reward = (
            distance_reward
            + detect_reward
            + catch_reward
            + collision_reward
            + speed_reward
            + smoothness_reward
        )

        done  = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        )

        if torch.any(done):
            # update weights
            self.gan_buffer.insert_weights(self.stats["success"])
            # update buffer, insert latest tasks
            self.update_iter += 1
            if self.update_iter >= self.eval_iter:
                self.update_iter = 0
                self.gan_buffer.update()
                # update info
                for i in range(self.num_cylinders + 1):
                    ratio_i = (self.active_cylinders == i).float().sum() / self.active_cylinders.shape[0]
                    self.stats['ratio_cylinders_{}'.format(i)] = torch.ones(self.num_envs, 1, device=self.device) * ratio_i
                    success_i = self.gan_buffer._weight_buffer[(self.active_cylinders == i).cpu()]
                    if len(success_i) > 0:
                        success_i = success_i.mean()
                    else:
                        success_i = 0.0
                    self.stats['success_cylinders_{}'.format(i)] = torch.ones(self.num_envs, 1, device=self.device) * success_i
                self.train_GAN()

        ep_len = self.progress_buf.unsqueeze(-1)
        self.stats["collision"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["action_error_order1_mean"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["target_predicted_error"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['smoothness_mean'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats['smoothness_reward'].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
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
        self.stats["collision_wall"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["collision_drone"].div_(
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["collision_cylinder"].div_(
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
        target_origin_dist = torch.norm(target_pos[..., :2],dim=-1)
        force_r[..., 0] = - target_pos[...,0] / ((self.arena_size - target_origin_dist)**2 + 1e-5)
        force_r[..., 1] = - target_pos[...,1] / ((self.arena_size - target_origin_dist)**2 + 1e-5)
        # up
        force_r[...,2] = - (self.max_height - target_pos[..., 2]) / ((self.max_height - target_pos[..., 2])**2 + 1e-5)
        # down
        force_r[...,2] += - (0.0 - target_pos[..., 2]) / ((0.0 - target_pos[..., 2])**2 + 1e-5)
        force += force_r
        
        # only get force from the nearest cylinder to the target
        # get the nearest cylinder to the target
        target_cylinders_mdist = torch.norm(target_cylinders_rpos, dim=-1) - self.cylinder_size
        target_min_distance_idx = torch.argmin(target_cylinders_mdist, dim=-1)
        # inactive mask
        target_min_distance_mask = self.cylinders_mask.gather(1, target_min_distance_idx)
        target_min_distance_idx_expanded = target_min_distance_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, target_cylinders_rpos.shape[-1])
        nearest_cylinder_to_target = target_cylinders_rpos.gather(2, target_min_distance_idx_expanded)
        
        force_c = torch.zeros_like(force)
        dist_target_cylinder = torch.norm(nearest_cylinder_to_target[..., :2], dim=-1)
        detect_cylinder = (dist_target_cylinder < self.target_detect_radius)
        force_c[..., :2] = ~target_min_distance_mask.unsqueeze(-1) * detect_cylinder * nearest_cylinder_to_target[..., :2].squeeze(2) / (dist_target_cylinder**2 + 1e-5)
        force += force_c

        return force.type(torch.float32)

    # GAN
    def create_goalproposal_gan(self):
        self.gan_configs = {
            'cuda':True,
            'batch_size': 64,
            'gan_outer_iters':100,
            'num_labels' : 1,  # normally only 1 label(success or not)  
            'goal_size': 0,
            'goal_center': [0,0],
            'goal_range': 0,
            'goal_noise_level': [0.1],
            'gan_noise_size': 16,
            'lr': 0.001,
        }
        
        self.goal_configs = {
            'num_GAN': 0.5,
            'R_min': 0.3,
            'R_max': 0.7,
        }

        # init the gan
        self.gan_configs['goal_size'] = self.num_agents * 2 + 1 * 2 + self.num_cylinders * 2 # pos
        self.gan_configs['goal_center'] = np.zeros(self.gan_configs['goal_size'], dtype=int)
        # agent, set x center
        self.gan_configs['goal_center'][0] = (0.1 + self.arena_size / math.sqrt(2.0)) / 2
        self.gan_configs['goal_center'][2] = (0.1 + self.arena_size / math.sqrt(2.0)) / 2
        self.gan_configs['goal_center'][4] = (0.1 + self.arena_size / math.sqrt(2.0)) / 2
        self.gan_configs['goal_center'][6] = (0.1 + self.arena_size / math.sqrt(2.0)) / 2
        # target
        self.gan_configs['goal_center'][8] = (-self.arena_size / math.sqrt(2.0) - 0.1) / 2
        # init range
        agent_range_low = [0.1, -self.arena_size / math.sqrt(2.0)]
        target_range_low = [-self.arena_size / math.sqrt(2.0), -self.arena_size / math.sqrt(2.0)]
        cylinders_range_low = [-self.boundary, -self.boundary]
        agent_range_up = [self.arena_size / math.sqrt(2.0), self.arena_size / math.sqrt(2.0)]
        target_range_up = [-0.1, self.arena_size / math.sqrt(2.0)]
        cylinders_range_up = [self.boundary, self.boundary]
        self.gan_configs['goal_range'] = np.vstack([np.array(agent_range_low * self.num_agents + target_range_low * 1 + cylinders_range_low * self.num_cylinders),
                                                    np.array(agent_range_up * self.num_agents + target_range_up * 1 + cylinders_range_up * self.num_cylinders)])
        self.gan = StateGAN(gan_configs = self.gan_configs, state_range=self.gan_configs['goal_range'])

        # pretrain the gan
        dis_loss, gen_loss = self.gan.pretrain_uniform(size=50000, outer_iters=self.gan_configs['gan_outer_iters'])
        print('discriminator_loss:',str(dis_loss.cpu()), 'generator_loss:',str(gen_loss.cpu()))

    def train_GAN(self):
        labels = np.zeros((len(self.gan_buffer._weight_buffer),1), dtype = float)
        tmp_buffer = []
        for i in range(len(self.gan_buffer._weight_buffer)):
            if self.gan_buffer._weight_buffer[i] <= self.goal_configs['R_max'] and self.gan_buffer._weight_buffer[i] >= self.goal_configs['R_min']:
                labels[i] = 1.0
                tmp_buffer.append(self.gan_buffer._state_buffer[i])
        self.gan_buffer.insert_history(np.array(tmp_buffer))
        dis_loss, gen_loss = self.gan.train(self.gan_buffer._state_buffer, labels)
        self.stats['dis_loss'] = torch.ones_like(self.stats['dis_loss']) * dis_loss
        self.stats['gen_loss'] = torch.ones_like(self.stats['dis_loss']) * gen_loss

    # visualize functions
    def _draw_court_circle(self):
        self.draw.clear_lines()

        point_list_1, point_list_2, colors, sizes = draw_court_circle(
            self.arena_size, self.max_height, line_size=5.0
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

        # drone detection
        drone_pos = self.drone_states[..., :3]
        drone_ori = self.drone_states[..., 3:7]
        drone_xaxis = quat_axis(drone_ori, 0)
        drone_yaxis = quat_axis(drone_ori, 1)
        drone_zaxis = quat_axis(drone_ori, 2)
        drone_point_list, drone_colors, drone_sizes = draw_detection(
            pos=drone_pos[self.central_env_idx, :],
            xaxis=drone_xaxis[self.central_env_idx, 0, :],
            yaxis=drone_yaxis[self.central_env_idx, 0, :],
            zaxis=drone_zaxis[self.central_env_idx, 0, :],
            drange=self.drone_detect_radius,
        )

        # target detection
        target_pos, target_ori = self.get_env_poses(self.target.get_world_poses())
        target_xaxis = quat_axis(target_ori, 0)
        target_yaxis = quat_axis(target_ori, 1)
        target_zaxis = quat_axis(target_ori, 2)
        target_point_list, target_colors, target_sizes = draw_detection(
            pos=target_pos[self.central_env_idx, :],
            xaxis=target_xaxis[self.central_env_idx, 0, :],
            yaxis=target_yaxis[self.central_env_idx, 0, :],
            zaxis=target_zaxis[self.central_env_idx, 0, :],
            drange=self.target_detect_radius,
        )
        
        point_list = drone_point_list + target_point_list
        colors = drone_colors + target_colors
        sizes = drone_sizes + target_sizes
        point_list = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list
        ]
        self.draw.draw_points(point_list, colors, sizes)

    def _draw_catch(self):
        self.draw.clear_points()
        # drone detection
        drone_pos = self.drone_states[..., :3]
        drone_ori = self.drone_states[..., 3:7]
        drone_xaxis = quat_axis(drone_ori, 0)
        drone_yaxis = quat_axis(drone_ori, 1)
        drone_zaxis = quat_axis(drone_ori, 2)
        # catch
        point_list, colors, sizes = draw_catch(
            pos=drone_pos[self.central_env_idx, :],
            xaxis=drone_xaxis[self.central_env_idx, 0, :],
            yaxis=drone_yaxis[self.central_env_idx, 0, :],
            zaxis=drone_zaxis[self.central_env_idx, 0, :],
            drange=self.catch_radius,
        )
        # predicted target
        for step in range(self.target_pos_predicted.shape[1]):
            point_list.append(Float3(self.target_pos_predicted[self.central_env_idx, step].cpu().numpy().tolist()))
            colors.append((1.0, 1.0, 0.0, 0.3))
            sizes.append(20.0)
        point_list = [
            _carb_float3_add(p, self.central_env_pos) for p in point_list
        ]
        # catch, green
        catch_mask = self.capture[self.central_env_idx].unsqueeze(1).expand(-1, 400).reshape(-1)
        for idx in range(len(catch_mask)):
            if catch_mask[idx]:
                colors[idx] = (0.0, 1.0, 0.0, 0.3)
        # blocked, red
        block_mask = self.blocked[self.central_env_idx].unsqueeze(1).expand(-1, 400).reshape(-1)
        for idx in range(len(block_mask)):
            if block_mask[idx]:
                colors[idx] = (1.0, 0.0, 0.0, 0.3)
        self.draw.draw_points(point_list, colors, sizes)