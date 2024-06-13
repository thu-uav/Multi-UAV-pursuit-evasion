from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core import objects
import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import torch
import torch.distributions as D
from omni_drones.views import RigidPrimView
from omni.isaac.core.prims import GeometryPrimView
import numpy as np

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, make_cells, euler_to_quaternion
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from omni.isaac.debug_draw import _debug_draw

REAL_TRIANGLE = [
    [-1, 0, 0],
    [0, -0.6, 0],
    [0, 0.6, 0]
]

REGULAR_HEXAGON = [
    [0, 0, 0],
    [1.7321, -1, 0],
    [0, -2, 0],
    [-1.7321, -1, 0],
    [-1.7321, 1.0, 0],
    [0.0, 2.0, 0.0],
    [1.7321, 1.0, 0.0],
]

REGULAR_TETRAGON = [
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
    [0, 0, 0],
]

REGULAR_TRIANGLE = [
    [1, 0, 0],
    [-0.5, 0.866, 0],
    [-0.5, -0.866, 0]
]

SINGLE = [
    #[0.618, -1.9021, 0],
    [0, 0, 0],
    [2, 0, 0]
    #[0.618, 1.9021, 0],
]

REGULAR_PENTAGON = [
    [2., 0, 0],
    [0.618, 1.9021, 0],
    [-1.618, 1.1756, 0],
    [-1.618, -1.1756, 0],
    [0.618, -1.9021, 0],
    [0, 0, 0]
]

REGULAR_SQUARE = [
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
]

DENSE_SQUARE = [
    [1, 1, 0],
    [1, 0, 0],
    [1, -1, 0],
    [0, 1, 0],
    [0, 0, 0],
    [0, -1, 0],
    [-1, -1, 0],
    [-1, 0, 0],
    [-1, 1, 0],
]

FORMATIONS = {
    "hexagon": REGULAR_HEXAGON,
    "tetragon": REGULAR_TETRAGON,
    "square": REGULAR_SQUARE,
    "dense_square": DENSE_SQUARE,
    "regular_pentagon": REGULAR_PENTAGON,
    "single": SINGLE,
    "triangle": REGULAR_TRIANGLE,
    'real': REAL_TRIANGLE,
}


class MultiCatch_old(IsaacEnv):
    def __init__(self, cfg, headless):
        self.ball_num = cfg.task.ball_num
        self.static_obs_num = cfg.task.static_obs_num if cfg.task.static_obs_type==2 else cfg.task.static_obs_num*10
        self.formation_type = cfg.task.formation_type
        self.obs_hit_distance = cfg.task.obs_hit_distance
        self.cfg = cfg
        self.frame_counter = 0
        self.real_flight = cfg.task.get("real_drone", False)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.time_encoding = self.cfg.task.time_encoding
        self.safe_distance = self.cfg.task.safe_distance
        self.formation_size = self.cfg.task.formation_size
        self.obs_safe_distance = self.cfg.task.obs_safe_distance
        self.soft_obs_safe_distance = self.cfg.task.soft_obs_safe_distance
        self.ball_reward_coeff = self.cfg.task.ball_reward_coeff
        self.ball_speed = self.cfg.task.ball_speed
        self.random_ball_speed = self.cfg.task.random_ball_speed
        self.velocity_coeff = self.cfg.task.velocity_coeff
        self.formation_coeff = self.cfg.task.formation_coeff
        self.height_coeff = self.cfg.task.height_coeff
        self.throw_threshold = self.cfg.task.throw_threshold
        self.ball_hard_coeff = self.cfg.task.ball_hard_reward_coeff / self.cfg.task.ball_reward_coeff

        assert self.formation_type in ['h', 'l']
        super().__init__(cfg, headless)

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.total_frame = self.cfg.total_frames
        self.drone.initialize() 
        self.randomization = cfg.task.get("randomization", {})
        if "drone" in self.randomization:
            self.drone.setup_randomization(self.randomization["drone"])
            
        if self.ball_num > 0:
            # create and initialize additional views
            self.ball = RigidPrimView(
                "/World/envs/env_*/ball_*",
                reset_xform_properties=False,
            )

            self.ball.initialize()
            self.ball.set_masses(torch.ones_like(self.ball.get_masses()))

        # initial state distribution
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.1, -.1, -.2], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, .2], device=self.device) * torch.pi
        )
        self.target_heading = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)
        self.target_heading[..., 0] = 1.

        self.flag = torch.zeros((self.num_envs, self.ball_num), dtype=bool, device=self.device)
        self.t_throw = torch.zeros((self.num_envs, self.ball_num), device=self.device)
        self.mask_observation = torch.tensor([self.cfg.task.obs_range, -1, -1, -1, -1, -1, -1, -1, -1, -1], device=self.device).float()
        self.t_launched = torch.full(size=(self.num_envs, self.ball_num), fill_value=torch.nan, device=self.device)
        self.ball_reward_flag = torch.zeros((self.num_envs, self.ball_num), dtype=bool, device=self.device)
        self.ball_alarm = torch.ones((self.num_envs, self.ball_num), dtype=bool, device=self.device)
        self.separation_penalty = torch.zeros(self.num_envs, self.drone.n, self.drone.n-1, device=self.device)
        self.t_moved = torch.full(size=(self.num_envs,self.ball_num), fill_value=torch.nan, device=self.device)
        self.t_difference = torch.full(size=(self.num_envs,self.ball_num), fill_value=torch.nan, device=self.device)
        self.t_hit = torch.full(size=(self.num_envs, self.ball_num), fill_value=torch.nan, device=self.device)

        self.morl_smooth_coeff = self.cfg.task.morl_smooth_coeff
        self.morl_formation_coeff = self.cfg.task.morl_formation_coeff
        self.morl_obstacle_coeff = self.cfg.task.morl_obstacle_coeff
        self.morl_forward_coeff = self.cfg.task.morl_forward_coeff

        self.alpha = 0.
        self.gamma = 0.995
        self.last_cost_h = torch.zeros(self.num_envs, 1, device=self.device)

    def _design_scene(self) -> Optional[List[str]]:
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        scene_utils.design_scene()

        formation = self.cfg.task.formation
        if isinstance(formation, str):
            self.formation = torch.as_tensor(
                FORMATIONS[formation], device=self.device
            ).float()
        elif isinstance(formation, list):
            self.formation = torch.as_tensor(
                self.cfg.task.formation, device=self.device
            )
        else:
            raise ValueError(f"Invalid target formation {formation}")

        self.target_pos_single = torch.tensor([0., 0., 1.5], device=self.device)

        self.init_pos_single = self.target_pos_single.clone()
        self.target_height = self.target_pos_single[2].item()

        self.formation = self.formation*self.cfg.task.formation_size/2
        self.formation_L = laplacian(self.formation, self.cfg.task.normalize_formation)
        relative_pos = cpos(self.formation, self.formation)
        drone_pdist = off_diag(torch.norm(relative_pos, dim=-1, keepdim=True))
        self.standard_formation_size = drone_pdist.max(dim=-2).values.max(dim=-2).values

        for ball_id in range(self.ball_num):
            ball = objects.DynamicSphere(
                prim_path=f"/World/envs/env_0/ball_{ball_id}",  
                position=torch.tensor([ball_id, 1., 0.]),
                radius = 0.15,
                color=torch.tensor([1., 0., 0.]),
            )

        self.margin = self.cfg.task.static_margin
        self.border = self.cfg.task.grid_border
        self.grid_size = self.cfg.task.grid_size

        self.drone.spawn(translations=self.formation+self.init_pos_single)
        self.target_pos = self.target_pos_single.expand(self.num_envs, self.drone.n, 3) + self.formation
        self.drone_id = torch.Tensor(np.arange(self.drone.n)).to(self.device)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[0]
        obs_self_dim = 3 + 3 + 4 + 3 + 3 # position, velocity, quaternion, heading, up
        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            obs_self_dim += self.time_encoding_dim

        state_dim = drone_state_dim

        obs_others_dim = 3 + 1 + 3
    
        state_spec = CompositeSpec({
                        "drones": UnboundedContinuousTensorSpec((self.drone.n, state_dim)),
                        "balls": UnboundedContinuousTensorSpec((self.ball_num, 6)),
                    })

        self.ball_obs_dim = self.ball_num if self.ball_num>0 else 1
        
        agent_obs_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)), # 20
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, obs_others_dim)),
            "attn_obs_ball": UnboundedContinuousTensorSpec((self.ball_obs_dim, 10)), # 7
        }).expand(self.drone.n)

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": agent_obs_spec, 
                "state": state_spec
            }
        }).expand(self.num_envs) # .to(self.device)

        self.action_spec = CompositeSpec({
            "agents": {
                "action": torch.stack([self.drone.action_spec]*self.drone.n, dim=0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((self.drone.n, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state")
        )

        stats_spec = CompositeSpec({
            "cost_l": UnboundedContinuousTensorSpec(1), 
            "cost_l_unnormalized": UnboundedContinuousTensorSpec(1), 
            "reward_formation": UnboundedContinuousTensorSpec(1),
            "reward_size": UnboundedContinuousTensorSpec(1),
            "separation_reward": UnboundedContinuousTensorSpec(1),
            "morl_formation_reward": UnboundedContinuousTensorSpec(1),
            
            "height_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "pos_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "vel_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "reward_heading": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_forward_reward": UnboundedContinuousTensorSpec(self.drone.n),
            
            "ball_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "cube_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "hit_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_obstacle_reward": UnboundedContinuousTensorSpec(self.drone.n),
            
            "reward_effort": UnboundedContinuousTensorSpec(self.drone.n),
            "reward_action_smoothness": UnboundedContinuousTensorSpec(self.drone.n),
            "reward_spin": UnboundedContinuousTensorSpec(self.drone.n),
            "reward_throttle_smoothness": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_smooth_reward": UnboundedContinuousTensorSpec(self.drone.n),
            
            "reward": UnboundedContinuousTensorSpec(self.drone.n),
            "return": UnboundedContinuousTensorSpec(1),
            
            "t_launched": UnboundedContinuousTensorSpec(1),
            "t_moved": UnboundedContinuousTensorSpec(1),
            "t_difference": UnboundedContinuousTensorSpec(1),
            "t_hit": UnboundedContinuousTensorSpec(1),
            "truncated": UnboundedContinuousTensorSpec(1), 
            # "terminated": UnboundedContinuousTensorSpec(1),
            "forward_success": UnboundedContinuousTensorSpec(1), 
            "crash": UnboundedContinuousTensorSpec(1),
            "not straight": UnboundedContinuousTensorSpec(1), 
            "hit": UnboundedContinuousTensorSpec(1),
            "hit_b": UnboundedContinuousTensorSpec(1),
            "hit_c": UnboundedContinuousTensorSpec(1),
            "too close": UnboundedContinuousTensorSpec(1),
            "done": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "formation_success": UnboundedContinuousTensorSpec(1),
            "action_success": UnboundedContinuousTensorSpec(1),
            "success": UnboundedContinuousTensorSpec(1),
            
            "size": UnboundedContinuousTensorSpec(1),
            "indi_b_dist": UnboundedContinuousTensorSpec(1),
            "curr_formation_dist": UnboundedContinuousTensorSpec(1),

            # "survival_reward": UnboundedContinuousTensorSpec(1),
            # "survival_return": UnboundedContinuousTensorSpec(1),

            "morl_obstacle": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_forward": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_smooth": UnboundedContinuousTensorSpec(self.drone.n),
            "morl_formation": UnboundedContinuousTensorSpec(self.drone.n),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
            "network_output":  UnboundedContinuousTensorSpec((self.drone.n, 4)),
            "prev_network_output":  UnboundedContinuousTensorSpec((self.drone.n, 4)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.ball_reward_flag[env_ids] = 0.

        pos = (
            (self.formation+self.init_pos_single).expand(len(env_ids), *self.formation.shape) # (k, 3) -> (len(env_ids), k, 3)
            + self.envs_positions[env_ids].unsqueeze(1)
        )
        rpy = torch.zeros((*env_ids.shape, self.drone.n, 3)).to(pos.device)
        rot = euler_to_quaternion(rpy)

        vel = torch.zeros(len(env_ids), self.drone.n, 6, device=self.device)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(vel, env_ids)
        self.last_cost_h[env_ids] = vmap(cost_formation_hausdorff)(
            pos, desired_p=self.formation
        )

        ball_ids = torch.arange(self.ball_num, device=self.device).expand(len(env_ids), -1)
        pos = (
            torch.tensor([[i, 0, 0.15] for i in range(self.ball_num)], device=self.device).expand(len(env_ids), self.ball_num, 3)
            + self.envs_positions[env_ids].unsqueeze(1)
        )
        vel = torch.zeros(len(env_ids)*self.ball_num, 6, device=self.device)
        env_ids = env_ids.unsqueeze(1).expand(-1, self.ball_num).reshape(-1)
        env_indices = env_ids * self.ball_num + ball_ids.reshape(-1)
            
        self.ball.set_world_poses(pos.reshape(-1, 3), env_indices=env_indices)
        self.ball.set_velocities(vel, env_indices)

        self.flag[env_ids] = False

        self.stats[env_ids] = 0.
        self.t_launched[env_ids]=torch.nan
        self.t_moved[env_ids]=torch.nan
        self.t_difference[env_ids]=torch.nan
        self.t_hit[env_ids] =torch.nan
        self.ball_alarm[env_ids] = 1
        self.separation_penalty[env_ids] = 0.
        if not self.cfg.task.throw_together:
            self.t_throw[env_ids] = torch.rand(len(env_ids), self.ball_num, device=self.device) * self.cfg.task.throw_time_range + self.throw_threshold
        else:
            self.t_throw[env_ids] = self.throw_threshold + torch.rand(()) * self.cfg.task.throw_time_range
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.last_network = tensordict[('info', 'prev_network_output')]
        self.current_network = tensordict[('info', 'network_output')]
        self.effort = self.drone.apply_action(actions)
        self.throw_multi_ball()

    def throw_multi_ball(self):
        flag = (self.progress_buf.unsqueeze(-1).expand(self.num_envs, self.ball_num) >= self.t_throw)
        should_throw = flag & (~self.flag) # [envs_num, ball_num]
        if self.random_ball_speed:
            ball_speed = torch.rand(size=()) * (self.cfg.task.max_ball_speed - self.cfg.task.min_ball_speed) + self.cfg.task.min_ball_speed
        else:
            ball_speed = self.ball_speed
        if should_throw.any():
            throw_indices = should_throw.nonzero(as_tuple=True)
            throw_ball_num = throw_indices[0].shape[0]
            self.t_launched[throw_indices] = self.progress_buf.unsqueeze(-1).expand(-1, self.ball_num)[throw_indices]
            self.ball_reward_flag[throw_indices] = 1
            # self.mask[should_throw, -1] = False #0.
            # Compute centre of 4 drones for all the environment\
            # The first index represent for the environment
            # 2nd for the Drone ID
            # 3rd for the position state

            # Approximate the maximum distance between drones after forming square
            if self.cfg.task.throw_center:
                centre_D = self.drone.pos[throw_indices[0]][..., :2].mean(1)
                centre_z = self.drone.pos[throw_indices[0]][..., 2].mean(1)
            else:
                random_idx = torch.randint(low=0, high=self.drone.n, size=(len(throw_indices[0]), ))
                centre = self.drone.pos[throw_indices[0], random_idx]
                centre_D = centre[..., :2]
                centre_z = centre[..., 2]
                throw_center_envs = torch.rand(throw_ball_num) < self.cfg.task.throw_center_ratio
                centre_D[throw_center_envs] = self.drone.pos[throw_indices[0]][throw_center_envs][..., :2].mean(1) # [env, 2]
                centre_z[throw_center_envs] = self.drone.pos[throw_indices[0]][throw_center_envs][..., 2].mean(1)

            # target = torch.rand(centre_D.shape, device=self.device)*2
            target_ball_pos = torch.zeros(throw_ball_num,3, device=self.device)
            ball_pos = torch.zeros(throw_ball_num,3, device=self.device)

            throw_ball_pattern = self.cfg.task.throw_ball_pattern
            if throw_ball_pattern < 0:
                if not self.cfg.task.eval:
                    if self.frame_counter < 1/2 * self.total_frame:
                        thres = 1.0
                    else:
                        thres = 1.0 - (self.frame_counter / self.total_frame - 1/2)
                else:
                    thres = 0.5
                p = torch.rand(size=())
                if p < thres:
                    throw_ball_pattern = 2
                else:
                    throw_ball_pattern = 1
            # print("throw_pattern =", throw_ball_pattern)
            # firstly, calculate vel_z
            #============
            if self.cfg.task.throw_ball_pattern == 0: # throw_z might <= 0
                # 注意 ball_vel 和 ball_target_vel 的差别在 vz 上
                # given t_hit, randomize ball init position & final position
                t_hit = torch.rand(throw_ball_num, device=self.device) * 1.5 + 0.5
            
                ball_target_vel = torch.ones(throw_ball_num, 3, device=self.device)
                ball_target_vel[:, 2] = - torch.rand(throw_ball_num, device=self.device) - 1. #[-2, -1]
                ball_target_vel[:, :2] = 2*(torch.rand(throw_ball_num, 2, device=self.device)-0.5) #[-1, 1]
                ball_target_vel = ball_target_vel/torch.norm(ball_target_vel, p=2, dim=1, keepdim=True) * ball_speed
                
                ball_vel = torch.ones(throw_ball_num, 6, device=self.device)
                ball_vel[:, :3] = ball_target_vel.clone()
                ball_vel[:, 2] = ball_target_vel[:, 2] + 9.81*t_hit
                
            elif self.cfg.task.throw_ball_pattern == 1:
                ball_vxy = 2 * (torch.rand(throw_ball_num, 2, device=self.device) - 0.5)
                ball_vxy = ball_vxy/torch.norm(ball_vxy, p=2, dim=1, keepdim=True) * ball_speed
                ball_vel = torch.zeros(throw_ball_num, 6, device=self.device)
                ball_vel[:, :2] = ball_vxy
                t_hit = torch.rand(throw_ball_num, device=self.device) * 0.8 + 0.8 # (0.8, 1.6)
                z = torch.rand(throw_ball_num, device=self.device) * centre_z + 0.5 * centre_z # [0.5h, 1.5h]
                ball_vel[:, 2] = (centre_z - z)/t_hit + 0.5*9.81*t_hit
                ball_vel[:, 3:] = 1.0
            elif self.cfg.task.throw_ball_pattern == 2:
                ball_target_vel = torch.ones(throw_ball_num, 3, device=self.device)
                ball_target_vel[:, 2] = - torch.rand(throw_ball_num, device=self.device) - 0.5 #[-1.5, -0.5]
                ball_target_vel[:, :2] = 2*(torch.rand(throw_ball_num, 2, device=self.device)-0.5) #[-1, 1]
                ball_target_vel = ball_target_vel/torch.norm(ball_target_vel, p=2, dim=1, keepdim=True) * ball_speed
                target_vz = ball_target_vel[:, 2]
                z_max = centre_z + target_vz**2/(2*9.81)
                z = torch.rand(throw_ball_num, device=self.device) * z_max
                # delta_z = v_t*t + 1/2*g*t^2
                t_hit = 1/9.81 * (-target_vz + torch.sqrt(target_vz**2 + 2*9.81*(centre_z - z)))
                ball_vel = torch.ones(throw_ball_num, 6, device=self.device)
                ball_vel[:, :3] = ball_target_vel.clone()
                ball_vel[:, 2] = ball_target_vel[:, 2] + 9.81*t_hit
                
            else:
                raise NotImplementedError()
            
            drone_x_speed = torch.mean(self.root_states[throw_indices[0]][..., 7], 1)
            drone_x_dist = drone_x_speed * t_hit

            drone_y_speed = torch.mean(self.root_states[throw_indices[0]][..., 8], 1)
            drone_y_dist = drone_y_speed * t_hit

            if self.cfg.task.throw_center:
                drone_x_max = torch.max(self.root_states[throw_indices[0]][..., 0], -1)[0]
                drone_x_min = torch.min(self.root_states[throw_indices[0]][..., 0], -1)[0]
                drone_y_max = torch.max(self.root_states[throw_indices[0]][..., 1], -1)[0]
                drone_y_min = torch.min(self.root_states[throw_indices[0]][..., 1], -1)[0]

                target_ball_pos[:, 0] = drone_x_dist + \
                    torch.rand(throw_ball_num, device=self.device) * (drone_x_max - drone_x_min) + drone_x_min
                target_ball_pos[:, 1] = drone_y_dist + \
                    torch.rand(throw_ball_num, device=self.device) * (drone_y_max - drone_y_min) + drone_y_min
                target_ball_pos[:, 2] = centre_z
            else:
                target_ball_pos[:, 0] = drone_x_dist + \
                    torch.rand(throw_ball_num, device=self.device) * 0.6 - 0.3 + centre_D[..., 0]
                target_ball_pos[:, 1] = drone_y_dist + \
                    torch.rand(throw_ball_num, device=self.device) * 0.6 - 0.3 + centre_D[..., 1]
                target_ball_pos[:, 2] = centre_z

            ball_pos[:, :2] = target_ball_pos[:, :2] - ball_vel[:, :2]*t_hit.view(-1, 1)
            ball_pos[:, 2] = target_ball_pos[:, 2] - ball_vel[:, 2]*t_hit + 0.5*9.81*t_hit**2
            
            #============
            self.t_hit[throw_indices] = t_hit / self.cfg.sim.dt
            assert throw_ball_num == ball_pos.shape[0]

            index_1d = throw_indices[0] * self.ball_num + throw_indices[1]
            self.ball.set_world_poses(positions=ball_pos + self.envs_positions[throw_indices[0]], env_indices=index_1d)
            self.ball.set_velocities(ball_vel, env_indices=index_1d)

            # draw target in red, draw init in green
            # draw_target_coordinates = target_ball_pos + self.envs_positions[should_throw]
            # draw_init_coordinates = ball_pos + self.envs_positions[should_throw]
            # colors = [(1.0, 0.0, 0.0, 1.0) for _ in range(throw_ball_num)] + [(0.0, 1.0, 0.0, 1.0) for _ in range(throw_ball_num)]
            # sizes = [2.0 for _ in range(2*throw_ball_num)]
            
            # self.draw.draw_points(draw_target_coordinates.tolist() + draw_init_coordinates.tolist(), colors, sizes)
        self.flag.bitwise_or_(flag)

    def _compute_state_and_obs(self):
        obs_self = []
        self.drone_states = self.drone.get_state()
        self.info["drone_state"][:] = self.drone_states[..., :13]
        drone_pos = self.drone_states[..., :3]
        drone_vel = self.drone.get_velocities()

        # Relative position between the ball and all the drones
        reshape_ball_world_poses = (self.ball.get_world_poses()[0].view(-1, self.ball_num, 3), self.ball.get_world_poses()[1].view(-1, self.ball_num, 4))
        balls_pos, balls_rot = self.get_env_poses(reshape_ball_world_poses) # [env_num, ball_num, 3]
        balls_vel = self.ball.get_linear_velocities().view(-1, self.ball_num, 3) # [env_num, 1, ball_num, 3]
        ball_state = torch.cat([balls_pos, balls_vel], dim=-1)
        relative_b_pos = balls_pos.unsqueeze(1) - drone_pos[..., :3].unsqueeze(2) # [env_num, drone_num, 1, 3] - [env_num, 1, ball_num, 3]
        relative_b_vel = balls_vel - drone_vel

        obs_self.append(-relative_b_pos)
        obs_self.append(-relative_b_vel)
        obs_self.append(self.root_state[..., 3:10])
        obs_self.append(self.root_state[..., 13:19])

        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs_self.append(t.expand(-1, self.drone.n, self.time_encoding_dim))
        
        obs_self = torch.cat(obs_self, dim=-1)

        # obs_others
        obs_others = []
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)
        self.drone_rvel = vmap(cpos)(drone_vel, drone_vel)
        self.drone_rvel = vmap(off_diag)(self.drone_rve)
        obs_others.append(self.drone_rpos)
        obs_others.append(self.drone_rvel)
 
        obs_ball = torch.cat([
            relative_b_pos, # [n, k, m, 3]
            relative_b_vel,
            balls_vel.unsqueeze(1).expand(-1,self.drone.n,-1,-1)
        ], dim=-1).view(self.num_envs, self.drone.n, -1, 10) #[env, agent, obstacle_num, *]
        
        # TODO: add top k mask
        # ball mask
        # not thrown
        ball_mask = (~self.ball_reward_flag).unsqueeze(-2).expand(-1, self.drone.n, -1) #[env_num, drone, ball_num]
        # after landed
        after_landed = (balls_pos[..., 2] < 0.2).unsqueeze(-2).expand(-1, self.drone.n, -1) #[env_num, drone, ball_num]
        ball_mask = ball_mask | after_landed
        if self.cfg.task.use_mask_behind:
            mask_behind = (relative_b_pos[..., 1] < 0) #[env_num, drone, ball_num]
            ball_mask = ball_mask | mask_behind
        if self.cfg.task.use_mask_front:
            mask_front = (relative_b_pos[..., 1] > self.cfg.task.obs_range) #[env_num, drone, ball_num]
            ball_mask = ball_mask | mask_front
        if self.cfg.task.mask_range:
            mask_range = (relative_b_dis > self.cfg.task.obs_range) #[env_num, drone, ball_num]
            ball_mask = ball_mask | mask_range

        obs = TensorDict({ 
            "obs_self": obs_self.unsqueeze(2),  # [N, K, 1, obs_self_dim]
            "obs_others": obs_others, # [N, K, K-1, obs_others_dim]
            "attn_obs_ball": obs_ball, # [N, K, ball_num, *]
        }, [self.num_envs, self.drone.n]) # [N, K, n_i, m_i]
        if self.cfg.task.use_attn_mask:
            obs["attn_ball_mask"] = ball_mask

        state = TensorDict({
            "drones": self.root_states,
            "balls": ball_state,
            }, self.batch_size)

        return TensorDict({
            "agents":{
                "observation": obs,    # input for the network
                "state": state,
            },
            "stats": self.stats,
            "info": self.info
        }, self.batch_size)

    def _compute_reward_and_done(self):
        pos, rot = self.get_env_poses(self.drone.get_world_poses())

        # formation objective
        normalized = self.cfg.task.normalize_formation
        cost_l = vmap(cost_formation_laplacian)(pos, desired_L=self.formation_L, normalized=True)
        cost_l_unnormalized = vmap(cost_formation_laplacian)(pos, desired_L=self.formation_L, normalized=False)
        size = self.drone_pdist.max(dim=-2).values.max(dim=-2).values
        reward_formation = 1 / (1 + torch.square(cost_l * 10))
        # size_error = torch.abs(size - self.standard_formation_size)
        # reward_size = torch.clip(1-size_error, min=0)
        reward_size = 1 / (1 + torch.square(size - self.standard_formation_size))
        reward_size += 1 / (1 + cost_l_unnormalized)
        # reward_formation = 1 / (1 + cost_l_unnormalized)
        # reward_size = torch.zeros_like(reward_formation)
        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values
        separation_reward = -(separation < self.safe_distance).float() #[env_num, drone_num]
        too_close = separation < self.cfg.task.hard_safe_distance # [env_num, drone_num]
        too_close_reward = too_close.float()

        # flight objective
        head_error = torch.norm(self.rheading, dim=-1)
        reward_heading = torch.clip(1-head_error, min=0) 
        pos_diff = self.root_states[..., :3] - self.final_pos
        # pos_error = torch.norm(pos_diff, p = 2, dim=-1)
        indi_p_reward = 1 / (1 + torch.norm(pos_diff, dim=-1))
        vel_diff = self.root_states[..., 7:10] - self.target_vel #[env_num, drone_num, 3]
        # acceptable_vel = (torch.abs(vel_diff[:, :, 1]) < self.cfg.task.acceptable_v_diff) # [env_num, drone_num]
        # indi_v_reward = 1 / (1 + torch.norm(vel_diff, dim=-1)) * self.velocity_coeff # [env_num, drone_num]
        indi_v_reward = torch.clip(torch.clip(torch.norm(self.target_vel, dim=-1), min=1) - torch.norm(vel_diff, dim=-1), min=0)

        height = pos[..., 2]   # [num_envs, drone.n]
        height_diff = height - self.target_height
        height_reward = torch.clip(1 - height_diff.abs(), min=0)

        # not_straight = (pos[..., 0] < - (self.border + 0.5)) | (pos[..., 0] > (self.border + 0.5))
        # not_straight_reward = not_straight
        if self.real_flight:
            crash = (pos[..., 2] < 0.2) | (pos[..., 2] > 1.8)
        else:
            crash = (pos[..., 2] < 0.2) | (pos[..., 2] > 2.8)  # [env_num, drone_num]
        # crash_reward = crash # crash_penalty < 0
        
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        acceptable_mean_vel = (torch.abs(torch.mean(vel_diff[:, :, 1], dim=1)) < self.cfg.task.acceptable_v_diff).unsqueeze(-1) # vel_diff: [env_num, drone.n, 3]
        # terminated = truncated & acceptable_mean_vel
        # no_enough_vel_reward = truncated & (~acceptable_mean_vel)
        # terminated_reward = terminated

        # obstacle objective
        if self.cfg.task.ball_num > 0:
            ball_vel = self.ball.get_linear_velocities().view(-1, self.ball_num, 3) # [env_num, ball_num, 3]
            reshape_ball_world_poses = (self.ball.get_world_poses()[0].view(-1, self.ball_num, 3), self.ball.get_world_poses()[1].view(-1, self.ball_num, 4))
            ball_pos, balls_rot = self.get_env_poses(reshape_ball_world_poses) # [env_num, ball_num, 3]
            # ball_pos = self.ball.get_world_poses()[0].view(-1, self.ball_num, 3)

            should_neglect = ((ball_vel[...,2] < 1e-6) & (ball_pos[...,2] < 0.5)) # [env_num, ball_num] 
            self.ball_alarm[should_neglect] = 0 # [env_num, ball_num]
            self.ball_alarm[~should_neglect] = 1
            
            ball_mask = (self.ball_alarm & self.ball_reward_flag) # [env_num, 1, ball_num]
            # compute ball hard reward (< self.obs_safe_distance)
            should_penalise = self.relative_b_dis < self.obs_safe_distance # [env_num, drone_num, ball_num]
            ball_hard_reward = torch.zeros(self.num_envs, self.drone.n, self.ball_num, device=self.device)
            ball_hard_reward[should_penalise] = -self.ball_hard_coeff

            # compute ball soft reward (encourage > self.soft_obs_safe_distance)
            indi_b_dis = self.relative_b_dis #[env_num, drone_num, ball_num]
            # smaller than ball_safe_dist, only consider hard reward
            k = 0.5 * self.ball_hard_coeff / (self.soft_obs_safe_distance-self.obs_safe_distance)
            # between ball_safe_dist and soft_ball_safe_dist, apply linear penalty
            indi_b_reward = (torch.clamp(indi_b_dis, min=self.obs_safe_distance, max=self.soft_obs_safe_distance) - self.soft_obs_safe_distance) * k
            # larger than soft_ball_safe_dist, apply linear
            indi_b_reward += torch.clamp(indi_b_dis-self.soft_obs_safe_distance, min=0)
            
            total_ball_reward = ball_hard_reward + indi_b_reward
            total_ball_reward *= ball_mask.unsqueeze(1)
            ball_reward, _ = torch.min(total_ball_reward, dim=-1) # [env_num, drone_num]

            ball_any_mask = ball_mask.any(dim=-1).unsqueeze(-1) # [env_num, 1]

        else: # self.ball_num == 0
            ball_any_mask = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)
            ball_reward = torch.zeros(self.num_envs, self.drone.n, device=self.device)
            
        if self.static_obs_num > 0: # 球和静态障碍物最好可以分开算，因为球要考虑 mask 的问题，cube 不太用
            # hit_c = torch.sum(self.relative_c_dis < self.obs_hit_distance, dim=-1)
            cube_hard_reward = (torch.clamp(self.relative_c_dis, min=self.obs_hit_distance, max=self.obs_safe_distance) - self.obs_safe_distance)
            cube_reward = torch.mean(cube_hard_reward, dim=-1) # [env, drone_num]
            if self.cfg.task.use_cube_reward_mask:
                cube_any_mask = (self.relative_c_dis < (self.soft_obs_safe_distance + 1.)).any(dim=-1).any(dim=-1).unsqueeze(-1) # [env_num, 1]
            else:
                cube_any_mask = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)
        else:
            cube_any_mask = torch.zeros(self.num_envs, 1, dtype=bool, device=self.device)
            cube_reward = torch.zeros(self.num_envs, self.drone.n, device=self.device)

        ball_coeff = ((~(ball_any_mask | cube_any_mask)) * self.cfg.task.no_ball_coeff
                                            + (ball_any_mask | cube_any_mask) * self.cfg.task.has_ball_coeff)

        # after throw mask, still only consider ball case
        if self.ball_num > 0:
            after_throw_mask = (~ball_any_mask) & ((~torch.isnan(self.t_launched)).all(dim=-1, keepdim=True))
        else:
            after_throw_mask = torch.zeros_like(ball_any_mask)        
        
        if self.ball_num > 0:
            hit_b = torch.sum(self.relative_b_dis < self.obs_hit_distance, dim=-1)
        else:
            hit_b = torch.zeros(self.num_envs, 1, device=self.device, dtype=bool)
        if self.static_obs_num > 0:
            hit_c = torch.sum(self.relative_c_dis < self.obs_hit_distance, dim=-1)
        else:
            hit_c = torch.zeros(self.num_envs, 1, device=self.device, dtype=bool)

        hit = self.relative_obs_dis < self.obs_hit_distance # [env_num, drone_num, ball_num]
        hit = torch.sum(hit, dim=-1)
        hit_reward = hit.float()

        bad_terminate = crash | too_close | hit
        bad_terminate = torch.sum(bad_terminate, dim=-1) > 0

        done = crash | too_close | hit | truncated
        done = torch.sum(done, dim=-1) > 0

        # survival_reward = 1 - bad_terminate.float().unsqueeze(-1)
        bad_terminate_penalty = bad_terminate.float().unsqueeze(-1)

        dynamic_coeff = ((~(ball_any_mask | cube_any_mask)) * self.cfg.task.no_ball_coeff
                          + (ball_any_mask | cube_any_mask) * self.cfg.task.has_ball_coeff)

        morl_obstacle = (ball_reward * self.cfg.task.ball_reward_coeff
            + cube_reward * self.cfg.task.static_hard_coeff
            + hit_reward * self.cfg.task.hit_penalty
            + truncated * self.cfg.task.truncated_reward
            - bad_terminate_penalty * self.cfg.task.truncated_reward)

        self.stats["ball_reward"].lerp_(ball_reward, (1-self.alpha))
        self.stats["cube_reward"].lerp_(cube_reward, (1-self.alpha))
        self.stats["hit_reward"].lerp_(hit_reward, (1-self.alpha))
        self.stats["morl_obstacle_reward"].lerp_(morl_obstacle, (1-self.alpha))

        # action objective

        # reward_effort = torch.exp(-self.effort) # throttle sum
        reward_effort = torch.clip(2.5-self.effort, min=0) # 2.5->1.4
        # reward_throttle_smoothness = torch.exp(-self.drone.throttle_difference)
        reward_throttle_smoothness = torch.clip(.5-self.drone.throttle_difference, min=0) # 0.4->0.1
        output_diff = torch.norm((self.last_network - self.current_network), dim=-1)
        # reward_action_smoothness = torch.exp(-output_diff)
        reward_action_smoothness = torch.clip(2.5-output_diff, min=0) # 2.3->1.8
        y_spin = torch.abs(self.drone.vel[..., -1])
        # reward_spin = 1 / (1 + y_spin)
        reward_spin = torch.clip(1.5-y_spin, min=0) # 1.5->0.25

        morl_smooth = (reward_effort * self.reward_effort_weight
            + reward_action_smoothness * self.cfg.task.reward_action_smoothness_weight
            + reward_spin * self.cfg.task.spin_reward_coeff
            + reward_throttle_smoothness * self.cfg.task.reward_throttle_smoothness_weight
            + truncated * self.cfg.task.truncated_reward
            - bad_terminate_penalty * self.cfg.task.truncated_reward)

        self.stats["reward_effort"].lerp_(reward_effort, (1-self.alpha))
        self.stats["reward_action_smoothness"].lerp_(reward_action_smoothness, (1-self.alpha))
        self.stats["reward_spin"].lerp_(reward_spin, (1-self.alpha))
        self.stats["reward_throttle_smoothness"].lerp_(reward_throttle_smoothness, (1-self.alpha))
        self.stats["morl_smooth_reward"].lerp_(morl_smooth, (1-self.alpha))

        morl_formation = (
            (reward_size * ball_coeff + reward_size * after_throw_mask * self.cfg.task.after_throw_coeff) * self.cfg.task.formation_size_coeff
            + reward_formation * self.formation_coeff * dynamic_coeff
            + separation_reward * self.cfg.task.separation_coeff
            + too_close_reward * self.cfg.task.too_close_penalty
            + truncated * self.cfg.task.truncated_reward
            - bad_terminate_penalty * self.cfg.task.truncated_reward)
        
        self.stats["cost_l"].lerp_(cost_l, (1-self.alpha))
        self.stats["cost_l_unnormalized"].lerp_(cost_l_unnormalized, (1-self.alpha))
        self.stats["reward_formation"].lerp_(reward_formation, (1-self.alpha))
        self.stats["reward_size"].lerp_(reward_size, (1-self.alpha))
        self.stats["separation_reward"].lerp_(separation_reward, (1-self.alpha))
        self.stats["morl_formation_reward"].lerp_(morl_formation, (1-self.alpha))

        morl_forward = (
            height_reward * self.height_coeff * ball_coeff
            + indi_p_reward * self.cfg.task.position_reward_coeff * truncated
            + indi_v_reward * self.velocity_coeff * ball_coeff
            + reward_heading * self.cfg.task.heading_coeff) * dynamic_coeff
        
        morl_forward += (truncated * self.cfg.task.truncated_reward
            - bad_terminate_penalty * self.cfg.task.truncated_reward) 
        
        self.stats["height_reward"].lerp_(height_reward, (1-self.alpha))
        self.stats["pos_reward"].lerp_(indi_p_reward, (1-self.alpha))
        self.stats["vel_reward"].lerp_(indi_v_reward, (1-self.alpha))
        self.stats["reward_heading"].lerp_(reward_heading, (1-self.alpha))
        self.stats["morl_forward_reward"].lerp_(morl_forward, (1-self.alpha))

        # additional constraints regarding all objectives
        size_ratio = size / self.standard_formation_size
        formation_success = (cost_l < 0.15) & (size_ratio > 0.75) & (size_ratio < 1.25)
        action_norm = self.current_network.norm(dim=-1).mean(dim=-1, keepdim=True)
        action_success = action_norm < 1.7
        
        success = truncated & acceptable_mean_vel & formation_success & action_success

        self.stats["formation_success"][:] = (formation_success.float())
        self.stats["action_success"][:] = (action_success.float())
        self.stats["success"][:] = (success.float())

        if self.cfg.task.rescale_reward:
            reward = (morl_smooth * self.morl_smooth_coeff / 14
                    + morl_obstacle * self.morl_obstacle_coeff / 40
                    + morl_forward * self.morl_forward_coeff / 25
                    + morl_formation * self.morl_formation_coeff / 4
                    ).reshape(-1, self.drone.n)
        else:
            reward = (morl_smooth * self.morl_smooth_coeff
                    + morl_obstacle * self.morl_obstacle_coeff
                    + morl_forward * self.morl_forward_coeff
                    + morl_formation * self.morl_formation_coeff
                    ).reshape(-1, self.drone.n)
        self.stats["reward"].lerp_(reward, (1-self.alpha))
        self.stats["return"] = self.stats["return"] * self.gamma + torch.mean(reward, dim=1, keepdim=True)

        # self.stats["survival_reward"].lerp_(survival_reward, (1-self.alpha))
        # self.stats["survival_return"].add_(torch.mean(survival_reward))

        # formation_dis = compute_formation_dis(pos, self.formation).expand(-1, self.ball_num) # [env_num, ball_num]
        formation_dis = compute_formation_dis(pos, self.formation) # [env_num, 1]
        # print(formation_dis.shape)
        drone_moved = ((~torch.isnan(self.t_launched)) & (formation_dis > 0.35) & (torch.isnan(self.t_moved))) # [env_num, ball_num]
        self.t_moved[drone_moved] = self.progress_buf.unsqueeze(-1).expand(self.num_envs, self.ball_num)[drone_moved]
        self.t_moved[drone_moved] = self.t_moved[drone_moved] - self.t_launched[drone_moved]
        self.t_difference[drone_moved] = self.t_moved[drone_moved] - self.t_hit[drone_moved]

        self.frame_counter += (torch.sum((done.squeeze()).float() * self.progress_buf)).item()

        self.stats["t_launched"][:] = torch.nanmean(self.t_launched.unsqueeze(1), keepdim=True)
        self.stats["t_moved"][:] = torch.nanmean(self.t_moved.unsqueeze(1), keepdim=True)
        self.stats["t_difference"][:] =  torch.nanmean(self.t_difference.unsqueeze(1), keepdim=True)
        self.stats["t_hit"][:] =  torch.nanmean(self.t_hit.unsqueeze(1), keepdim=True)
        self.stats["truncated"][:] = (truncated.float())
        # self.stats["terminated"][:] = (terminated.float())
        self.stats["forward_success"][:] = (acceptable_mean_vel.float())
        self.stats["crash"][:] = torch.any(crash, dim=-1, keepdim=True).float()
        # self.stats["not straight"][:] = torch.any(not_straight, dim=-1, keepdim=True).float()
        self.stats["hit"][:] = torch.any(hit, dim=-1, keepdim=True).float()
        self.stats["hit_b"][:] = torch.any(hit_b, dim=-1, keepdim=True).float()
        self.stats["hit_c"][:] = torch.any(hit_c, dim=-1, keepdim=True).float()
        self.stats["too close"][:] = (too_close.float())
        self.stats["done"][:] = (done.float()).unsqueeze(1)
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)

        # self.stats["morl_smooth"].add_(torch.mean(morl_smooth, dim=1, keepdim=True))
        # self.stats["morl_formation"].add_(torch.mean(morl_formation, dim=1, keepdim=True))
        # self.stats["morl_obstacle"].add_(torch.mean(morl_obstacle, dim=1, keepdim=True))
        # self.stats["morl_forward"].add_(torch.mean(morl_forward, dim=1, keepdim=True))

        self.stats["morl_smooth"] = self.stats["morl_smooth"] * self.gamma + morl_smooth
        self.stats["morl_formation"] = self.stats["morl_formation"] * self.gamma + morl_formation
        self.stats["morl_obstacle"] = self.stats["morl_obstacle"] * self.gamma + morl_obstacle
        self.stats["morl_forward"] = self.stats["morl_forward"] * self.gamma + morl_forward

        self.stats["size"][:] = size
        if self.cfg.task.ball_num > 0:
            self.stats["indi_b_dist"].add_((torch.mean(torch.mean(indi_b_dis, dim=1), dim=1)/self.progress_buf).unsqueeze(-1))
        self.stats["curr_formation_dist"][:] = formation_dis
        
        assert self.ball_reward_flag.dtype == torch.bool
        assert self.ball_alarm.dtype == torch.bool

        return TensorDict(
            {
                "agents": {
                    "reward": reward
                },
                "done": done,
            },
            self.batch_size
        )


def cost_formation_laplacian(
    p: torch.Tensor,
    desired_L: torch.Tensor,
    normalized=False,
) -> torch.Tensor:
    """
    A scale and translation invariant formation similarity cost
    """
    L = laplacian(p, normalized)
    cost = torch.linalg.matrix_norm(desired_L - L)
    return cost.unsqueeze(-1)


def laplacian(p: torch.Tensor, normalize=False):
    """
    symmetric normalized laplacian

    p: (n, dim)
    """
    assert p.dim() == 2
    A = torch.cdist(p, p) # A[i, j] = norm_2(p[i], p[j]), A.shape = [n, n]
    D = torch.sum(A, dim=-1) # D[i] = \sum_{j=1}^n norm_2(p[i], p[j]), D.shape = [n, ]
    if normalize:
        DD = D**-0.5
        A = torch.einsum("i,ij->ij", DD, A)
        A = torch.einsum("ij,j->ij", A, DD)
        L = torch.eye(p.shape[0], device=p.device) - A
    else:
        L = D - A
    return L
    
def cost_formation_hausdorff(p: torch.Tensor, desired_p: torch.Tensor) -> torch.Tensor:
    p = p - p.mean(-2, keepdim=True)
    desired_p = desired_p - desired_p.mean(-2, keepdim=True)
    cost = torch.max(directed_hausdorff(p, desired_p), directed_hausdorff(desired_p, p))
    return cost.unsqueeze(-1)


def directed_hausdorff(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    p: (*, n, dim)
    q: (*, m, dim)
    """
    d = torch.cdist(p, q, p=2).min(-1).values.max(-1).values
    return d

def compute_formation_dis(pos: torch.Tensor, formation_p: torch.Tensor):
    rel_pos = pos - pos.mean(-2, keepdim=True) # [env_num, drone_num, 3]
    rel_f = formation_p - formation_p.mean(-2, keepdim=True) # [drone_num, 3]
    # [env_num, drone_num]
    dist = torch.norm(rel_f-rel_pos, p=2, dim=-1)
    dist = torch.mean(dist, dim=-1, keepdim=True)
    return dist