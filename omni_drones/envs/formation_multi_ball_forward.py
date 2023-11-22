from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core import objects
import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import torch
import torch.distributions as D
from omni_drones.views import RigidPrimView
import numpy as np

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, make_cells, euler_to_quaternion
from omni_drones.robots.drone import MultirotorBase
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from omni.isaac.debug_draw import _debug_draw

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
    [0, 0, 0],
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [-1, 1, 0],
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
}

def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]

class FormationMultiBallForward(IsaacEnv):
    def __init__(self, cfg, headless):
        self.ball_num = cfg.task.ball_num
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.time_encoding = self.cfg.task.time_encoding
        self.safe_distance = self.cfg.task.safe_distance
        self.formation_size = self.cfg.task.formation_size
        self.ball_hit_distance = self.cfg.task.ball_hit_distance
        self.ball_safe_distance = self.cfg.task.ball_safe_distance
        self.soft_ball_safe_distance = self.cfg.task.soft_ball_safe_distance
        self.ball_reward_coeff = self.cfg.task.ball_reward_coeff
        self.ball_hard_reward_coeff = self.cfg.task.ball_hard_reward_coeff
        self.ball_speed = self.cfg.task.ball_speed
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.total_frame = self.cfg.total_frames
        self.drone.initialize() 
        self.throw_threshold = self.cfg.task.throw_threshold
        # create and initialize additional views
        self.ball = RigidPrimView(
            "/World/envs/env_*/ball_*",
            reset_xform_properties=False,
        )

        self.ball.initialize()
        self.ball.set_masses(torch.ones_like(self.ball.get_masses()))

        self.init_poses = self.drone.get_world_poses(clone=True)

        # initial state distribution
        self.cells = (
            make_cells([-2, -2, 0.5], [2, 2, 2], [0.5, 0.5, 0.25])
            .flatten(0, -2)
            .to(self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.0, -.0, 0.], device=self.device) * torch.pi,
            torch.tensor([0.0, 0.0, 2.], device=self.device) * torch.pi
        )
        self.target_heading = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)
        self.flag = torch.zeros((self.num_envs, self.ball_num), dtype=bool, device=self.device)
        self.t_throw = torch.zeros((self.num_envs, self.ball_num), device=self.device)
        self.cost_h = torch.ones(self.num_envs, dtype=bool, device=self.device)
        self.t_formed_indicator = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        self.t_formed = torch.full(size=(self.num_envs,1), fill_value=torch.nan, device=self.device).squeeze(1)
        self.t_launched = torch.full(size=(self.num_envs, self.ball_num), fill_value=torch.nan, device=self.device).squeeze(1)
        self.ball_reward_flag = torch.zeros((self.num_envs, self.ball_num), dtype=bool, device=self.device)
        self.ball_alarm = torch.ones((self.num_envs, self.ball_num), dtype=bool, device=self.device)
        self.height_penalty = torch.zeros(self.num_envs, self.drone.n, device=self.device)
        self.separation_penalty = torch.zeros(self.num_envs, self.drone.n, self.drone.n-1, device=self.device)
        self.t_moved = torch.full(size=(self.num_envs,self.ball_num), fill_value=torch.nan, device=self.device).squeeze(1)
        self.t_difference = torch.full(size=(self.num_envs,self.ball_num), fill_value=torch.nan, device=self.device).squeeze(1)
        self.t_hit = torch.full(size=(self.num_envs, self.ball_num), fill_value=torch.nan, device=self.device).squeeze(1)
        self.frame_counter = 0

        self.alpha = 0.
        self.last_cost_h = torch.zeros(self.num_envs, 1, device=self.device)

    def _design_scene(self) -> Optional[List[str]]:
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        scene_utils.design_scene()

        # Set height of the drones
        self.target_pos_single = torch.tensor([0., 0., 1.5], device=self.device)
        self.target_vel = list(self.cfg.task.target_vel)
        self.target_vel = torch.Tensor(self.target_vel)
        self.target_vel = self.target_vel.to(device=self.device)

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

        self.init_pos_single = torch.zeros_like(self.target_pos_single)
        self.init_pos_single[-1] = self.target_pos_single[-1].clone()
        self.target_height = self.init_pos_single.clone()

        self.formation = self.formation*self.cfg.task.formation_size/2

        for ball_id in range(self.ball_num):
            ball = objects.DynamicSphere(
                prim_path=f"/World/envs/env_0/ball_{ball_id}",  
                position=torch.tensor([ball_id, 0., -1.]),
                radius = 0.15,
                color=torch.tensor([1., 0., 0.]),
            )

        self.drone.spawn(translations=self.formation+self.init_pos_single)
        self.target_pos = self.target_pos_single.expand(self.num_envs, self.drone.n, 3) + self.formation
        self.drone_id = torch.Tensor(np.arange(self.drone.n)).to(self.device)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[0]
        obs_self_dim = drone_state_dim
        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            obs_self_dim += self.time_encoding_dim
        if self.cfg.algo.share_actor:
            self.id_dim = 3
            obs_self_dim += self.id_dim

        state_dim = drone_state_dim

        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": CompositeSpec({
                    "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)), # 23
                    "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 13+1)), # 5 * 14 =70
                    "attn_obs_ball": UnboundedContinuousTensorSpec((self.ball_num, 3+1+3)), # 7
                }).expand(self.drone.n), 
                "state": CompositeSpec({
                    "drones": UnboundedContinuousTensorSpec((self.drone.n, state_dim))
                })
            }
        }).expand(self.num_envs).to(self.device)

        self.action_spec = CompositeSpec({
            "agents": {
                "action": torch.stack([self.drone.action_spec]*self.drone.n, dim=0),
            }
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((self.drone.n, 1))
            }
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "state")
        )

        stats_spec = CompositeSpec({
            "cost_hausdorff": UnboundedContinuousTensorSpec(1),
            # "mean_dis_error": UnboundedContinuousTensorSpec(1),
            "ball_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "soft_ball_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "drone_reward": UnboundedContinuousTensorSpec(self.drone.n),
            "formation reward": UnboundedContinuousTensorSpec(1),
            # "heading reward": UnboundedContinuousTensorSpec(1),
            "t_launched": UnboundedContinuousTensorSpec(1),
            "t_moved": UnboundedContinuousTensorSpec(1),
            "t_difference": UnboundedContinuousTensorSpec(1),
            "t_hit": UnboundedContinuousTensorSpec(1),
            "terminated": UnboundedContinuousTensorSpec(1),
            "crash": UnboundedContinuousTensorSpec(1),
            "hit": UnboundedContinuousTensorSpec(1),
            "too close": UnboundedContinuousTensorSpec(1),
            "done": UnboundedContinuousTensorSpec(1),
            "height_penalty": UnboundedContinuousTensorSpec(self.drone.n),
            "separation_penalty": UnboundedContinuousTensorSpec(1),
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "ball_return": UnboundedContinuousTensorSpec(1),
            "drone_return": UnboundedContinuousTensorSpec(1),
            "hard_ball_return": UnboundedContinuousTensorSpec(1),
            "indi_b_dist": UnboundedContinuousTensorSpec(1),
            "formation_dist": UnboundedContinuousTensorSpec(1)
            # "soft_ball_return": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()


    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.t_formed_indicator[env_ids] = False
        self.ball_reward_flag[env_ids] = 0.
        pos = (
            (self.formation+self.init_pos_single).expand(len(env_ids), *self.formation.shape) # (k, 3) -> (len(env_ids), k, 3)
            + self.envs_positions[env_ids].unsqueeze(1)
        )
        rpy = self.init_rpy_dist.sample((*env_ids.shape, self.drone.n))
        rot = euler_to_quaternion(rpy)
        vel = torch.zeros(len(env_ids), self.drone.n, 6, device=self.device)
        self.drone.set_world_poses(pos, rot, env_ids)
        self.drone.set_velocities(vel, env_ids)
        self.last_cost_h[env_ids] = vmap(cost_formation_hausdorff)(
            pos, desired_p=self.formation
        )

        pos = (
            torch.tensor([0., 0., -10], device=self.device).expand(len(env_ids), 3)
            + self.envs_positions[env_ids]
        )
        vel = torch.zeros(len(env_ids), 6, device=self.device)
        self.ball.set_world_poses(pos, env_indices=env_ids)
        self.ball.set_velocities(vel, env_ids)

        target_height = torch.tensor(self.cfg.task.target_height, device = self.device)
        self.flag[env_ids] = False

        self.stats[env_ids] = 0.
        self.t_formed[env_ids]=torch.nan
        self.t_launched[env_ids]=torch.nan
        self.t_moved[env_ids]=torch.nan
        self.t_difference[env_ids]=torch.nan
        self.t_hit[env_ids] =torch.nan
        self.ball_alarm[env_ids] = 1
        self.height_penalty[:] = 0.
        self.separation_penalty[:] = 0.
        self.t_throw = torch.rand(self.num_envs, self.ball_num, device=self.device) * 300 + self.throw_threshold
        
    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)
        self.throw_multi_ball()

    def throw_multi_ball(self):
        pos, rot = self.get_env_poses(self.drone.get_world_poses())
        flag = (self.progress_buf.unsqueeze(-1).expand(self.num_envs, self.ball_num) >= self.t_throw)
        should_throw = flag & (~self.flag)
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
            centre_D = self.drone.pos[throw_indices[0]][..., :2].mean(1)

            # Approximate the maximum distance between drones after forming square

            # target = torch.rand(centre_D.shape, device=self.device)*2
            target_ball_pos = torch.zeros(throw_ball_num,3, device=self.device)
            ball_pos = torch.zeros(throw_ball_num,3, device=self.device)

            # given t_hit, randomize ball init position & final position
            t_hit = torch.rand(throw_ball_num, device=self.device) * 1.5 + 0.5
            # firstly, calculate vel_z
            #============
            # 注意 ball_vel 和 ball_target_vel 的差别在 vz 上
            ball_target_vel = torch.ones(throw_ball_num, 3, device=self.device)
            ball_target_vel[:, 2] = - torch.rand(throw_ball_num, device=self.device) - 1. #[-2, -1]
            ball_target_vel[:, :2] = 2*(torch.rand(throw_ball_num, 2, device=self.device)-0.5) #[-1, 1]
            ball_target_vel = ball_target_vel/torch.norm(ball_target_vel, p=2, dim=1, keepdim=True) * self.ball_speed
            
            ball_vel = torch.ones(throw_ball_num, 6, device=self.device)
            ball_vel[:, :3] = ball_target_vel.clone()
            ball_vel[:, 2] += 9.81*t_hit

            drone_x_speed = torch.mean(self.root_states[throw_indices[0]][..., 7], 1)
            drone_x_dist = drone_x_speed * t_hit

            drone_y_speed = torch.mean(self.root_states[throw_indices[0]][..., 8], 1)
            drone_y_dist = drone_y_speed * t_hit
            
            target_ball_pos[:, 0] = torch.rand(throw_ball_num, device=self.device)*2-1 + drone_x_dist
            target_ball_pos[:, 1] = torch.rand(throw_ball_num, device=self.device)*2-1 + drone_y_dist
            target_ball_pos[:, :2] += centre_D
            target_ball_pos[:, 2] = self.drone.pos[throw_indices[0]][..., 2].mean(1)

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
        self.root_states = self.drone.get_state()  # Include pos, rot, vel, ...
        self.info["drone_state"][:] = self.root_states[..., :13]
        pos = self.drone.pos  # Position of all the drones relative to the environment's local coordinate
        
        # self.rheading = self.target_heading - self.root_states[..., 13:16]

        # indi_rel_pos = pos - self.target_pos
        obs_self = [self.root_states]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs_self.append(t.expand(-1, self.drone.n, self.time_encoding_dim))
        if self.cfg.algo.share_actor:
            obs_self.append(self.drone_id.reshape(1, -1, 1).expand(self.root_states.shape[0], -1, self.id_dim))
        obs_self = torch.cat(obs_self, dim=-1)

        relative_pos = vmap(cpos)(pos, pos)
        self.drone_pdist = vmap(off_diag)(torch.norm(relative_pos, dim=-1, keepdim=True))   # pair wise distance

        # Relative position between the ball and all the drones
        reshape_ball_world_poses = (self.ball.get_world_poses()[0].view(-1, self.ball_num, 3), self.ball.get_world_poses()[1].view(-1, self.ball_num, 4))
        balls_pos, balls_rot = self.get_env_poses(reshape_ball_world_poses) # [env_num, ball_num, 3]

        relative_b_pos = pos[..., :3].unsqueeze(2) - balls_pos.unsqueeze(1) # [env_num, drone_num, 1, 3] - [env_num, 1, ball_num, 3]
        balls_vel = self.ball.get_linear_velocities().view(-1, 1, self.ball_num, 3) # [env_num, 1, ball_num, 3]
        self.relative_b_dis = torch.norm(relative_b_pos, p=2, dim=-1) # [env_num, drone_num, ball_num, 3] -> [env_num, drone_num, ball_num]
        relative_b_dis = self.relative_b_dis # [env_num, drone_num, ball_num]
        relative_pos = vmap(off_diag)(relative_pos)

        obs_others = torch.cat([
            relative_pos,
            self.drone_pdist,
            vmap(others)(self.root_states[..., 3:13])
        ], dim=-1)
        
        obs_ball = torch.cat([
            relative_b_dis.unsqueeze(-1), 
            relative_b_pos, # [n, k, m, 3]
            balls_vel.expand_as(relative_b_pos)
        ], dim=-1).view(self.num_envs, self.drone.n, self.ball_num, 7) #[env, agent, ball_num, *]
        
        manual_mask = torch.isnan(self.t_launched)
        if manual_mask.any():
            manual_mask = torch.nonzero(manual_mask, as_tuple=True)
            obs_ball[manual_mask[0], :, manual_mask[1]] = -1.
        assert not torch.isnan(obs_self).any()
        assert not torch.isnan(obs_ball).any()
        # assert self.mask.dtype == torch.bool
        obs = TensorDict({ 
            "obs_self": obs_self.unsqueeze(2),  # [N, K, 1, obs_self_dim]
            "obs_others": obs_others, # [N, K, K-1, obs_others_dim]
            "attn_obs_ball": obs_ball, # [N, K, ball_num, *]
        }, [self.num_envs, self.drone.n]) # [N, K, n_i, m_i]

        state = TensorDict({"drones": self.root_states}, self.batch_size)

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

        self.cost_h = vmap(cost_formation_hausdorff)(pos, desired_p=self.formation)
        reward_formation =  1 / (1 + torch.square(self.cost_h * 1.6))

        # change to velocity reward
        vel_diff = self.root_states[..., 7:10] - self.target_vel #[env_num, drone_num, 3]
        indi_v_reward = 1 / (1 + torch.norm(vel_diff, p = 2, dim=-1)) # [env_num, drone_num]

        # cost if height drop or too high
        height = pos[..., 2]   # [num_envs, drone.n]
        height_penalty = ((height < 0.5) | (height > 2.5))
        if height_penalty.any():
            height_penalty = torch.nonzero(height_penalty, as_tuple=True)
            self.height_penalty[height_penalty] = -1
        height_reward = self.height_penalty # torch.sum(self.height_penalty, dim=-1, keepdim=True)

        ball_vel = self.ball.get_linear_velocities().view(-1, self.ball_num, 3) # [env_num, ball_num, 3]
        reshape_ball_world_poses = (self.ball.get_world_poses()[0].view(-1, self.ball_num, 3), self.ball.get_world_poses()[1].view(-1, self.ball_num, 4))
        ball_pos, ball_rot = self.get_env_poses(reshape_ball_world_poses) # [env_num, ball_num, 3]

        should_neglect = ((ball_vel[:,2] < -0.1) & (ball_pos[:,2] < 0.5)) # ball_pos[:, 2] < 1.45
        if should_neglect.any():
            should_neglect = torch.nonzero(should_neglect, as_tuple=True)
            self.ball_alarm[should_neglect] = 0
            # self.mask[should_neglect, -1] = True
        # compute ball hard reward (< self.ball_safe_distance)
        should_penalise = self.relative_b_dis < self.ball_safe_distance
        ball_hard_reward = torch.zeros(self.num_envs, self.drone.n, self.ball_num, device=self.device)
        if should_penalise.any():
            should_penalise = torch.nonzero(should_penalise, as_tuple=True)
            ball_hard_reward[should_penalise] = -self.ball_hard_reward_coeff
        ball_hard_reward = ball_hard_reward * (self.ball_alarm.float()).unsqueeze(1) * (self.ball_reward_flag.float()).unsqueeze(1)
        ball_hard_reward = torch.sum(ball_hard_reward, dim=-1) # [n, k, m] -> [n, k,]

        # compute ball soft reward (encourage > self.soft_ball_safe_distance)
        indi_b_dis = self.relative_b_dis #[env_num, drone_num]
        # smaller than ball_safe_dist, apply exponential penaty (it should be consider together with hard penalty, so it is positive)
        k = 0.5 * self.ball_hard_reward_coeff/(self.soft_ball_safe_distance-self.ball_safe_distance)
        indi_b_reward = torch.exp(torch.clamp(indi_b_dis, max=self.ball_safe_distance) - self.ball_safe_distance) * k
        # between ball_safe_dist and soft_ball_safe_dist, apply linear penalty
        indi_b_reward = (torch.clamp(indi_b_dis, min=self.ball_safe_distance, max=self.soft_ball_safe_distance) - self.soft_ball_safe_distance) * k
        # larger than soft_ball_safe_dist, apply linear
        indi_b_reward += torch.clamp(indi_b_dis-self.soft_ball_safe_distance, min=0, max=1.0) * self.ball_reward_coeff 
        indi_b_reward *= (self.ball_alarm.float()).unsqueeze(1) * (self.ball_reward_flag.float()).unsqueeze(1)
        indi_b_reward, _ = torch.max(indi_b_reward, dim=-1) # [n, k, m] -> [n, k,]

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort)

        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values

        terminated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated_reward = self.cfg.task.terminated_reward * terminated
        
        crash = pos[..., 2] < 0.2
        crash_reward = self.cfg.task.crash_penalty * crash # crash_penalty < 0
        hit = self.relative_b_dis < self.ball_hit_distance
        hit = torch.sum(hit, dim=-1)
        hit_reward = self.cfg.task.hit_penalty * hit
        too_close = separation<self.safe_distance
        too_close_reward = self.cfg.task.too_close_penalty * hit
        done = terminated | crash | too_close | hit
        done = torch.sum(done, dim=-1) > 0

        reward = (
            ball_hard_reward # < ball_soft_distance, penalize a big constant # [n, k]
            + indi_b_reward # > ball_extra_soft_dictance # [n, k]
            + height_reward # [n, k]
            + reward_formation * 2. # [n, 1]
            + reward_effort # [n, k]
            + crash_reward # [n, k]
            + hit_reward # [n, k]
            + too_close_reward # [n, 1]
            + terminated_reward # [n, 1]
            + indi_v_reward # [n, k]
        ).unsqueeze(1).reshape(-1, self.drone.n, 1) 

        formation_dis = compute_formation_dis(pos, self.formation).expand(-1, self.ball_num)
        drone_moved = ((~torch.isnan(self.t_launched)) & (formation_dis > 0.35) & (torch.isnan(self.t_moved)))
        if drone_moved.any():
            drone_moved = torch.nonzero(drone_moved, as_tuple=True)
            self.t_moved[drone_moved] = self.progress_buf.unsqueeze(-1).expand(self.num_envs, self.ball_num)[drone_moved]
            self.t_moved[drone_moved] = self.t_moved[drone_moved] - self.t_launched[drone_moved]
            self.t_difference[drone_moved] = self.t_moved[drone_moved] - self.t_hit[drone_moved]

        self.last_cost_h[:] = self.cost_h

        formed_indicator = (self.t_formed_indicator == False) & (self.progress_buf >= 100)
        if formed_indicator.any():
            formed_indicator = torch.nonzero(formed_indicator, as_tuple=True)
            self.t_formed[formed_indicator] = self.progress_buf[formed_indicator]
            self.t_formed_indicator[formed_indicator] = True        

        self.frame_counter += (torch.sum((done.squeeze()).float() * self.progress_buf)).item()
        
        # assert torch.isclose(-torch.log(indi_d_reward), mean_dis_error, atol=1e-5).all()
        self.stats["cost_hausdorff"].lerp_(self.cost_h, (1-self.alpha))
        # self.stats["mean_dis_error"].lerp_(mean_dis_error, (1-self.alpha))
        self.stats["ball_reward"].lerp_(ball_hard_reward, (1-self.alpha))
        self.stats["soft_ball_reward"].lerp_(indi_b_reward, (1-self.alpha))
        self.stats["drone_reward"].lerp_(indi_v_reward, (1-self.alpha))
        self.stats["formation reward"].lerp_(reward_formation, (1-self.alpha))
        # self.stats["heading reward"].lerp_(reward_heading, (1-self.alpha))
        self.stats["t_launched"][:] = torch.nanmean(self.t_launched.unsqueeze(1), keepdim=True)
        self.stats["t_moved"][:] = torch.nanmean(self.t_moved.unsqueeze(1), keepdim=True)
        self.stats["t_difference"][:] =  torch.nanmean(self.t_difference.unsqueeze(1), keepdim=True)
        self.stats["t_hit"][:] =  torch.nanmean(self.t_hit.unsqueeze(1), keepdim=True)
        self.stats["terminated"][:] = (terminated.float())
        self.stats["crash"][:] = torch.mean(crash.float())
        self.stats["hit"][:] = torch.mean(hit.float())
        self.stats["too close"][:] = (too_close.float())
        self.stats["done"][:] = (done.float()).unsqueeze(1)
        self.stats["height_penalty"].lerp_(height_reward, (1-self.alpha))
        # self.stats["separation_penalty"].lerp_(separation_reward, (1-self.alpha))
        self.stats["return"].add_(torch.mean(reward, dim=1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        # self.stats["ball_return"].add_(torch.mean(((1/reward_separation) * (ball_hard_reward + indi_b_reward)).unsqueeze(1).expand(-1, self.drone.n, 1),dim=1))
        self.stats["drone_return"].add_(torch.mean(indi_v_reward))
        # self.stats["soft_ball_return"].add_(torch.mean(indi_b_reward.unsqueeze(1).expand(-1, self.drone.n, 1),dim=1))
        self.stats["hard_ball_return"].add_(torch.mean(ball_hard_reward, dim=1).unsqueeze(1))
        self.stats["indi_b_dist"][:] = torch.mean(torch.mean(indi_b_dis, dim=1), dim=1).unsqueeze(1)
        self.stats["formation_dist"][:] = formation_dis[:, :1]
        
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
    

def new_cost(
        d: torch.Tensor
) -> torch.Tensor:
    " Account for the distance between the drone's actual position and targeted position"
    d = torch.clamp(d.square()-0.15**2, min=0) # if the difference is less then 0.1, generating 0 cost  
    return torch.sum(d)     

def huber_cost(
        d: torch.Tensor
) -> torch.Tensor:
    " Account for the distance between the drone's actual position and targeted position"
    d = torch.clamp(d-0.15, min=0) # if the difference is less then 0.1, generating 0 cost  
    return torch.sum(d)    

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
    A = torch.cdist(p, p)
    D = torch.sum(A, dim=-1)
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