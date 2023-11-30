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
import collections

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
    'triangle': REGULAR_TRIANGLE,
}

def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]

class FormationForward(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.time_encoding = self.cfg.task.time_encoding
        self.safe_distance = self.cfg.task.safe_distance
        self.formation_size = self.cfg.task.formation_size
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.total_frame = self.cfg.total_frames
        self.drone.initialize() 

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
        self.flag = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        self.cost_h = torch.ones(self.num_envs, dtype=bool, device=self.device)
        self.t_formed_indicator = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        self.t_formed = torch.full(size=(self.num_envs,), fill_value=torch.nan, device=self.device)

        self.height_penalty = torch.zeros(self.num_envs, self.drone.n, device=self.device)
        self.frame_counter = 0

        # add some log helper variable
        self.formation_dist_sum = torch.zeros(size=(self.num_envs, 1), device=self.device)
        # self.stats = stats_spec.zero()
        self.alpha = 0.

        self.last_cost_h = torch.zeros(self.num_envs, 1, device=self.device)

    def _design_scene(self) -> Optional[List[str]]:
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        scene_utils.design_scene()

        # Set height of the drones
        self.target_pos_single = torch.tensor([0., 0., 1.0], device=self.device)
        self.target_vel = list(self.cfg.task.target_vel)
        self.target_vel = torch.Tensor(self.target_vel)
        self.target_vel = self.target_vel.to(device=self.device)

        # raise NotImplementedError
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

        # # target position for each drone in the pentagon
        # self.formation = self.formation*self.cfg.task.formation_size/2 + self.target_pos
        # target position for each drone in the pentagon
        self.init_pos_single = torch.zeros_like(self.target_pos_single)
        self.init_pos_single[-1] = self.target_pos_single[-1].clone()
        self.target_height = self.init_pos_single.clone()
        # self.middle_xy = ((self.init_pos_single+self.target_pos_single)/2)[:2].clone()
        
        self.formation = self.formation*self.cfg.task.formation_size/2

        self.drone.spawn(translations=self.formation+self.init_pos_single)
        self.target_pos = self.target_pos_single.expand(self.num_envs, self.drone.n, 3) + self.formation
        self.drone_id = torch.Tensor(np.arange(self.drone.n)).to(self.device)
        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = 3 + 3 + 4 + 3 + 3 # position, velocity, quaternion, heading, up
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
                    "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)),
                    "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, 10+1)),
                }).expand(self.drone.n), 
                "state": CompositeSpec({
                    "drones": UnboundedContinuousTensorSpec((self.drone.n, state_dim)),
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
            "terminated": UnboundedContinuousTensorSpec(1),
            "crash": UnboundedContinuousTensorSpec(1),
            "too close": UnboundedContinuousTensorSpec(1),
            "done": UnboundedContinuousTensorSpec(1),
            "height_penalty": UnboundedContinuousTensorSpec(1),
            "return": UnboundedContinuousTensorSpec(1),
            "episode_len": UnboundedContinuousTensorSpec(1),
            "drone_return": UnboundedContinuousTensorSpec(1),
            "formation_dist": UnboundedContinuousTensorSpec(1),
            "crash_return": UnboundedContinuousTensorSpec(1),
            "too_close_return": UnboundedContinuousTensorSpec(1),
            "terminated_return": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        info_spec = CompositeSpec({
            "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

        self.latency = 2 if self.cfg.task.latency else 0
        self.obs_buffer = collections.deque(maxlen=self.latency)

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)
        self.t_formed_indicator[env_ids] = False
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

        target_height = torch.tensor(self.cfg.task.target_height, device = self.device)
        self.stats[env_ids] = 0.
        self.t_formed[env_ids]=torch.nan
        self.height_penalty[env_ids] = 0.
        self.formation_dist_sum[env_ids] = 0.

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")]
        self.effort = self.drone.apply_action(actions)
  
    def _compute_state_and_obs(self):
        self.root_states = self.drone.get_state()  # Include pos, rot, vel, ...
        self.info["drone_state"][:] = self.root_states[..., :13]
        pos = self.drone.pos  # Position of all the drones relative to the environment's local coordinate

        obs_self = [self.root_states[..., :10], self.root_states[..., 13:19]]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs_self.append(t.expand(-1, self.drone.n, self.time_encoding_dim))
        if self.cfg.algo.share_actor:
            obs_self.append(self.drone_id.reshape(1, -1, 1).expand(self.root_states.shape[0], -1, self.id_dim))
        obs_self = torch.cat(obs_self, dim=-1)

        relative_pos = vmap(cpos)(pos, pos)
        self.drone_pdist = vmap(off_diag)(torch.norm(relative_pos, dim=-1, keepdim=True))   # pair wise distance
        relative_pos = vmap(off_diag)(relative_pos)

        obs_others = torch.cat([
            relative_pos,
            self.drone_pdist,
            vmap(others)(self.root_states[..., 3:10])
        ], dim=-1)

        obs = TensorDict({ 
            "obs_self": obs_self.unsqueeze(2),  # [N, K, 1, obs_self_dim]
            "obs_others": obs_others, # [N, K, K-1, obs_others_dim]
        }, [self.num_envs, self.drone.n]) # [N, K, n_i, m_i]

        state = TensorDict({
            "drones": self.root_states,
            }, self.batch_size)
        terminated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        
        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values
        crash = (pos[..., 2] < 0.2).any(-1, keepdim=True)
        too_close = separation<self.safe_distance
        done = terminated | crash | too_close
        self.stats["terminated"][:] = (terminated.float())
        self.stats["crash"][:] = (crash.float())
        self.stats["too close"][:] = (too_close.float())
        self.stats["done"][:] = (done.float())

        if self.latency:
            self.obs_buffer.append(obs)
            obs = self.obs_buffer[0]

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

        # change to velocity reward
        vel_diff = self.root_states[..., 7:10] - self.target_vel #[env_num, drone_num, 3]
        indi_v_reward = 1 / (1 + torch.norm(vel_diff, p = 2, dim=-1)) # [env_num, drone_num]

        self.cost_h = vmap(cost_formation_hausdorff)(pos, desired_p=self.formation)
        reward_formation_base =  1 / (1 + torch.square(self.cost_h * 1.6))
        reward_formation = reward_formation_base
        
        # cost if height drop or too high
        height = pos[..., 2]   # [num_envs, drone.n]
        height_penalty = ((height < 0.8) | (height > 1.2))
        self.height_penalty[height_penalty] = -1
        height_reward = self.height_penalty

        reward_effort = (self.reward_effort_weight * torch.exp(-self.effort))
        
        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values

        terminated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        terminated_reward = self.cfg.task.terminated_reward * terminated
        
        crash = (pos[..., 2] < 0.2)
        crash_reward = self.cfg.task.crash_penalty * crash # crash_penalty < 0

        too_close = separation<self.safe_distance
        too_close_reward = self.cfg.task.too_close_penalty * too_close
        done = terminated | crash.any(-1, keepdim=True) | too_close

        reward = (
            + reward_formation
            + too_close_reward
            + terminated_reward
        ).unsqueeze(1).expand(-1, self.drone.n, 1) 
        + (indi_v_reward
            + height_reward
            + reward_effort
            + crash_reward).unsqueeze(-1)

        formation_dis = compute_formation_dis(pos, self.formation) #[env_num, 1]

        self.last_cost_h[:] = self.cost_h

        formed_indicator = (self.t_formed_indicator == False) & (self.progress_buf >= 100)
        self.t_formed[formed_indicator] = self.progress_buf[formed_indicator]
        self.t_formed_indicator[formed_indicator] = True        
        
        self.frame_counter += (torch.sum((done.squeeze()).float() * self.progress_buf)).item()

        self.stats["height_penalty"].add_(torch.mean(height_reward, dim=-1, keepdim=True))
        self.stats["return"].add_(torch.mean(reward, dim=1))
        self.stats["episode_len"][:] = self.progress_buf.unsqueeze(1)
        self.stats["drone_return"].add_(torch.mean(indi_v_reward.unsqueeze(-1)))
        self.formation_dist_sum += torch.mean(formation_dis)
        self.stats["formation_dist"][:] = self.formation_dist_sum / self.stats["episode_len"]
        self.stats["crash_return"].add_(torch.mean(crash_reward, dim=-1, keepdim=True))
        self.stats["too_close_return"].add_(too_close_reward)
        self.stats["terminated_return"].add_(terminated_reward)
        
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