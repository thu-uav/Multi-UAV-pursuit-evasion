import functorch
import torch
import torch.distributions as D

from .fake_env import AgentSpec, FakeEnv
from omni_drones.utils.torch import euler_to_quaternion, quat_axis

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, DiscreteTensorSpec, BoundedTensorSpec
import collections
from tensordict.tensordict import TensorDict, TensorDictBase
from functorch import vmap

class FakeTrack(FakeEnv):
    def __init__(self, cfg, connection, swarm):
        self.alpha = 0.8
        self.num_envs = 1
        self.cfg = cfg
        self.future_traj_steps = 4
        self.dt = 0.01
        self.num_cf = 1

        super().__init__(cfg, connection, swarm)

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-.2, -.2, 0.], device=self.device) * torch.pi,
            torch.tensor([0.2, 0.2, .2], device=self.device) * torch.pi
        )

        self.a_scale_dist = D.Uniform(
            torch.tensor(3.0, device=self.device),
            torch.tensor(3.01, device=self.device)
        )
        
        self.threshold_scale_dist = D.Uniform(
            torch.tensor(0.5 * scale_time(torch.tensor(self.max_episode_length * self.dt)), device=self.device),
            torch.tensor(0.501 * scale_time(torch.tensor(self.max_episode_length * self.dt)), device=self.device)
        )
        
        self.c_scale_dist = D.Uniform(
            torch.tensor(0.9, device=self.device),
            torch.tensor(0.901, device=self.device)
        )

        self.origin = torch.tensor([0., 0., 1.], device=self.device)

        self.traj_t0 = 0.0
        # self.v_scale = torch.zeros(self.num_envs, device=self.device)
        self.a_scale = torch.zeros(self.num_envs, device=self.device)
        self.threshold_scale = torch.zeros(self.num_envs, device=self.device)
        self.c_scale = torch.zeros(self.num_envs, device=self.device)

        self.last_linear_v = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_v = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_linear_a = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_a = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_linear_jerk = torch.zeros(self.num_envs, 1, device=self.device)
        self.last_angular_jerk = torch.zeros(self.num_envs, 1, device=self.device)

        self.target_pos = torch.zeros(self.num_envs, self.future_traj_steps, 3, device=self.device)

        # reset / initialize
        env_ids = torch.tensor([0])
        self.a_scale[env_ids] = self.a_scale_dist.sample(env_ids.shape)
        self.threshold_scale[env_ids] = self.threshold_scale_dist.sample(env_ids.shape)
        self.c_scale[env_ids] = self.c_scale_dist.sample(env_ids.shape)

        self.target_poses = []

    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = 3 + 3 + 4 + 3 + 3 # position, velocity, quaternion, heading, up, relative heading
        observation_dim += 3 * (self.future_traj_steps-1)

        obs_dim += 4 # acc + jerk

        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim

        self.observation_spec = CompositeSpec({
            "agents": CompositeSpec({
                "observation": UnboundedContinuousTensorSpec((1, observation_dim), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        self.action_spec = CompositeSpec({
            "agents": CompositeSpec({
                "action":  BoundedTensorSpec(-1, 1, 4, device=self.device).unsqueeze(0),
            })
        }).expand(self.num_envs).to(self.device)
        self.reward_spec = CompositeSpec({
            "agents": CompositeSpec({
                "reward": UnboundedContinuousTensorSpec((1, 1))
            })
        }).expand(self.num_envs).to(self.device)
        self.done_spec = CompositeSpec({
            "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
            "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
        }).expand(self.num_envs).to(self.device)
        self.info_spec = CompositeSpec({
            "agents": CompositeSpec({
                "target_position": UnboundedContinuousTensorSpec((1, 3), device=self.device),
                "real_position": UnboundedContinuousTensorSpec((1, 3), device=self.device),
            })
        }).expand(self.num_envs).to(self.device)
        self.agent_spec["drone"] = AgentSpec(
            "drone", 1,
            observation_key=("agents", "observation"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "intrinsics"),
        )

    def _compute_state_and_obs(self) -> TensorDictBase:
        self.update_drone_state()
        self.target_pos[:] = self._compute_traj(self.future_traj_steps, step_size=5)
        # print(self.target_pos[:, 0])
        self.rpos = self.target_pos.cpu() - self.drone_state[..., :3]
        obs = [
            self.rpos.flatten(1).unsqueeze(1),
            self.root_state[..., 3:10], self.root_state[..., 13:19],
        ]
        t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
        obs.append(t.expand(-1, self.time_encoding_dim).unsqueeze(1))

        # linear_v, angular_v
        self.linear_v = torch.norm(self.root_state[..., 7:10], dim=-1)
        self.angular_v = torch.norm(self.root_state[..., 10:13], dim=-1)
        # linear_a, angular_a
        self.linear_a = torch.abs(self.linear_v - self.last_linear_v) / self.dt
        self.angular_a = torch.abs(self.angular_v - self.last_angular_v) / self.dt
        # linear_jerk, angular_jerk
        self.linear_jerk = torch.abs(self.linear_a - self.last_linear_a) / self.dt
        self.angular_jerk = torch.abs(self.angular_a - self.last_angular_a) / self.dt

        # set last
        self.last_linear_v = self.linear_v.clone()
        self.last_angular_v = self.angular_v.clone()
        self.last_linear_a = self.linear_a.clone()
        self.last_angular_a = self.angular_a.clone()
        self.last_linear_jerk = self.linear_jerk.clone()
        self.last_angular_jerk = self.angular_jerk.clone()
        
        # add acc and jerk
        obs.append(self.linear_a.unsqueeze(1) / 10.0)
        obs.append(self.angular_a.unsqueeze(1) / 100.0)
        obs.append(self.linear_jerk.unsqueeze(1) / 1000.0)
        obs.append(self.angular_jerk.unsqueeze(1) / 10000.0)
        
        obs = torch.cat(obs, dim=-1)

        return TensorDict({
            "agents": {
                "observation": obs,
                "target_position": self.target_pos[..., 0, :],
                "real_position": self.drone_state[..., :3]
            },
        }, self.num_envs)

    def _compute_reward_and_done(self) -> TensorDictBase:
        distance = torch.norm(self.rpos[:, [0]][:2], dim=-1)
        # reward = torch.zeros((self.num_envs, 1, 1))
        # reward[..., 0] = distance.mean()
        reward = distance
        done = torch.zeros((self.num_envs, 1, 1)).bool()
        return TensorDict(
            {
                "agents": {
                    "reward": reward,
                },
                "done": done,
                "terminated": done,
                "truncated": done
            },
            self.num_envs,
        )

    def _compute_traj(self, steps: int, env_ids=None, step_size: float=1.):
        if env_ids is None:
            env_ids = ...
        t = self.progress_buf[env_ids].unsqueeze(1) + step_size * torch.arange(steps, device=self.device)
        t = self.traj_t0 + scale_time(torch.ones((self.num_envs, 1), device=self.device)[env_ids] * t * self.dt)
        # t = self.traj_t0 + torch.ones((self.num_envs, 1), device=self.device) * t * self.dt
        # traj_rot = self.traj_rot[env_ids].unsqueeze(1).expand(-1, t.shape[1], 4)
        
        # target_pos = vmap(lemniscate)(t, self.traj_c[env_ids])
        # target_pos = vmap(line_segments)(t, self.v_scale[env_ids], self.threshold_scale[env_ids], torch.pi * self.c_scale[env_ids])
        target_pos = vmap(line_segments_acc)(t, self.a_scale[env_ids], self.threshold_scale[env_ids], torch.pi * self.c_scale[env_ids])
        # target_pos = vmap(torch_utils.quat_rotate)(traj_rot, target_pos) * self.traj_scale[env_ids].unsqueeze(1)

        return self.origin + target_pos

    def save_target_traj(self, name):
        torch.save(self.target_poses, name)

def line_segments_acc(t, a, threshold, c):
    # v = torch.tensor(v)
    # threshold = torch.tensor(threshold)
    # c = torch.tensor(c)
    v_turn = a * threshold
    x = torch.where(t <= threshold, 0.5 * a * t**2, 0.5 * a * threshold**2 + v_turn * (t - threshold) * torch.cos(c))
    y = torch.where(t <= threshold, torch.zeros_like(t), v_turn * (t - threshold) * torch.sin(c))
    z = torch.zeros_like(t)

    return torch.stack([x, y, z], dim=-1)

def pentagram(t, c):
    x = -1.5 * torch.sin(2 * t) - 0.5 * torch.sin(3 * t)
    y = 1.5 * torch.cos(2 * t) - 0.5 * torch.cos(3 * t)
    # x = -1.1 * torch.sin(2 * t) - 0.5 * torch.sin(3 * t)
    # y = 1.1 * torch.cos(2 * t) - 0.5 * torch.cos(3 * t)
    z = torch.zeros_like(t)
    return torch.stack([x,y,z], dim=-1)

def lemniscate(t, c):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    sin2p1 = torch.square(sin_t) + 1

    x = torch.stack([
        cos_t, sin_t * cos_t, c * sin_t
    ], dim=-1) / sin2p1.unsqueeze(-1)

    return x

def circle(t):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    x = torch.stack([
        cos_t, sin_t, torch.zeros_like(sin_t)
    ], dim=-1)

    return x

def square(t_s):
    x_s = []
    for t_ in t_s[0]:
        t = torch.abs(t_).item()
        while t >= 8:
            t -= 8
        if t < 2:
            x = torch.tensor([-1., 1-t, 0.])
        elif t < 4:
            x = torch.tensor([t-3, -1., 0.])
        elif t < 6:
            x = torch.tensor([1., t-5, 0.])
        elif t < 8:
            x = torch.tensor([7-t, 1., 0.])
        x_s.append(x)
    x_s = torch.stack(x_s, dim=0).unsqueeze(0).to(t_s.device)
    return x_s

def scale_time(t, a: float=1.0):
    return t / (1 + 1/(a*torch.abs(t)))

import functools
def manual_batch(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        batch_shapes = set(arg.shape[:-1] for arg in args if isinstance(arg, torch.Tensor))
        if not len(batch_shapes) == 1:
            raise ValueError
        batch_shape = batch_shapes.pop()
        args = (
            arg.reshape(-1, arg.shape[-1]) if isinstance(arg, torch.Tensor) else arg 
            for arg in args
        )
        kwargs = {
            k: v.reshape(-1, v.shape[-1]) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }
        out = func(*args, **kwargs)
        return out.unflatten(0, batch_shape)
    return wrapped

@manual_batch
def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c