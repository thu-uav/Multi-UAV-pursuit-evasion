import abc

from typing import Dict, List, Optional, Tuple, Type, Union, Callable

import torch
import logging
# import carb
import numpy as np
import yaml

from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, TensorSpec, DiscreteTensorSpec, BoundedTensorSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from functorch import vmap
from omni_drones.utils.torch import quaternion_to_euler
from omni_drones.utils.torchrl import AgentSpec
import os.path as osp

import sys
sys.path.append('..')
import time

def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)

class FakeRobot():
    def __init__(self, cfg, name, device):
        self.name = name
        self.device = device
        self.cfg = cfg
        if name == "Hummingbird":
            self.num_rotors = 4
        elif name == "crazyflie":
            self.num_rotors = 4
        elif name == "Firefly":
            self.num_rotors = 6

        self.action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)
        self.intrinsics_spec = CompositeSpec({
            "mass": UnboundedContinuousTensorSpec(1),
            "inertia": UnboundedContinuousTensorSpec(3),
            "KF": UnboundedContinuousTensorSpec(self.num_rotors),
            "KM": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_up": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_down": UnboundedContinuousTensorSpec(self.num_rotors),
            "drag_coef": UnboundedContinuousTensorSpec(1),
            "rotor_offset": UnboundedContinuousTensorSpec(1),
        }).to(self.device)

        if self.cfg.force_sensor:
            self.use_force_sensor = True
            state_dim = 19 + self.num_rotors + 3
        else:
            self.use_force_sensor = False
            state_dim = 19 + self.num_rotors
        self.state_spec = UnboundedContinuousTensorSpec(state_dim, device=self.device)

        self.n = 1

class FakeEnv(EnvBase):
    REGISTRY: Dict[str, Type["FakeEnv"]] = {}

    def __init__(self, cfg, headless):
        super().__init__(
            device=cfg.sim.device, batch_size=[cfg.env.num_envs], run_type_checks=False
        )
        # store inputs to class
        self.cfg = cfg
        # extract commonly used parameters
        self.num_envs = self.cfg.env.num_envs
        self.max_episode_length = self.cfg.env.max_episode_length
        self.min_episode_length = self.cfg.env.min_episode_length

        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        self._set_specs()
        self.batch_size = [1]

        self.num_cf = 1
        self.drone_state = torch.zeros((self.num_cf, 16))
        self.drone_state[0][3] = 1.
        self.drone_state[0][2] = 1.

    @property
    def agent_spec(self):
        if not hasattr(self, "_agent_spec"):
            self._agent_spec = {}
        return _AgentSpecView(self)
    
    @agent_spec.setter
    def agent_spec(self, value):
        raise AttributeError(
            "Do not set agent_spec directly."
            "Use `self.agent_spec[agent_name] = AgentSpec(...)` instead."
        )

    def close(self):
        return
        
    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        tensordict = TensorDict({}, self.batch_size, device=self.device)
        tensordict.update(self._compute_state_and_obs())
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = TensorDict({"next": {}}, self.batch_size)
        obs = self._compute_state_and_obs()
        reward = self._compute_reward_and_done()
        tensordict["next"].update(obs)
        tensordict["next"].update(reward)
        tensordict.update(obs)
        tensordict.update(reward)
        return tensordict

    @abc.abstractmethod
    def _compute_state_and_obs(self) -> TensorDictBase:
        raise NotImplementedError

    @abc.abstractmethod
    def _compute_reward_and_done(self) -> TensorDictBase:
        raise NotImplementedError

    def _set_seed(self, seed: Optional[int] = -1):
        torch.manual_seed(seed)

    def to(self, device) -> EnvBase:
        if torch.device(device) != self.device:
            raise RuntimeError(
                f"Cannot move IsaacEnv on {self.device} to a different device {device} once it's initialized."
            )
        return self

    def update_drone_state(self):
        rot = self.drone_state[..., 3:7]
        self.drone_state[..., 10:13] = vmap(quat_axis)(rot.unsqueeze(0), axis=0).squeeze()
        self.drone_state[..., 13:16] = vmap(quat_axis)(rot.unsqueeze(0), axis=2).squeeze()

class _AgentSpecView(Dict[str, AgentSpec]):
    def __init__(self, env: FakeEnv):
        super().__init__(env._agent_spec)
        self.env = env

    def __setitem__(self, k: str, v: AgentSpec) -> None:
        v._env = self.env
        return self.env._agent_spec.__setitem__(k, v)

