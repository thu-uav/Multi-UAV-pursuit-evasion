r"""Core functionalities for simulation optimization with CrazyFlie drone.
"""
import abc
import time
# import gymnasium as gym
import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional

class DataBufferBase(object):
    def __init__(self, path: str, mini_trajectory_size: int):
        self.path = path
        self.mini_trajectory_size = mini_trajectory_size

    @abc.abstractmethod
    def load_from_disk(self):
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        r"""Delete data from buffer."""
        raise NotImplementedError


class RealWorldDataBuffer(DataBufferBase):
    def __init__(
            self,
            path: str,
            mini_trajectory_size: int = 35
    ):
        super().__init__(path, mini_trajectory_size)
        self.pre_steps = 5   # steps taken for calculating internal motor state
        self.observations = []
        self.actions = []
        self.pre_inputs = []
        self.load_from_disk()

    def create_trajectory_slices(
            self,
            obs,
            PWMs,
            skip=10,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Big trajectories are split into smaller 'mini'-trajectories.

        Some drone flights may take up to several minutes. This method slices
        a trajectory of size M into (M-pre_steps) mini-trajectories.
        """
        M = obs.shape[0]
        obs_slices = []
        acs_slices = []
        pre_in_slices = []

        # acs = PWMs / 30000.0 - 1
        acs = PWMs / 2**16 * 2.0 - 1.0
        T = self.mini_trajectory_size
        assert M > (T + self.pre_steps)
        # only proceed when actual trajectory has greater size than
        # the desired mini-trajectories

        # i typically iterates over: 5, 15, 25, 35, ...
        for i in range(self.pre_steps, M-T, skip):
            _slice = slice(i, i+T)
            obs_slices.append(obs[_slice])
            acs_slices.append(acs[_slice])
            slc = slice(i-self.pre_steps, i)
            pre_in_slices.append(acs[slc])

        return np.array(obs_slices), np.array(acs_slices), np.array(pre_in_slices)

    @classmethod
    def exclude_battery_compensation(cls, PWMs, voltages):
        r"""Make PWM motor signals as if battery is 100% charged."""
        percentage = PWMs / 65535
        volts = percentage * voltages

        a = -0.0006239
        b = 0.088
        c = -volts
        c_min = b**2/(4*a)
        D = np.clip(b ** 2 - 4 * a * c, c_min, np.inf)
        thrust = (-b + np.sqrt(D)) / (2 * a)
        PWMs_cleaned = np.clip(thrust / 60, 0, 1) * 65535
        return PWMs_cleaned

    @classmethod
    def extract_from_data_frame(cls, df):
        obs = df[[
            'pos.x', 'pos.y', 'pos.z',
            'vel.x', 'vel.y', 'vel.z',
            'quat.w', 'quat.x', 'quat.y', 'quat.z',
            'omega.r', 'omega.p', 'omega.y'
        ]].to_numpy(dtype=np.float64)
        PWMs = df[['motor.m1', 'motor.m2', 'motor.m3', 'motor.m4']].to_numpy(dtype=np.float64)
        return obs, PWMs
        # voltages = df[['bat']].to_numpy(dtype=np.float64)  # in [V]
        # return obs, PWMs, voltages

    @classmethod
    def sanity_check_of_data(cls, df: pd.DataFrame):
        r"""Check if CSV values are in the right column.

        Errors could happen if some values are not communicated from drone to
        host PC.
        """
        LOG_FREQ = 100  # data logging frequency on CrazyFlie

        # take every 1/LOG_FREQ value from time such that we get:
        # [1-1, 2-2, 3-3, ..] as time entries from data frame
        ts = df['time'].values[::LOG_FREQ]

        ts_diff = (ts[1:] - ts[:-1] - 1) / 1e9
        if np.all(np.abs(ts_diff) < 0.005):
            print('Time data within tolerance < 5 ms')

        elif np.all(np.abs(ts_diff) < 0.050):
            print(f'Time data within tolerance < 50 ms. '
                         f'Max={np.max(np.abs(ts_diff))*1000:0.0f}ms')
        else:
            print(f'Time data within tolerance > 50 ms. '
                          f'Max={np.max(np.abs(ts_diff))* 1000:0.0f}ms')
            raise ValueError

    def preprocess(self, df: pd.DataFrame):
        # use data with PWM > 0
        return df[(df[['motor.m1']].to_numpy()[:,0] > 0)]
    
    def load_from_disk(self):
        # walk recursively through path
        i = 0
        observations = []
        actions = []
        pre_inputs = []
        for dir_path, dirs, files in os.walk(self.path):
            for file_name in files:
                if file_name.endswith(".csv"):
                    i += 1
                    abs_path = os.path.abspath(
                        os.path.join(dir_path, file_name))
                    df = pd.read_csv(abs_path)

                    # print(f'Check data in {file_name}')
                    # self.sanity_check_of_data(df)
                    df = self.preprocess(df)
                    # obs, PWMs, voltages = self.extract_from_data_frame(df)
                    obs, PWMs = self.extract_from_data_frame(df)
                    # PWMs_compensated = RealWorldDataBuffer.exclude_battery_compensation(
                    #     PWMs, voltages)

                    M = obs.shape[0]
                    T = self.mini_trajectory_size
                    if M > (T + self.pre_steps):
                        # only proceed when actual trajectory has greater size
                        # than the desired mini-trajectories
                        # obs, acs, pre_ins = self.create_trajectory_slices(
                        #     obs, PWMs_compensated)
                        obs, acs, pre_ins = self.create_trajectory_slices(
                            obs, PWMs)
                        observations.append(obs)
                        actions.append(acs)
                        pre_inputs.append(pre_ins)
        
        assert i > 0, f'Did not find any CSV files at: {self.path}'

        self.actions = np.concatenate(actions, axis=0)
        self.observations = np.concatenate(observations)
        self.pre_inputs = np.concatenate(pre_inputs)

        assert self.actions.shape[:2] == self.observations.shape[:2]
        print(f'Loaded {i} CSV files from: {self.path}')
        print(f'mini-batches of obs: {self.observations.shape}')

    def reset(self):
        r"""Delete data from buffer."""
        self.observations = []
        self.actions = []
        self.pre_inputs = []


# class ObjectiveFunctionBase(abc.ABC):

#     def __init__(
#             self,
#             files_path: str,
#             seed=None
#     ):
#         self.files_path = files_path
#         if seed is None:
#             self.seed = int(time.time()) % 2 ** 16
#         else:
#             self.seed = seed
#         # super(ObjectiveFunctionDrone).__init__()
#         # === Set up simulation
#         self.sim_env = self._load_simulation()
#         # === Set up real world data
#         self.real_data = self._load_real_world_data()
#         self.parameter_space = None  # set by child class

#     @abc.abstractmethod
#     def _load_real_world_data(self) -> DataBufferBase:
#         r"""Loads logged real-world data from hardware disk."""
#         raise NotImplementedError

#     @abc.abstractmethod
#     def _load_simulation(self) -> gym.Env:
#         r"""Creates an instance of the simulation environment.."""
#         raise NotImplementedError

#     def check_parameters(self, params) -> bool:
#         r"""Check dimensionality of parameters."""
#         check = (params.shape == self.sample().shape)
#         return check

#     @abc.abstractmethod
#     def evaluate(self, params) -> float:
#         r"""Evaluate the performance (fitness) of the suggested parameters. """
#         raise NotImplementedError

#     @abc.abstractmethod
#     def get_parameters(self) -> np.ndarray:
#         r"""Returns the current parameters of the simulation."""
#         raise NotImplementedError

#     @abc.abstractmethod
#     def sample(self) -> np.ndarray:
#         r"""Samples a random parameter vector from a valid parameter range."""
#         raise NotImplementedError

#     @abc.abstractmethod
#     def set_parameters(self, params):
#         r"""Set parameters to simulation env."""
#         raise NotImplementedError