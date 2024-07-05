import os

from typing import Dict, Optional
import torch
from functorch import vmap
import torch.optim as optim
from scipy import optimize
import time

import hydra
from omegaconf import OmegaConf
from omni_drones import init_simulation_app
from tensordict import TensorDict
import pandas as pd
import pdb
import numpy as np
import yaml
from skopt import Optimizer
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse
import matplotlib.pyplot as plt
import numpy as np

def sanity_check_of_data(df: pd.DataFrame):
    r"""Check if CSV values are in the right column.

    Errors could happen if some values are not communicated from drone to
    host PC.
    """
    LOG_FREQ = 100  # data logging frequency on CrazyFlie

    # take every 1/LOG_FREQ value from time such that we get:
    # [1-1, 2-2, 3-3, ..] as time entries from data frame
    # ts = df['time'].values[::LOG_FREQ]
    ts = df['pos.time'].values[::1]

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

rosbags = [
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_large_data1.csv',
    # '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_large_data2.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_large_data3.csv',
    # '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_large_data4.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_large_data5.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_small_data1.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_small_data2.csv',
    # '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_small_data3.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_small_data4.csv',
    # '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_small_data5.csv',
]

for idx in range(len(rosbags)):
    df = pd.read_csv(rosbags[idx], skip_blank_lines=True)
    print('current idx', idx)
    sanity_check_of_data(df)