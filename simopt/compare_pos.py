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
import quaternion
import numpy as np
import yaml
from skopt import Optimizer
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse
import matplotlib.pyplot as plt
import numpy as np

cf13_rosbags = [
    '/home/jiayu/OmniDrones/examples/real_data/cf13_hover_pos.csv',
]

cf14_rosbags = [
    '/home/jiayu/OmniDrones/examples/real_data/cf14_hover_pos.csv',
]

def preprocess(rosbags):
    df = pd.read_csv(rosbags[0], skip_blank_lines=True)
    df = np.array(df)
    # preprocess, motor > 0
    use_preprocess = True
    if use_preprocess:
        preprocess_df = []
        for df_one in df:
            if df_one[-1] > 0:
                preprocess_df.append(df_one)
        preprocess_df = np.array(preprocess_df)
    else:
        preprocess_df = df
    # episode_len = preprocess_df.shape[0] # contains land
    start_idx = 10
    episode_len = min(2000, preprocess_df.shape[0])
    real_data = []
    T = 1
    skip = 1
    for i in range(start_idx, episode_len-T, skip):
        _slice = slice(i, i+T)
        real_data.append(preprocess_df[_slice])
    return np.array(real_data)

@hydra.main(version_base=None, config_path=".", config_name="real2sim")
def main(cfg):
    """
        preprocess real data
        real_data: [batch_size, T, dimension]
    """
    cf13 = preprocess(cf13_rosbags)
    cf14 = preprocess(cf14_rosbags)

    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    fig.subplots_adjust()

    axs[0, 0].scatter((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9, cf13[:, 0, 1], s=5, c='red', label='cf13')
    axs[0, 0].scatter((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9, cf14[:, 0, 1], s=5, c='blue', label='cf14')
    axs[0, 0].set_xlabel('steps')
    axs[0, 0].set_ylabel('m')
    axs[0, 0].set_title('x')
    axs[0, 0].legend()

    axs[0, 1].scatter((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9, cf13[:, 0, 2], s=5, c='red', label='cf13')
    axs[0, 1].scatter((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9, cf14[:, 0, 2], s=5, c='blue', label='cf14')
    axs[0, 1].set_xlabel('steps')
    axs[0, 1].set_ylabel('m')
    axs[0, 1].set_title('y')
    axs[0, 1].legend()

    axs[0, 2].scatter((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9, cf13[:, 0, 3], s=5, c='red', label='cf13')
    axs[0, 2].scatter((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9, cf14[:, 0, 3], s=5, c='blue', label='cf14')
    axs[0, 2].set_xlabel('steps')
    axs[0, 2].set_ylabel('m')
    axs[0, 2].set_title('z')
    axs[0, 2].legend()

    # vel
    cf13_dt = ((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9)[1:] - ((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9)[:-1]
    cf14_dt = ((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9)[1:] - ((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9)[:-1]
    cf13_dx = cf13[1:, 0, 1] - cf13[:-1, 0, 1]
    cf14_dx = cf14[1:, 0, 1] - cf14[:-1, 0, 1]
    cf13_dy = cf13[1:, 0, 2] - cf13[:-1, 0, 2]
    cf14_dy = cf14[1:, 0, 2] - cf14[:-1, 0, 2]
    cf13_dz = cf13[1:, 0, 3] - cf13[:-1, 0, 3]
    cf14_dz = cf14[1:, 0, 3] - cf14[:-1, 0, 3]
    cf13_vx = cf13_dx / cf13_dt
    cf14_vx = cf14_dx / cf14_dt
    cf13_vy = cf13_dy / cf13_dt
    cf14_vy = cf14_dy / cf14_dt
    cf13_vz = cf13_dz / cf13_dt
    cf14_vz = cf14_dz / cf14_dt
    
    axs[1, 0].scatter(((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9)[:-1], cf13_vx, s=5, c='red', label='cf13')
    axs[1, 0].scatter(((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9)[:-1], cf14_vx, s=5, c='blue', label='cf14')
    axs[1, 0].set_xlabel('steps')
    axs[1, 0].set_ylabel('m/s')
    axs[1, 0].set_title('vel x')
    axs[1, 0].legend()

    axs[1, 1].scatter(((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9)[:-1], cf13_vy, s=5, c='red', label='cf13')
    axs[1, 1].scatter(((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9)[:-1], cf14_vy, s=5, c='blue', label='cf14')
    axs[1, 1].set_xlabel('steps')
    axs[1, 1].set_ylabel('m/s')
    axs[1, 1].set_title('vel y')
    axs[1, 1].legend()

    axs[1, 2].scatter(((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9)[:-1], cf13_vz, s=5, c='red', label='cf13')
    axs[1, 2].scatter(((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9)[:-1], cf14_vz, s=5, c='blue', label='cf14')
    axs[1, 2].set_xlabel('steps')
    axs[1, 2].set_ylabel('m/s')
    axs[1, 2].set_title('vel z')
    axs[1, 2].legend()

    # acc
    cf13_dvx = cf13_vx[1:] - cf13_vx[:-1]
    cf14_dvx = cf14_vx[1:] - cf14_vx[:-1]
    cf13_dvy = cf13_vy[1:] - cf13_vy[:-1]
    cf14_dvy = cf14_vy[1:] - cf14_vy[:-1]
    cf13_dvz = cf13_vz[1:] - cf13_vz[:-1]
    cf14_dvz = cf14_vz[1:] - cf14_vz[:-1]
    cf13_ax = cf13_dvx / cf13_dt[1:]
    cf14_ax = cf14_dvx / cf14_dt[1:]
    cf13_ay = cf13_dvy / cf13_dt[1:]
    cf14_ay = cf14_dvy / cf14_dt[1:]
    cf13_az = cf13_dvz / cf13_dt[1:]
    cf14_az = cf14_dvz / cf14_dt[1:]
    
    axs[2, 0].scatter(((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9)[:-2], cf13_ax, s=5, c='red', label='cf13')
    axs[2, 0].scatter(((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9)[:-2], cf14_ax, s=5, c='blue', label='cf14')
    axs[2, 0].set_xlabel('steps')
    axs[2, 0].set_ylabel('m/s^2')
    axs[2, 0].set_title('acc x')
    axs[2, 0].legend()

    axs[2, 1].scatter(((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9)[:-2], cf13_ay, s=5, c='red', label='cf13')
    axs[2, 1].scatter(((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9)[:-2], cf14_ay, s=5, c='blue', label='cf14')
    axs[2, 1].set_xlabel('steps')
    axs[2, 1].set_ylabel('m/s^2')
    axs[2, 1].set_title('acc y')
    axs[2, 1].legend()

    axs[2, 2].scatter(((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9)[:-2], cf13_az, s=5, c='red', label='cf13')
    axs[2, 2].scatter(((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9)[:-2], cf14_az, s=5, c='blue', label='cf14')
    axs[2, 2].set_xlabel('steps')
    axs[2, 2].set_ylabel('m/s^2')
    axs[2, 2].set_title('acc z')
    axs[2, 2].legend()

    plt.tight_layout()
    plt.savefig('cf13_vs_cf14_pos.png')


if __name__ == "__main__":
    main()