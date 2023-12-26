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
import numpy as np
import matplotlib.pyplot as plt
import copy

rosbags = [
    '/home/jiayu/OmniDrones/realdata/crazyflie/8_100hz_cjy.csv',
    # '/home/cf/ros2_ws/rosbags/takeoff.csv',
    # '/home/cf/ros2_ws/rosbags/square.csv',
    # '/home/cf/ros2_ws/rosbags/rl.csv',
]    
def main():
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
    episode_length = preprocess_df.shape[0]
    real_data = []
    # T = 20
    # skip = 5
    T = 1
    skip = 1
    for i in range(0, episode_length-T, skip):
        _slice = slice(i, i+T)
        real_data.append(preprocess_df[_slice])
    real_data = np.array(real_data)
    
    '''
    df:
    Index(['pos.time', 'pos.x', 'pos.y', 'pos.z', (1:4)
        'quat.time', 'quat.w', 'quat.x','quat.y', 'quat.z', (5:9)
        'vel.time', 'vel.x', 'vel.y', 'vel.z', (10:13)
        'omega.time','omega.r', 'omega.p', 'omega.y', (14:17)
        'real_rate.time', 'real_rate.r', 'real_rate.p', 'real_rate.y', 
        'real_rate.thrust', (18:22)
        'target_rate.time','target_rate.r', 'target_rate.p', 'target_rate.y', 
        'target_rate.thrust',(23:27)
        'motor.time', 'motor.m1', 'motor.m2', 'motor.m3', 'motor.m4'],(28:32)
        dtype='object')
    '''
    gyro_deg = torch.tensor(real_data[:, 0, 14:17])
    gyro_rad = gyro_deg * torch.pi / 180
    real_rate_rad = torch.tensor(real_data[:, 0, 18:21]) # body-axis
    target_rate_deg = torch.tensor(real_data[:, 0, 23:26]) # body-axis
    target_rate_rad = - target_rate_rad * torch.pi / 180 
    # target_rate_rad = target_rate_deg * torch.pi / 180  # body-axis
    quat = torch.tensor(real_data[:, 0, 5:9])
    current_rate1 = quat_rotate_inverse(quat, gyro_rad)
    current_rate2 = quat_rotate(quat, gyro_rad)

    x = np.arange(0, target_rate_deg.shape[0])
    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    # ax.scatter(x[1:], gyro_rad[1:, 0], s=5, label='gyro0')
    # ax.scatter(x[1:], real_rate_rad[1:, 0], s=5, label='real0')
    # ax.scatter(x, target_rate_rad[:, 0], s=5, label='target0')
    # ax.scatter(x[1:], -gyro_rad[1:, 1], s=5, label='-gyro1')
    # ax.scatter(x[1:], real_rate_rad[1:, 1], s=5, label='real1')
    # ax.scatter(x, target_rate_rad[:, 1], s=5, label='target1')
    # ax.scatter(x[1:], gyro_rad[1:, 2], s=5, label='gyro2')
    ax.scatter(x[1:], real_rate_rad[1:, 2], s=5, label='real2')
    ax.scatter(x, target_rate_rad[:, 2], s=5, label='target2')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    # plt.savefig('gyro_real_rate1')
    plt.savefig('real_target_rate2')

if __name__ == "__main__":
    main()