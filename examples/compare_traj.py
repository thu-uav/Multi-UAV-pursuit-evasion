import logging
import os
import time

import hydra
import torch
import numpy as np
from functorch import vmap
from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict, TensorDictBase
from typing import Sequence
import pandas as pd
rl_rosbags = [
    '/home/jiayu/OmniDrones/realdata/crazyflie/hover_rl_worandom_woopt.csv',
    # '/home/cf/ros2_ws/rosbags/takeoff.csv',
    # '/home/cf/ros2_ws/rosbags/square.csv',
    # '/home/cf/ros2_ws/rosbags/rl.csv',
]
gt_rosbags = [
    '/home/jiayu/OmniDrones/realdata/crazyflie/hover_rl_worandom_woopt.csv',
    # '/home/cf/ros2_ws/rosbags/takeoff.csv',
    # '/home/cf/ros2_ws/rosbags/square.csv',
    # '/home/cf/ros2_ws/rosbags/rl.csv',
]

def main():
    # plot trajectory
    import matplotlib.pyplot as plt
    import numpy as np

    sim_poses = []
    real_poses = []
    average_dt = 0.01
    df = pd.read_csv(rosbags[0])
    episode_length = df.index.stop

    for i in range(episode_length):
        # real drone state
        gt_state = df.loc[i]
        rl_state = 

    real_poses = torch.stack(real_poses).detach().cpu().numpy()
    sim_poses = torch.stack(sim_poses).detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(sim_poses[:, 0], sim_poses[:, 1], sim_poses[:, 2], s=5, label='sim')
    ax.scatter(real_poses[:, 0], real_poses[:, 1], real_poses[:, 2], s=5, label='real')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.savefig('cf_goodrl_rl')

if __name__ == "__main__":
    main()
