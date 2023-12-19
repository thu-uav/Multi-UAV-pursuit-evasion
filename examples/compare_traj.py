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
    '/home/jiayu/OmniDrones/realdata/crazyflie/8_100hz_gt.csv',
    # '/home/cf/ros2_ws/rosbags/takeoff.csv',
    # '/home/cf/ros2_ws/rosbags/square.csv',
    # '/home/cf/ros2_ws/rosbags/rl.csv',
]
gt_rosbags = [
    '/home/jiayu/OmniDrones/realdata/crazyflie/8_100hz_gt.csv',
    # '/home/cf/ros2_ws/rosbags/takeoff.csv',
    # '/home/cf/ros2_ws/rosbags/square.csv',
    # '/home/cf/ros2_ws/rosbags/rl.csv',
]

def main():
    # plot trajectory
    import matplotlib.pyplot as plt
    import numpy as np

    rl_poses = []
    gt_poses = []
    average_dt = 0.01
    rl_df = pd.read_csv(rl_rosbags[0])
    gt_df = pd.read_csv(gt_rosbags[0])
    episode_length = rl_df.index.stop

    for i in range(episode_length):
        # real drone state
        gt_state = gt_df.loc[i]
        rl_state = rl_df.loc[i]
        gt_pos = torch.tensor([gt_state['pos.x'], gt_state['pos.y'], gt_state['pos.z']])
        rl_pos = torch.tensor([rl_state['pos.x'], rl_state['pos.y'], rl_state['pos.z']])
        rl_poses.append(rl_pos)
        gt_poses.append(gt_pos)

    gt_poses = torch.stack(gt_poses).detach().cpu().numpy()
    rl_poses = torch.stack(rl_poses).detach().cpu().numpy()

    # TODO: compute error

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(rl_poses[:, 0], rl_poses[:, 1], rl_poses[:, 2], s=5, label='rl')
    ax.scatter(gt_poses[:, 0], gt_poses[:, 1], gt_poses[:, 2], s=5, label='gt')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.savefig('cf_compare')

if __name__ == "__main__":
    main()
