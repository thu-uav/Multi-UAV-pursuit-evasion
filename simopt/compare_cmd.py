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
    '/home/jiayu/OmniDrones/examples/real_data/cf13_hover_msg.csv',
]

cf14_rosbags = [
    '/home/jiayu/OmniDrones/examples/real_data/cf14_hover_msg.csv',
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

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    fig.subplots_adjust()

    axs[0].scatter((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9, cf13[:, 0, 1], s=5, c='red', label='cf13')
    axs[0].scatter((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9, cf14[:, 0, 1], s=5, c='blue', label='cf14')
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('degree/s')
    axs[0].set_title('roll rate')
    axs[0].legend()

    axs[1].scatter((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9, cf13[:, 0, 2], s=5, c='red', label='cf13')
    axs[1].scatter((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9, cf14[:, 0, 2], s=5, c='blue', label='cf14')
    axs[1].set_xlabel('steps')
    axs[1].set_ylabel('degree/s')
    axs[1].set_title('pitch rate')
    axs[1].legend()

    axs[2].scatter((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9, cf13[:, 0, 3], s=5, c='red', label='cf13')
    axs[2].scatter((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9, cf14[:, 0, 3], s=5, c='blue', label='cf14')
    axs[2].set_xlabel('steps')
    axs[2].set_ylabel('degree/s')
    axs[2].set_title('yaw rate')
    axs[2].legend()

    axs[3].scatter((cf13[:, 0, 0] - cf13[0, 0, 0]) / 1e9, cf13[:, 0, 4], s=5, c='red', label='cf13')
    axs[3].scatter((cf14[:, 0, 0] - cf14[0, 0, 0]) / 1e9, cf14[:, 0, 4], s=5, c='blue', label='cf14')
    axs[3].set_xlabel('steps')
    axs[3].set_ylabel('pwm')
    axs[3].set_title('thrust')
    axs[3].legend()

    plt.tight_layout()
    plt.savefig('cf13_vs_cf14_cmd.png')


if __name__ == "__main__":
    main()