import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import pdb
rosbags = [
    # '/home/chenjiayu/OmniDrones/realdata/crazyflie/8_100hz_light.csv',
    '/home/chenjiayu/OmniDrones/realdata/crazyflie/100hz_8.csv',
    # '/home/chenjiayu/OmniDrones/realdata/crazyflie/100hz_rotate.csv'
]

df = pd.read_csv(rosbags[0], skip_blank_lines=True)
df = np.array(df)
# preprocess, motor > 0
use_preprocess = False
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
T = 1
skip = 1
for i in range(0, episode_length-T, skip):
    _slice = slice(i, i+T)
    real_data.append(preprocess_df[_slice])
real_data = np.array(real_data)

# compute real_rate
real_rpy = real_data[:,:, 32:]
time = real_data[:,:, 0]
compute_rate = []
for i in range(real_rpy.shape[0]-1):
    compute_rate.append((real_rpy[i + 1] - real_rpy[i]) / (time[i + 1] - time[i]) * 1e9)
compute_rate = np.clip((np.array(compute_rate) * np.pi / 180), -5, 5)[:1500]

steps = np.arange(0, real_data.shape[0])

real_rate = real_data[..., 18:21]
real_thrust = real_data[..., 21]
target_rate = np.clip((real_data[..., 23:26] / 180 * torch.pi), -5, 5)
target_thrust = real_data[..., 26]

fig, axs = plt.subplots(2, 2, figsize=(10, 6))  # 2 * 2    
# sim
# axs.scatter(steps[:1500], real_rate[:1500, 0, 0], s=5, c='red', label='real')
# axs.scatter(steps[:1500], target_rate[:1500, 0, 0], s=5, c='green', label='target')
# axs.set_xlabel('steps')
# axs.set_ylabel('rad/s')
# axs.set_title('real/target_body_rate_x')
# axs.legend()

# compute rate
axs[0,0].scatter(steps[:compute_rate.shape[0]], compute_rate[:, 0, 0], s=5, c='red', label='real')
axs[0,0].scatter(steps[:compute_rate.shape[0]], target_rate[:compute_rate.shape[0], 0, 0], s=5, c='green', label='target')
axs[0,0].set_xlabel('steps')
axs[0,0].set_ylabel('rad/s')
axs[0,0].set_title('real/target_body_rate_x')
axs[0,0].legend()

axs[0,1].scatter(steps[:compute_rate.shape[0]], compute_rate[:, 0, 1], s=5, c='red', label='real')
axs[0,1].scatter(steps[:compute_rate.shape[0]], target_rate[:compute_rate.shape[0], 0, 1], s=5, c='green', label='target')
axs[0,1].set_xlabel('steps')
axs[0,1].set_ylabel('rad/s')
axs[0,1].set_title('real/target_body_rate_y')
axs[0,1].legend()

axs[1,0].scatter(steps[:compute_rate.shape[0]], compute_rate[:, 0, 2], s=5, c='red', label='real')
axs[1,0].scatter(steps[:compute_rate.shape[0]], target_rate[:compute_rate.shape[0], 0, 2], s=5, c='green', label='target')
axs[1,0].set_xlabel('steps')
axs[1,0].set_ylabel('rad/s')
axs[1,0].set_title('real/target_body_rate_z')
axs[1,0].legend()

axs[1,1].scatter(steps[:compute_rate.shape[0]], real_thrust[:compute_rate.shape[0], 0], s=5, c='red', label='real')
axs[1,1].scatter(steps[:compute_rate.shape[0]], target_thrust[:compute_rate.shape[0], 0], s=5, c='green', label='target')
axs[1,1].set_xlabel('steps')
axs[1,1].set_ylabel('N')
axs[1,1].set_title('real/target_thrust')
axs[1,1].legend()


# # gyro
# clip_step = 1500
# axs[0,0].scatter(steps[:clip_step], real_rate[:clip_step, 0, 0], s=5, c='red', label='real')
# axs[0,0].scatter(steps[:clip_step], target_rate[:clip_step, 0, 0], s=5, c='green', label='target')
# axs[0,0].set_xlabel('steps')
# axs[0,0].set_ylabel('rad/s')
# axs[0,0].set_title('real/target_body_rate_x')
# axs[0,0].legend()

# axs[0,1].scatter(steps[:clip_step], real_rate[:clip_step, 0, 1], s=5, c='red', label='real')
# axs[0,1].scatter(steps[:clip_step], target_rate[:clip_step, 0, 1], s=5, c='green', label='target')
# axs[0,1].set_xlabel('steps')
# axs[0,1].set_ylabel('rad/s')
# axs[0,1].set_title('real/target_body_rate_y')
# axs[0,1].legend()

# axs[1,0].scatter(steps[:clip_step], real_rate[:clip_step, 0, 2], s=5, c='red', label='real')
# axs[1,0].scatter(steps[:clip_step], target_rate[:clip_step, 0, 2], s=5, c='green', label='target')
# axs[1,0].set_xlabel('steps')
# axs[1,0].set_ylabel('rad/s')
# axs[1,0].set_title('real/target_body_rate_z')
# axs[1,0].legend()

# axs[1,1].scatter(steps[:clip_step], real_thrust[:clip_step, 0], s=5, c='red', label='real')
# axs[1,1].scatter(steps[:clip_step], target_thrust[:clip_step, 0], s=5, c='green', label='target')
# axs[1,1].set_xlabel('steps')
# axs[1,1].set_ylabel('N')
# axs[1,1].set_title('real/target_thrust')
# axs[1,1].legend()

plt.savefig('compute_real_target_8')