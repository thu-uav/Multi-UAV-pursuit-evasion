import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
rosbags = [
    '/home/chenjiayu/OmniDrones/realdata/crazyflie/8_100hz_light.csv',
]

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
T = 1
skip = 1
for i in range(0, episode_length-T, skip):
    _slice = slice(i, i+T)
    real_data.append(preprocess_df[_slice])
real_data = np.array(real_data)

steps = np.arange(0, real_data.shape[0])

real_rate = real_data[..., 18:21]
real_thrust = real_data[..., 21]
target_rate = real_data[..., 23:26] / 180 * torch.pi
target_thrust = real_data[..., 26]

fig, axs = plt.subplots(2, 2, figsize=(10, 6))  # 2 * 2    
# sim
# axs.scatter(steps[:1500], real_rate[:1500, 0, 0], s=5, c='red', label='real')
# axs.scatter(steps[:1500], target_rate[:1500, 0, 0], s=5, c='green', label='target')
# axs.set_xlabel('steps')
# axs.set_ylabel('rad/s')
# axs.set_title('real/target_body_rate_x')
# axs.legend()
axs[0,0].scatter(steps[:1500], real_rate[:1500, 0, 0], s=5, c='red', label='real')
axs[0,0].scatter(steps[:1500], target_rate[:1500, 0, 0], s=5, c='green', label='target')
axs[0,0].set_xlabel('steps')
axs[0,0].set_ylabel('rad/s')
axs[0,0].set_title('real/target_body_rate_x')
axs[0,0].legend()

axs[0,1].scatter(steps[:1500], real_rate[:1500, 0, 1], s=5, c='red', label='real')
axs[0,1].scatter(steps[:1500], target_rate[:1500, 0, 1], s=5, c='green', label='target')
axs[0,1].set_xlabel('steps')
axs[0,1].set_ylabel('rad/s')
axs[0,1].set_title('real/target_body_rate_y')
axs[0,1].legend()

axs[1,0].scatter(steps[:1500], real_rate[:1500, 0, 2], s=5, c='red', label='real')
axs[1,0].scatter(steps[:1500], target_rate[:1500, 0, 2], s=5, c='green', label='target')
axs[1,0].set_xlabel('steps')
axs[1,0].set_ylabel('rad/s')
axs[1,0].set_title('real/target_body_rate_z')
axs[1,0].legend()

axs[1,1].scatter(steps[:1500], real_thrust[:1500, 0], s=5, c='red', label='real')
axs[1,1].scatter(steps[:1500], target_thrust[:1500, 0], s=5, c='green', label='target')
axs[1,1].set_xlabel('steps')
axs[1,1].set_ylabel('N')
axs[1,1].set_title('real/target_thrust')
axs[1,1].legend()

plt.savefig('real_target_compare')