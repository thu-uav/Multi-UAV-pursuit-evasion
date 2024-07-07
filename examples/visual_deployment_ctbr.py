import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/home/jiayu/OmniDrones/examples/real_data/trad_figure8.csv')

start_idx = 650
clip_idx = -1
timestamps = (np.array(df['pos.time']) - np.array(df['pos.time'])[0])[start_idx:clip_idx] / 10**9

real_rate_r = np.array(df['real_rate.r'])[start_idx:clip_idx] * 180.0 / np.pi
real_rate_p = np.array(df['real_rate.p'])[start_idx:clip_idx] * 180.0 / np.pi
real_rate_y = np.array(df['real_rate.y'])[start_idx:clip_idx] * 180.0 / np.pi
real_thrust = np.array(df['real_rate.thrust'])[start_idx:clip_idx] / 2**16

target_rate_r = np.array(df['target_rate.r'])[start_idx:clip_idx]
target_rate_p = np.array(df['target_rate.p'])[start_idx:clip_idx]
target_rate_y = np.array(df['target_rate.y'])[start_idx:clip_idx]
target_thrust = np.array(df['target_rate.thrust'])[start_idx:clip_idx] / 2**16

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(timestamps, real_rate_r, label='real_roll_rate', color='blue')
axs[0].plot(timestamps, target_rate_r, label='target_roll_rate', color='red')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(timestamps, real_rate_p, label='real_pitch_rate', color='blue')
axs[1].plot(timestamps, target_rate_p, label='target_pitch_rate', color='red')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(timestamps, real_rate_y, label='real_yaw_rate', color='blue')
axs[2].plot(timestamps, target_rate_y, label='target_yaw_rate', color='red')
axs[2].legend()
axs[2].grid(True)

axs[3].plot(timestamps, real_thrust, label='real_thrust', color='blue')
axs[3].plot(timestamps, target_thrust, label='target_thrust', color='red')
axs[3].legend()
axs[3].grid(True)

plt.savefig('real_trad_figure8.png')
