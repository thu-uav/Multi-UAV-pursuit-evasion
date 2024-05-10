import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

real_br = np.load('/home/jiayu/OmniDrones/scripts/real.npy')
target_br = np.load('/home/jiayu/OmniDrones/scripts/target.npy')
target_thrust = np.load('/home/jiayu/OmniDrones/scripts/thrust.npy')
timestamps = np.arange(0, len(real_br)) * 0.01

real_rate_r = real_br[:,0]
real_rate_p = real_br[:,1]
real_rate_y = real_br[:,2]

target_rate_r = target_br[:,0]
target_rate_p = target_br[:,1]
target_rate_y = target_br[:,2]
target_thrust = target_thrust[:, 0]

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

axs[3].plot(timestamps, target_thrust, label='target_thrust', color='red')
axs[3].legend()
axs[3].grid(True)

plt.savefig('sim_controller_ctbr_clipbr04.png')
