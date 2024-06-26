import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
rosbags = [
    '/home/jiayu/OmniDrones/simopt/real_data/goto0_5.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/goto0_8.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/goto1_0.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/goto1_2.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/goto1_5.csv',
]
gt_time = 0.01 # 100Hz
threshold = 0.005
for idx in range(len(rosbags)):
    df = pd.read_csv(rosbags[idx], skip_blank_lines=True)
    pos_time = df['pos.time']
    vel_time = df['vel.time']
    quat_time = df['quat.time']
    omega_time = df['omega.time']
    action_time = df['motor.time']
    bat_time = df['bat.time']
    pos_time = (pos_time - pos_time[0]).to_numpy() / 1e9
    vel_time = (vel_time - vel_time[0]).to_numpy() / 1e9
    quat_time = (quat_time - quat_time[0]).to_numpy() / 1e9
    omega_time = (omega_time - omega_time[0]).to_numpy() / 1e9
    action_time = (action_time - action_time[0]).to_numpy() / 1e9
    bat_time = (bat_time - bat_time[0]).to_numpy() / 1e9
    diff_pos_time = np.abs(np.diff(pos_time) - gt_time)
    diff_vel_time = np.abs(np.diff(vel_time) - gt_time)
    diff_quat_time = np.abs(np.diff(quat_time) - gt_time)
    diff_omega_time = np.abs(np.diff(omega_time) - gt_time)
    diff_action_time = np.abs(np.diff(action_time) - gt_time)
    diff_bat_time = np.abs(np.diff(bat_time) - gt_time)
    print('Traj_{}_illegal_pos'.format(idx), np.sum(diff_pos_time > threshold))
    print('Traj_{}_illegal_vel'.format(idx), np.sum(diff_vel_time > threshold))
    print('Traj_{}_illegal_quat'.format(idx), np.sum(diff_quat_time > threshold))
    print('Traj_{}_illegal_omega'.format(idx), np.sum(diff_omega_time > threshold))
    print('Traj_{}_illegal_action'.format(idx), np.sum(diff_action_time > threshold))
    print('Traj_{}_illegal_bat'.format(idx), np.sum(diff_bat_time > threshold))
