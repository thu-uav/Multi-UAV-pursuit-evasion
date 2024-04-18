import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 从CSV文件中读取数据
data = pd.read_csv('/home/chenjy/OmniDrones/scripts/drone_1.csv')

# 提取持续时间和系数数据
durations = data['duration']
coeffs = data.iloc[:, 1:].to_numpy()[:,:24]

# 定义多项式函数
def polynomial_func(t, coeffs):
    return np.polyval(coeffs[::-1], t)

# 根据多项式系数生成轨迹
def generate_trajectory(duration, coeffs):
    t = np.linspace(0, duration, 100)
    x = polynomial_func(t, coeffs[:8])
    y = polynomial_func(t, coeffs[8:16])
    z = polynomial_func(t, coeffs[16:])
    # breakpoint()
    return x, y, z

# 计算速度和加速度
def compute_velocity_and_acceleration(duration, coeffs, prev_end_time=0):
    t = np.linspace(0, duration, 100)
    x_dot = np.polyder(coeffs[:8])
    y_dot = np.polyder(coeffs[8:16])
    z_dot = np.polyder(coeffs[16:])
    x_dot_values = np.polyval(x_dot[::-1], t)
    y_dot_values = np.polyval(y_dot[::-1], t)
    z_dot_values = np.polyval(z_dot[::-1], t)
    velocity = np.sqrt(x_dot_values**2 + y_dot_values**2 + z_dot_values**2)
    x_ddot = np.polyder(x_dot)
    y_ddot = np.polyder(y_dot)
    z_ddot = np.polyder(z_dot)
    x_ddot_values = np.polyval(x_ddot[::-1], t)
    y_ddot_values = np.polyval(y_ddot[::-1], t)
    z_ddot_values = np.polyval(z_ddot[::-1], t)
    acceleration = np.sqrt(x_ddot_values**2 + y_ddot_values**2 + z_ddot_values**2)
    t += prev_end_time
    return t, velocity, acceleration

# 绘制速度-时间曲线和加速度-时间曲线
def plot_velocity_and_acceleration(durations, coeffs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    prev_end_time = 0
    for duration, coeff in zip(durations, coeffs):
        t, velocity, acceleration = compute_velocity_and_acceleration(duration, coeff, prev_end_time)
        ax1.plot(t, velocity, label=f'Duration: {duration} s')
        ax2.plot(t, acceleration, label=f'Duration: {duration} s')
        prev_end_time += duration
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Velocity-Time')
    ax1.legend()
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration (m/s^2)')
    ax2.set_title('Acceleration-Time')
    ax2.legend()
    plt.savefig('velocity_acceleration.png')

# 绘制轨迹
def plot_all(durations, coeffs, name):
    fig = plt.figure(figsize=(15, 10))

    ax1 = fig.add_subplot(211, projection='3d')
    for duration, coeff in zip(durations, coeffs):
        x, y, z = generate_trajectory(duration, coeff)
        ax1.plot(x, y, z, label=f'Duration: {duration} s')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Trajectory')
    ax1.legend(loc='center left', bbox_to_anchor=(-1, 0.5))

    ax2 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224)
    prev_end_time = 0
    for duration, coeff in zip(durations, coeffs):
        t, velocity, acceleration = compute_velocity_and_acceleration(duration, coeff, prev_end_time)
        ax2.plot(t, velocity, label=f'Duration: {duration} s')
        ax3.plot(t, acceleration, label=f'Duration: {duration} s')
        prev_end_time += duration
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity-Time')
    # ax2.legend()
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s^2)')
    ax3.set_title('Acceleration-Time')
    # ax3.legend()
    plt.savefig('pos_vel_acc_{}.png'.format(name))    

# transfer polynomial csv to 3D traj, vel-time and acc-time
plot_all(durations, coeffs, name='drone1')
