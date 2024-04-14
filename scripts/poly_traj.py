import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# 定义多项式拟合函数
def polynomial_func(t, a, b, c, d, e, f, g, h):
    return a * t ** 7 + b * t ** 6 + c * t ** 5 + d * t ** 4 + e * t ** 3 + f * t ** 2 + g * t + h

# 按照持续时间划分轨迹为多个小段
def split_trajectory(duration, x, y, z, segment_duration):
    num_segments = int(np.ceil(duration[-1] / segment_duration))
    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, duration[-1])
        indices = np.where((duration >= start_time) & (duration <= end_time))[0]
        segments.append((duration[indices], x[indices], y[indices], z[indices]))
    return segments

# 对每个小段进行多项式拟合
def fit_segments(segments, segment_duration, idx=0):
    fitted_segments = []
    data_list = [['duration','x^0','x^1','x^2','x^3','x^4','x^5',
                  'x^6','x^7','y^0','y^1','y^2','y^3','y^4','y^5',
                  'y^6','y^7','z^0','z^1','z^2','z^3','z^4','z^5',
                  'z^6','z^7','yaw^0','yaw^1','yaw^2','yaw^3','yaw^4',
                  'yaw^5','yaw^6','yaw^7',]]
    for segment in segments:
        duration, x, y, z = segment
        popt_x, _ = curve_fit(polynomial_func, duration, x)
        popt_y, _ = curve_fit(polynomial_func, duration, y)
        popt_z, _ = curve_fit(polynomial_func, duration, z)
        one_line = []
        one_line += [np.array([segment_duration])]
        one_line += [popt_x]
        one_line += [popt_y]
        one_line += [popt_z]
        one_line += [np.zeros(shape=8)]
        one_line = np.concatenate(one_line)
        data_list.append(one_line)
        fitted_segments.append((duration, polynomial_func(duration, *popt_x), polynomial_func(duration, *popt_y), polynomial_func(duration, *popt_z)))
    df = pd.DataFrame(data_list)
    df.to_csv('drone{}.csv'.format(idx), index=False)
    return fitted_segments

# 绘制原始轨迹和拟合曲线
def plot_trajectory_with_fitted_segments(duration, x, y, z, fitted_segments, idx=0):
    plt.figure(figsize=(10, 6))
    # plt.plot(duration, x, 'r.', label='Original x data')
    # plt.plot(duration, y, 'b.', label='Original y data')
    # plt.plot(duration, z, 'g.', label='Original z data')
    for segment in fitted_segments:
        plt.plot(segment[0], segment[1], 'r-', alpha=0.5)
        plt.plot(segment[0], segment[2], 'b-', alpha=0.5)
        plt.plot(segment[0], segment[3], 'g-', alpha=0.5)
    plt.legend()
    plt.xlabel('Duration')
    plt.ylabel('Position')
    plt.title('Polynomial Fitting for Trajectory Data')
    plt.grid(True)
    plt.savefig('traj_{}.png'.format(idx))

# 绘制原始轨迹和拟合曲线
def plot_trajectory_3d(duration, x, y, z, fitted_segments, idx=0):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z, c='r', marker='.', label='Original data')
    for segment in fitted_segments:
        ax.plot(segment[1], segment[2], segment[3], 'b-', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Polynomial Fitting for Trajectory Data')
    plt.legend()
    plt.grid(True)
    plt.savefig('traj_3d_{}.png'.format(idx))

# 示例轨迹数据
t = np.linspace(0, 800 * 0.016, 800)
# x = 2 * t + 3 * t ** 2 + np.random.normal(0, 1, 100)  # 示例x轨迹
# y = t + 4 * t ** 2 + np.random.normal(0, 1, 100)      # 示例y轨迹
# z = 3 * t + 2 * t ** 2 + np.random.normal(0, 1, 100)  # 示例z轨迹
data = np.load('/home/chenjy/OmniDrones/scripts/predatorprey.npy')
x0 = data[:, 0, 0, 0]
y0 = data[:, 0, 0, 1]
z0 = data[:, 0, 0, 2]
x1 = data[:, 1, 0, 0]
y1 = data[:, 1, 0, 1]
z1 = data[:, 1, 0, 2]
x2 = data[:, 2, 0, 0]
y2 = data[:, 2, 0, 1]
z2 = data[:, 2, 0, 2]
x3 = data[:, 3, 0, 0]
y3 = data[:, 3, 0, 1]
z3 = data[:, 3, 0, 2]

# 按照持续时间划分轨迹为多个小段
segment_duration = 1.0  # 每个小段的持续时间
segments = split_trajectory(t, x0, y0, z0, segment_duration)

# 对每个小段进行多项式拟合
fitted_segments = fit_segments(segments, segment_duration, idx=0)

# 绘制原始轨迹和拟合曲线
plot_trajectory_with_fitted_segments(t, x0, y0, z0, fitted_segments, idx=0)
plot_trajectory_3d(t, x0, y0, z0, fitted_segments, idx=0)

# agent1
# 按照持续时间划分轨迹为多个小段
segments = split_trajectory(t, x1, y1, z1, segment_duration)

# 对每个小段进行多项式拟合
fitted_segments = fit_segments(segments, segment_duration, idx=1)

# 绘制原始轨迹和拟合曲线
plot_trajectory_with_fitted_segments(t, x1, y1, z1, fitted_segments, idx=1)
plot_trajectory_3d(t, x1, y1, z1, fitted_segments, idx=1)

# agent2
# 按照持续时间划分轨迹为多个小段
segments = split_trajectory(t, x2, y2, z2, segment_duration)

# 对每个小段进行多项式拟合
fitted_segments = fit_segments(segments, segment_duration, idx=2)

# 绘制原始轨迹和拟合曲线
plot_trajectory_with_fitted_segments(t, x2, y2, z2, fitted_segments, idx=2)
plot_trajectory_3d(t, x2, y2, z2, fitted_segments, idx=2)

# agent3
# 按照持续时间划分轨迹为多个小段
segments = split_trajectory(t, x3, y3, z3, segment_duration)

# 对每个小段进行多项式拟合
fitted_segments = fit_segments(segments, segment_duration, idx=3)

# 绘制原始轨迹和拟合曲线
plot_trajectory_with_fitted_segments(t, x3, y3, z3, fitted_segments, idx=3)
plot_trajectory_3d(t, x3, y3, z3, fitted_segments, idx=3)