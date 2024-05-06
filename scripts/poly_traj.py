import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# 定义多项式拟合函数
# def polynomial_func(t, a, b, c, d, e, f, g, h):
#     return a + b * t + c * t ** 2 + d * t ** 3 + e * t ** 4 + f * t ** 5 + g * t ** 6 + h * t ** 7
def polynomial_func(t, a, b, c, d, e, f, g, h):
    coeffs = [a, b, c, d, e, f, g, h]
    return np.polyval(coeffs[::-1], t)

# 按照持续时间划分轨迹为多个小段
def split_trajectory(duration, x, y, z, segment_duration):
    num_segments = int(np.ceil(duration[-1] / segment_duration))
    segments = []
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, duration[-1])
        indices = np.where((duration >= start_time) & (duration <= end_time))[0]
        segment_duration_list = np.linspace(0, segment_duration, indices.shape[0])
        segments.append((segment_duration_list, x[indices], y[indices], z[indices]))
    return segments

# 对每个小段进行多项式拟合
def fit_segments(segments, segment_duration, idx=0):
    fitted_segments = []
    columns = ['duration','x^0','x^1','x^2','x^3','x^4','x^5',
                  'x^6','x^7','y^0','y^1','y^2','y^3','y^4','y^5',
                  'y^6','y^7','z^0','z^1','z^2','z^3','z^4','z^5',
                  'z^6','z^7','yaw^0','yaw^1','yaw^2','yaw^3','yaw^4',
                  'yaw^5','yaw^6','yaw^7',]
    data_list = []
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
        # breakpoint()
    df = pd.DataFrame(data_list, columns=columns)
    df.to_csv('drone_{}.csv'.format(idx), index=False)
    return fitted_segments

# 绘制原始轨迹和拟合曲线
def plot_trajectory_with_fitted_segments(duration, x, y, z, fitted_segments, idx=0):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(duration, x, 'r.', label='Original x data')
    ax1.plot(duration, y, 'b.', label='Original y data')
    ax1.plot(duration, z, 'g.', label='Original z data')
    last_duration = 0 
    for segment in fitted_segments:
        ax2.plot(segment[0] + last_duration, segment[1], 'r-', alpha=0.5)
        ax2.plot(segment[0] + last_duration, segment[2], 'b-', alpha=0.5)
        ax2.plot(segment[0] + last_duration, segment[3], 'g-', alpha=0.5)
        last_duration += segment[0][-1] 
    # plt.legend()
    plt.xlabel('Duration')
    plt.ylabel('Position')
    plt.title('Polynomial Fitting for Trajectory Data')
    plt.grid(True)
    plt.savefig('traj_{}.png'.format(idx))

# 绘制原始轨迹和拟合曲线
def plot_trajectory_3d(duration, x, y, z, fitted_segments, idx=0):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    ax1.scatter(x, y, z, c='r', marker='.', label='Original data')
    for segment in fitted_segments:
        ax2.plot(segment[1], segment[2], segment[3], 'b-', alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    # plt.legend()
    plt.grid(True)
    plt.savefig('traj_3d_{}.png'.format(idx))

# transfer x-y-z to 3D traj and axis-t
t = np.linspace(0, 800 * 0.016, 800)
data = np.load('/home/chenjy/OmniDrones/scripts/outputs/Angenali.npy')
num_drones = 4
for idx in range(num_drones):
    x_origin = data[:, idx, 0, 0]
    y_origin = data[:, idx, 0, 1]
    z_origin = data[:, idx, 0, 2]

    # 按照持续时间划分轨迹为多个小段
    segment_duration = 1.0  # 每个小段的持续时间
    segments = split_trajectory(t, x_origin, y_origin, z_origin, segment_duration)

    # 对每个小段进行多项式拟合
    fitted_segments = fit_segments(segments, segment_duration, idx=idx)

    # 绘制原始轨迹和拟合曲线
    plot_trajectory_with_fitted_segments(t, x_origin, y_origin, z_origin, fitted_segments, idx=idx)
    plot_trajectory_3d(t, x_origin, y_origin, z_origin, fitted_segments, idx=idx)

    fitted_x = []
    fitted_y = []
    fitted_z = []
    for segment in fitted_segments:
        duration, x, y, z = segment
        fitted_x.append(x)
        fitted_y.append(y)
        fitted_z.append(z)
    fitted_x = np.concatenate(fitted_x)
    fitted_y = np.concatenate(fitted_y)
    fitted_z = np.concatenate(fitted_z)
    origin = np.array([x_origin, y_origin, z_origin])
    fitted = np.array([fitted_x, fitted_y, fitted_z])
    error = np.sum((fitted - origin)**2, axis=0).mean()
    print('fitted error of agent {}: '.format(idx), error)