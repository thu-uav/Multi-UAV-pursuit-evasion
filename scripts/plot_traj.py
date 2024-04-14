import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 从CSV文件中读取数据
data = pd.read_csv('/home/chenjy/OmniDrones/scripts/figure8.csv')

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
    return x, y, z

# 绘制轨迹
def plot_trajectory(durations, coeffs):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    for duration, coeff in zip(durations, coeffs):
        x, y, z = generate_trajectory(duration, coeff)
        ax.plot(x, y, z, label=f'Duration: {duration} s')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Trajectory')
    plt.legend()
    plt.savefig('figure8.png')

# 绘制轨迹
plot_trajectory(durations, coeffs)
