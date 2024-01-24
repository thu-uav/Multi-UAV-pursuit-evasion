import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

# 创建数据
np.random.seed(42)
t = torch.arange(0, 1000)
x = -1.1 * torch.sin(2 * t) - 0.5 * torch.sin(3 * t)
y = 1.1 * torch.cos(2 * t) - 0.5 * torch.cos(3 * t)
z = torch.zeros_like(t)

# 创建3D图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
ax.scatter(x, y, z, c='r', marker='o')

# 设置坐标轴标签
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')

# 显示图形
plt.savefig('test')
