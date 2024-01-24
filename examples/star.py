import torch
import matplotlib.pyplot as plt
import numpy as np

def pentagram(t):
    # 半径
    r = 1.0
    
    # 五角星的参数方程
    x = -15 * torch.sin(2 * t) - 5 * torch.sin(3 * t)
    y = 15 * torch.cos(2 * t) - 5 * torch.cos(3 * t)
    z = torch.zeros_like(t)

    return torch.stack([x, y, z], dim=-1)

# 生成一系列的 t 值
t_values = torch.linspace(0, 7, 1000)

# 调用函数获取五角星的坐标
star_coordinates = pentagram(t_values)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(star_coordinates[:, 0], star_coordinates[:, 1], star_coordinates[:, 2], s=5, label='rl')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.savefig('star')