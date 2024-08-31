import numpy as np
import matplotlib.pyplot as plt

data = np.load("/home/jiayu/OmniDrones/scripts/outputs/unif07_R03to07_fps_wostopbuffer_stop09/08-30_15-53/wandb/run-20240830_155338-qg2silf4/files/history_5000.npy")
n = data.shape[0]

num_drones = 3
num_obstacles = 5

# 选择第i个数据
i = -3

# 提取无人机、目标点和障碍物的坐标
drones_coords = data[i, :9].reshape(num_drones, 3)  # [3, 3]
target_coords = data[i, 9:12]  # [3]
obstacle_coords = data[i, 12:27].reshape(num_obstacles, 3)  # [5, 3]

# 计算坐标范围
x_min = min(np.min(drones_coords[:, 0]), target_coords[0], np.min(obstacle_coords[:, 0]))
x_max = max(np.max(drones_coords[:, 0]), target_coords[0], np.max(obstacle_coords[:, 0]))
y_min = min(np.min(drones_coords[:, 1]), target_coords[1], np.min(obstacle_coords[:, 1]))
y_max = max(np.max(drones_coords[:, 1]), target_coords[1], np.max(obstacle_coords[:, 1]))

# 计算格点数量
x_bins = int((x_max - x_min) / 0.2)
y_bins = int((y_max - y_min) / 0.2)

# 生成热度图
fig, ax = plt.subplots(figsize=(10, 10))

# 无人机热度图
for j in range(3):
    ax.scatter(drones_coords[j, 0], drones_coords[j, 1], label=f'Drone {j+1}', s=100, marker='*', color='blue')

# 目标点热度图
ax.scatter(target_coords[0], target_coords[1], label='Target', s=100, color='green', marker='^')

# 障碍物热度图
for j in range(5):
    circle = plt.Circle((obstacle_coords[j, 0], obstacle_coords[j, 1]), 0.1, color='red', fill=True)
    ax.add_artist(circle)
    ax.text(obstacle_coords[j, 0], obstacle_coords[j, 1], f'Ob {j+1}', color='black', fontsize=12)

# 设置图例和标签
ax.set_title('Spatial Distribution of Drones, Target, and Obstacles')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

# 设置坐标轴范围
ax.set_xlim(x_min - 0.1, x_max + 0.1)
ax.set_ylim(y_min - 0.1, y_max + 0.1)

plt.tight_layout()
plt.savefig("2d-task")