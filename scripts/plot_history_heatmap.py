import numpy as np
import matplotlib.pyplot as plt

data = np.load("/home/jiayu/OmniDrones/scripts/outputs/unif03_R05to09_samplenearby_withcylinders_expandstep0_1/08-31_22-18/wandb/run-20240831_221829-hwi2i9sy/files/history_22800.npy")
n = data.shape[0]

num_drones = 3
num_obstacles = 5

drones_coords = data[:, :9].reshape(-1, num_drones, 3)  # [n, 3, 3]
target_coords = data[:, 9:12]  # [n, 3]
obstacle_coords = data[:, 12:27].reshape(-1, num_obstacles, 3)  # [n, 5, 3]

# 生成热度图
fig, axs = plt.subplots(2, 5, figsize=(15, 7))

# 计算格点数量
x_max = 0.5
x_min = -0.5
y_max = 0.5
y_min = -0.5
x_bins = int((x_max - x_min) / 0.2)
y_bins = int((y_max - y_min) / 0.2)

# 无人机热度图
for i in range(3):
    axs[0, i].set_title(f'Drone {i+1} Heatmap')
    axs[0, i].set_xlabel('X')
    axs[0, i].set_ylabel('Y')
    hist, xedges, yedges, img = axs[0, i].hist2d(drones_coords[:, i, 0], drones_coords[:, i, 1], bins=[x_bins, y_bins], range=[[x_min, x_max], [y_min, y_max]], cmap='viridis')
    for x in range(hist.shape[0]):
        for y in range(hist.shape[1]):
            axs[0, i].text(xedges[x] + (xedges[x+1] - xedges[x]) / 2, yedges[y] + (yedges[y+1] - yedges[y]) / 2, f'{hist[x, y]}', color='w', ha='center', va='center')

# 目标点热度图
axs[0, 3].set_title('Target Heatmap')
axs[0, 3].set_xlabel('X')
axs[0, 3].set_ylabel('Y')
hist, xedges, yedges, img = axs[0, 3].hist2d(target_coords[:, 0], target_coords[:, 1], bins=[x_bins, y_bins], range=[[x_min, x_max], [y_min, y_max]], cmap='viridis')
for x in range(hist.shape[0]):
    for y in range(hist.shape[1]):
        axs[0, 3].text(xedges[x] + (xedges[x+1] - xedges[x]) / 2, yedges[y] + (yedges[y+1] - yedges[y]) / 2, f'{hist[x, y]}', color='w', ha='center', va='center')

# 障碍物热度图
for i in range(5):
    axs[1, i].set_title(f'Obstacle {i+1} Heatmap')
    axs[1, i].set_xlabel('X')
    axs[1, i].set_ylabel('Y')
    hist, xedges, yedges, img = axs[1, i].hist2d(obstacle_coords[:, i, 0], obstacle_coords[:, i, 1], bins=[x_bins, y_bins], range=[[x_min, x_max], [y_min, y_max]], cmap='viridis')
    for x in range(hist.shape[0]):
        for y in range(hist.shape[1]):
            axs[1, i].text(xedges[x] + (xedges[x+1] - xedges[x]) / 2, yedges[y] + (yedges[y+1] - yedges[y]) / 2, f'{hist[x, y]}', color='w', ha='center', va='center')

axs[0, -1].axis('off')

plt.tight_layout()
plt.savefig('heatmap')