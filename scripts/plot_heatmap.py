import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy

def plot_objects(a, b, obstacles):
    plt.figure()
    plt.plot(a[0], a[1], 'bo', label='drone')
    plt.plot(b[0], b[1], 'ro', label='target')
    for obstacle in obstacles:
        circle = plt.Circle((obstacle[0], obstacle[1]), 0.3, color='g', fill=False, label='Obstacle')
        plt.gca().add_patch(circle)
    circle = plt.Circle((0.0, 0.0), 0.9, color='r', fill=False)
    plt.gca().add_patch(circle)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Objects and Obstacles')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('show_pos.png')

tasks = np.load('/home/chenjy/OmniDrones/scripts/outputs/v1_6_cl_woupdateheight_startheight0_3_threshold1toinf_randomp0_3/04-01_13-18/wandb/run-20240401_131842-7o1odr7n/files/tasks/tasks_1100.npy')
num_drone = 4
num_target = 1
num_active_cylinder = 3
num_all_cylinder = 5
drones_pos = tasks[:, :num_drone * 3]
target_pos = tasks[:, num_drone * 3: num_drone * 3 + num_target * 3]

num_idx_0 = (tasks[:, -5:].sum(-1)==0.0)
num_idx_1 = (tasks[:, -5:].sum(-1)==1.0)
num_idx_2 = (tasks[:, -5:].sum(-1)==2.0)
num_idx_3 = (tasks[:, -5:].sum(-1)==3.0)

active_cylinder_pos = tasks[num_idx_3][:, num_drone * 3 + num_target * 3: num_drone * 3 + num_target * 3 + num_active_cylinder * 3]
active_cylinder_pos1 = active_cylinder_pos[:, 0:3]
active_cylinder_pos2 = active_cylinder_pos[:, 3:6]
active_cylinder_pos3 = active_cylinder_pos[:, 6:9]
drone_pos3 = drones_pos[num_idx_3]
target_pos3 = target_pos[num_idx_3]

# plot pos
show_idx = 800
plot_objects(drone_pos3[show_idx][:3], target_pos3[show_idx][:3], active_cylinder_pos[show_idx].reshape(-1, 3)[:, :3])

# heatmap
show_pos = copy.deepcopy(active_cylinder_pos3)
# 绘制二维热度图（x-y 平面）
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist2d(show_pos[:, 0], show_pos[:, 1], bins=50, cmap='viridis')
plt.colorbar()
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('2D Heatmap (X-Y Plane)')

# 绘制一维热度图（z 轴）
plt.subplot(1, 2, 2)
plt.hist(show_pos[:, 2], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Z Label')
plt.ylabel('Frequency')
plt.title('1D Heatmap (Z Axis)')

plt.tight_layout()
plt.savefig('hm_cylinder3.png')
breakpoint()

np.random.seed(42)
# random
# Define the range for each dimension
dimension_ranges = [(-1.2, 1.2), (-1.2, 1.2), (0.0, 2.4), 
                    (-1.2, 1.2), (-1.2, 1.2), (0.0, 2.4),
                    (-1.2, 1.2), (-1.2, 1.2), (0.0, 2.4), 
                    (-1.2, 1.2), (-1.2, 1.2), (0.0, 2.4), # drone
                    (-1.2, 1.2), (-1.2, 1.2), (0.0, 2.4), # target
                    ]  # Define ranges for each of the three dimensions

# Generate random high-dimensional data
num_samples = 10000
num_dimensions = tasks.shape[-1]
X = np.zeros((num_samples, num_dimensions))
for i, (min_val, max_val) in enumerate(dimension_ranges):
    X[:, i] = np.random.uniform(min_val, max_val, size=num_samples)

# Perform PCA to reduce dimensionality to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(tasks)

# Plot the original and reduced data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='r', alpha=0.1)
plt.title('Data after PCA (2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.savefig('PCA.png')