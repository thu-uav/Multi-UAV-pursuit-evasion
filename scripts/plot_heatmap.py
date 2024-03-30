import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy
tasks = np.load('/home/chenjy/OmniDrones/scripts/outputs/v1_6_cl_emptybuffer_from_heightbound0_3_threshold_1toinf/03-30_17-53/wandb/run-20240330_175303-f3mpf9ez/files/tasks/tasks_492.npy')
num_drone = 4
num_target = 1
num_active_cylinder = 1
num_all_cylinder = 5
drones_pos = tasks[:, :num_drone * 3]
target_pos = tasks[:, num_drone * 3: num_drone * 3 + num_target * 3]
active_cylinder_pos = tasks[:, num_drone * 3 + num_target * 3: num_drone * 3 + num_target * 3 + num_active_cylinder * 3]
cylinder_mask = tasks[:, -5:]

# heatmap
show_pos = copy.deepcopy(drones_pos)
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
plt.savefig('hm_drone.png')
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