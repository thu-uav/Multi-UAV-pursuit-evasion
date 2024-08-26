import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 生成task space
num_points = 10000
# 定义每个维度的范围
arena_size = 0.9
max_height = 1.2
dimension_ranges = [
    (-arena_size, arena_size),
    (-arena_size, arena_size),
    # (0.0, max_height),
    (max_height / 2 - 0.1, max_height / 2 + 0.1),
    (-arena_size, arena_size),
    (-arena_size, arena_size),
    # (0.0, max_height),
    (max_height / 2 - 0.1, max_height / 2 + 0.1),
    (-arena_size, arena_size),
    (-arena_size, arena_size),
    # (0.0, max_height), # agents
    (max_height / 2 - 0.1, max_height / 2 + 0.1),
    (-arena_size, arena_size),
    (-arena_size, arena_size),
    # (0.0, max_height), # target
    (max_height / 2 - 0.1, max_height / 2 + 0.1),
    (-arena_size, arena_size),
    (-arena_size, arena_size),
    (0.5 * max_height, 0.5 * max_height),
    (-arena_size, arena_size),
    (-arena_size, arena_size),
    (0.5 * max_height, 0.5 * max_height),
    (-arena_size, arena_size),
    (-arena_size, arena_size),
    (0.5 * max_height, 0.5 * max_height),
    (-arena_size, arena_size),
    (-arena_size, arena_size),
    (0.5 * max_height, 0.5 * max_height),
    (-arena_size, arena_size),
    (-arena_size, arena_size),
    (0.5 * max_height, 0.5 * max_height), # cylinders
]

num_points = 10000
num_dimensions = len(dimension_ranges)

task_space = np.zeros((num_points, num_dimensions))
for i, (low, high) in enumerate(dimension_ranges):
    task_space[:, i] = np.random.uniform(low, high, num_points)

A = np.load('/home/jiayu/OmniDrones/scripts/outputs/particle_TP_unif03_R03to07_buffer10000/08-25_19-19/wandb/run-20240825_191916-700arp79/files/history_4900.npy')
# A = np.load('/home/jiayu/OmniDrones/scripts/outputs/particle_TP_unif01_R03to07/08-25_12-55/wandb/run-20240825_125546-x5vdz221/files/tasks_2800.npy')
breakpoint()

pca = PCA(n_components=2)  # 只取前两个主成分
task_space_pca = pca.fit_transform(task_space)
A_pca = pca.transform(A)

plt.figure(figsize=(10, 6))
plt.scatter(task_space_pca[:, 0], task_space_pca[:, 1], s=10, c='blue', label='Task Space')
plt.scatter(A_pca[:, 0], A_pca[:, 1], s=20, c='orange', label='History Projection')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Projection of Task Space and History buffer')
plt.legend()
plt.grid(True)
plt.savefig('PCA')