import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy

def plot_objects(drones, b, obstacles):
    plt.figure()
    for drone in drones:
        plt.plot(drone[0], drone[1], 'bo', label='drone')
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
    plt.savefig('2D.png')

def plot_objects_3D(drones, b, obstacles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot drones
    for drone in drones:
        ax.scatter(drone[0], drone[1], drone[2], c='b', marker='o', label='Drone')

    # Plot target
    ax.scatter(b[0], b[1], b[2], c='r', marker='o', label='Target')

    def draw_cylinder(ax, x, y, z, height=1.2, radius=0.3, color='g'):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, height, 10) # Height of the cylinder is 1.2
        X = x + radius * np.outer(np.cos(u), np.ones_like(v))
        Y = y + radius * np.outer(np.sin(u), np.ones_like(v))
        Z = z + np.outer(np.ones_like(u), v)
        ax.plot_surface(X, Y, Z, color=color, alpha=0.5)

    # Plot obstacles as cylinders
    for obstacle in obstacles:
        x, y, z = obstacle
        if z == 0.0:
            height = 0.6
        else:
            height = 1.2
        draw_cylinder(ax, x, y, z=0.0, height=height)
    
    # plot wall
    draw_cylinder(ax, x=0.0, y=0.0, z=0.0, height=1.2, radius=0.9, color='gray')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Objects and Obstacles')
    ax.legend()
    plt.grid(True)
    plt.savefig('3D')

def get_dist(pos1, pos2):
    return (((pos1 - pos2)**2).sum(-1))**(0.5)

# check inside
arena_size = 0.9
max_height = 1.2
def check_inside(pos):
    x, y, z = pos
    if x**2 + y**2 > arena_size**2:
        return False
    if z < 0.0 or z > max_height:
        return False
    return True

tasks = np.load('/home/chenjy/OmniDrones/scripts/outputs/cl_new_reward_v1_2_catch0_12/04-15_12-19/wandb/run-20240415_121914-fd16d17a/files/tasks/tasks_3400.npy')
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

num_inside = 0
for idx in range(tasks.shape[0]):
    drones_pos_one = tasks[idx, :num_drone * 3].reshape(-1, 3)
    target_pos_one = tasks[idx, num_drone * 3: num_drone * 3 + num_target * 3]
    if check_inside(target_pos_one) and check_inside(drones_pos_one[0]) \
        and check_inside(drones_pos_one[1]) and check_inside(drones_pos_one[2]) \
        and check_inside(drones_pos_one[3]):
        num_inside += 1
print('num_inside', num_inside, 'num_tasks', tasks.shape[0])

# plot pos
show_idx = -8
plot_objects(drone_pos3[show_idx].reshape(-1, 3), target_pos3[show_idx], active_cylinder_pos[show_idx].reshape(-1, 3))
plot_objects_3D(drone_pos3[show_idx].reshape(-1, 3), target_pos3[show_idx], active_cylinder_pos[show_idx].reshape(-1, 3))
drone_target_dist = get_dist(drones_pos.reshape(-1, 4, 3), target_pos[:,np.newaxis])

# heatmap
show_pos = copy.deepcopy(active_cylinder_pos1)
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
plt.savefig('hm_cylinder1.png')
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