import numpy as np
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

def get_occupation_matrix(position_list, matrix_size, grid_size, name):
    occupy = np.zeros((matrix_size, matrix_size))
    reference_grid = [0, 0]
    reference_pos = [-1.0 + grid_size / 2, -1.0 + grid_size / 2] # left corner
    for pos in position_list:
        x_grid = round((pos[0] - reference_pos[0]) / grid_size) + reference_grid[0]
        y_grid = round((pos[1] - reference_pos[1]) / grid_size) + reference_grid[1]
        occupy[int(x_grid), int(y_grid)] += 1
    plot_heatmap(occupy, name) 

def plot_heatmap(matrix, name):
    # matrix [n, m]
    plt.figure()
    plt.imshow(matrix, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Heatmap')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig('{}.png'.format(name))

# generate 8 cylinders outside the arena
def generate_outside_cylinders_x_y(arena_size, num_envs, device):
    cylinder_inactive_pos_r = torch.tensor([arena_size + 1.0, arena_size + 1.0,
                                            arena_size + 1.0, arena_size + 1.0,
                                            arena_size + 1.0, arena_size + 1.0,
                                            arena_size + 1.0, arena_size + 1.0], device=device)
    cylinder_inactive_pos_angle = torch.tensor([0.0, torch.pi / 4,
                                                torch.pi / 2, torch.pi * 3 / 4,
                                                torch.pi, torch.pi * 5 / 4,
                                                torch.pi * 3 / 2, torch.pi * 7 / 4], device=device)
    # cylinder_inactive_pos_z = torch.tensor([arena_size, arena_size,
    #                                         arena_size, arena_size,
    #                                         arena_size, arena_size,
    #                                         arena_size, arena_size], device=device).unsqueeze(-1)
    cylinder_inactive_pos_x = (cylinder_inactive_pos_r * torch.cos(cylinder_inactive_pos_angle)).unsqueeze(-1)
    cylinder_inactive_pos_y = (cylinder_inactive_pos_r * torch.sin(cylinder_inactive_pos_angle)).unsqueeze(-1)
    cylinders_pos = torch.concat([cylinder_inactive_pos_x, cylinder_inactive_pos_y], dim=-1)
    if num_envs > 1:
        cylinders_pos = cylinders_pos.unsqueeze(0).expand(num_envs, -1, -1)
    return cylinders_pos

# DFS: check whether exists a fessible path
def is_valid_move(matrix, row, col, visited):
    rows, cols = len(matrix), len(matrix[0])
    return 0 <= row < rows and 0 <= col < cols and matrix[row][col] == 0 and (row, col) not in visited

def dfs(matrix, start, target, visited):
    row, col = start

    if start == target:
        return True

    visited.add((row, col))

    # up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if is_valid_move(matrix, new_row, new_col, visited):
            if dfs(matrix, (new_row, new_col), target, visited):
                return True

    return False

def has_path(matrix, start, target):
    visited = set()
    return dfs(matrix, start, target, visited)

def has_feasible_path(matrix, start_points, target):
    reached_targets = 0
    num_threshold = 3
    for start_point in start_points:
        visited = set()
        if dfs(matrix, start_point, target, visited):
            reached_targets += 1
            if reached_targets >= num_threshold:
                return True
    return False

###################################################

def rejection_sampling_drone_target_xy(arena_size, cylinder_size, num_drones, device, occupancy_matrix):
    # occupancy_matrix : (matrix_size, matrix_size) 
    grid_size = 2 * cylinder_size
    matrix_size = int(2 * arena_size / grid_size)
    origin_grid = [int(matrix_size / 2), int(matrix_size / 2)]
    origin_pos = [0.0, 0.0] # left corner   
    # expand to 10 * 10, and set corner to 1.0
    occupancy_matrix = np.kron(occupancy_matrix, np.ones((2, 2), dtype=occupancy_matrix.dtype))
    # set disabled idx to 1.0
    # up
    occupancy_matrix[0,0] = 1.0; occupancy_matrix[0,1] = 1.0; occupancy_matrix[0,2] = 1.0
    occupancy_matrix[0, -1] = 1.0; occupancy_matrix[0, -2] = 1.0; occupancy_matrix[0, -3] = 1.0
    occupancy_matrix[1,0] = 1.0; occupancy_matrix[1,1] = 1.0
    occupancy_matrix[1,-1] = 1.0; occupancy_matrix[1,-2] = 1.0
    occupancy_matrix[2,0] = 1.0; occupancy_matrix[2,-1] = 1.0
    # down
    occupancy_matrix[-1,0] = 1.0; occupancy_matrix[-1,1] = 1.0; occupancy_matrix[-1,2] = 1.0
    occupancy_matrix[-1, -1] = 1.0; occupancy_matrix[-1, -2] = 1.0; occupancy_matrix[-1, -3] = 1.0
    occupancy_matrix[-2,0] = 1.0; occupancy_matrix[-2,1] = 1.0
    occupancy_matrix[-2,-1] = 1.0; occupancy_matrix[-2,-2] = 1.0
    occupancy_matrix[-3,0] = 1.0; occupancy_matrix[-3,-1] = 1.0

    drone_target_pos = []
    start_grid = ()
    target_grid = ()
    small_reference_grid = [0, 0]
    small_reference_pos = [-0.9, -0.9]
    small_grid_size = grid_size / 2.0
    small_matrix_size = len(occupancy_matrix)
    
    for idx in range(num_drones + 1):
        while True:
            grid_idx = torch.randint(0, len(occupancy_matrix), (2,))
            x_grid = int(grid_idx[0])
            y_grid = int(grid_idx[1])
            
            x = (x_grid - small_reference_grid[0]) * small_grid_size + small_reference_pos[0]
            y = (y_grid - small_reference_grid[1]) * small_grid_size + small_reference_pos[1]

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < small_matrix_size and y_grid >= 0 and y_grid < small_matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    drone_target_pos.append(torch.tensor([x, y], device=device))
                    occupancy_matrix[x_grid, y_grid] = 1
                    if idx >= num_drones: # target
                        target_grid += (x_grid, y_grid)
                    else: # drones
                        start_grid += ((x_grid, y_grid), )
                    break
    drone_target_pos = torch.stack(drone_target_pos)
    
    return drone_target_pos, occupancy_matrix, start_grid, target_grid

def rejection_sampling_all_obj_xy(arena_size, cylinder_size, num_drones, num_cylinders, device):
    # set cylinders by rejection sampling
    grid_size = 2 * cylinder_size
    matrix_size = int(2 * arena_size / grid_size)
    origin_grid = [int(matrix_size / 2), int(matrix_size / 2)]
    origin_pos = [0.0, 0.0] # left corner
    occupancy_matrix = np.zeros((matrix_size, matrix_size))
    cylinder_occupancy_matrix = np.zeros((matrix_size, matrix_size))
    
    # first randomize the grid pos of cylinders
    # occupy the matrix
    x_y_grid_list = [[0,2], [1,1], [1,2], [1,3], [2,0], [2,1], [2,2], [2,3], [2,4], \
                        [3,1], [3,2], [3,3], [4,2]]
    cylinders_pos = []
    for cylinder_idx in range(num_cylinders):
        while True:
            grid_idx = torch.randint(0, len(x_y_grid_list), (1,)).item()
            x_grid = int(x_y_grid_list[grid_idx][0])
            y_grid = int(x_y_grid_list[grid_idx][1])
            # x_grid = torch.randint(0, matrix_size, (1,))
            # y_grid = torch.randint(0, matrix_size, (1,))
            
            x = (x_grid - origin_grid[0]) * grid_size
            y = (y_grid - origin_grid[1]) * grid_size

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < matrix_size and y_grid >= 0 and y_grid < matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    cylinders_pos.append(torch.tensor([x, y], device=device))
                    occupancy_matrix[x_grid, y_grid] = 1
                    break
    if num_cylinders == 0:
        cylinders_pos = torch.tensor([], device=device)
    else:
        cylinders_pos = torch.stack(cylinders_pos)
    
    # expand to 10 * 10, and set corner to 1.0
    occupancy_matrix = np.kron(occupancy_matrix, np.ones((2, 2), dtype=occupancy_matrix.dtype))
    # set disabled idx to 1.0
    # up
    occupancy_matrix[0,0] = 1.0; occupancy_matrix[0,1] = 1.0; occupancy_matrix[0,2] = 1.0
    occupancy_matrix[0, -1] = 1.0; occupancy_matrix[0, -2] = 1.0; occupancy_matrix[0, -3] = 1.0
    occupancy_matrix[1,0] = 1.0; occupancy_matrix[1,1] = 1.0
    occupancy_matrix[1,-1] = 1.0; occupancy_matrix[1,-2] = 1.0
    occupancy_matrix[2,0] = 1.0; occupancy_matrix[2,-1] = 1.0
    # down
    occupancy_matrix[-1,0] = 1.0; occupancy_matrix[-1,1] = 1.0; occupancy_matrix[-1,2] = 1.0
    occupancy_matrix[-1, -1] = 1.0; occupancy_matrix[-1, -2] = 1.0; occupancy_matrix[-1, -3] = 1.0
    occupancy_matrix[-2,0] = 1.0; occupancy_matrix[-2,1] = 1.0
    occupancy_matrix[-2,-1] = 1.0; occupancy_matrix[-2,-2] = 1.0
    occupancy_matrix[-3,0] = 1.0; occupancy_matrix[-3,-1] = 1.0
    cylinder_occupancy_matrix = occupancy_matrix.copy()

    drone_target_pos = []
    start_grid = ()
    target_grid = ()
    small_reference_grid = [0, 0]
    small_reference_pos = [-0.9, -0.9]
    small_grid_size = grid_size / 2.0
    small_matrix_size = len(occupancy_matrix)
    
    for idx in range(num_drones + 1):
        while True:
            grid_idx = torch.randint(0, len(occupancy_matrix), (2,))
            x_grid = int(grid_idx[0])
            y_grid = int(grid_idx[1])
            
            x = (x_grid - small_reference_grid[0]) * small_grid_size + small_reference_pos[0]
            y = (y_grid - small_reference_grid[1]) * small_grid_size + small_reference_pos[1]

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < small_matrix_size and y_grid >= 0 and y_grid < small_matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    drone_target_pos.append(torch.tensor([x, y], device=device))
                    occupancy_matrix[x_grid, y_grid] = 1
                    if idx >= num_drones: # target
                        target_grid += (x_grid, y_grid)
                    else: # drones
                        start_grid += ((x_grid, y_grid), )
                    break
    drone_target_pos = torch.stack(drone_target_pos)
    objects_pos = torch.concat([drone_target_pos, cylinders_pos])
    drone_target_occupancy_matrix = occupancy_matrix - cylinder_occupancy_matrix
    
    return objects_pos, occupancy_matrix, drone_target_occupancy_matrix, \
        cylinder_occupancy_matrix, start_grid, target_grid

def rejection_sampling_all_obj_xy_debug(arena_size, cylinder_size, num_drones, num_cylinders, device):
    # set cylinders by rejection sampling
    grid_size = 2 * cylinder_size
    matrix_size = int(2 * arena_size / grid_size)
    origin_grid = [int(matrix_size / 2), int(matrix_size / 2)]
    origin_pos = [0.0, 0.0] # left corner
    occupancy_matrix = np.zeros((matrix_size, matrix_size))
    cylinder_occupancy_matrix = np.zeros((matrix_size, matrix_size))
    
    # first randomize the grid pos of cylinders
    # occupy the matrix
    x_y_grid_list = [[0,2], [1,1], [1,2], [1,3], [2,0], [2,1], [2,2], [2,3], [2,4], \
                        [3,1], [3,2], [3,3], [4,2]]
    cylinders_pos = []
    for cylinder_idx in range(num_cylinders):
        while True:
            grid_idx = torch.randint(0, len(x_y_grid_list), (1,)).item()
            x_grid = int(x_y_grid_list[grid_idx][0])
            y_grid = int(x_y_grid_list[grid_idx][1])
            # x_grid = torch.randint(0, matrix_size, (1,))
            # y_grid = torch.randint(0, matrix_size, (1,))
            
            x = (x_grid - origin_grid[0]) * grid_size
            y = (y_grid - origin_grid[1]) * grid_size

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < matrix_size and y_grid >= 0 and y_grid < matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    cylinders_pos.append(torch.tensor([x, y], device=device))
                    occupancy_matrix[x_grid, y_grid] = 1
                    break
    if num_cylinders == 0:
        cylinders_pos = torch.tensor([], device=device)
    else:
        cylinders_pos = torch.stack(cylinders_pos)
    cylinder_occupancy_matrix = occupancy_matrix.copy()
    
    # expand to 10 * 10, and set corner to 1.0
    occupancy_matrix = np.kron(occupancy_matrix, np.ones((2, 2), dtype=occupancy_matrix.dtype))
    # set disabled idx to 1.0
    # up
    occupancy_matrix[0,0] = 1.0; occupancy_matrix[0,1] = 1.0; occupancy_matrix[0,2] = 1.0
    occupancy_matrix[0, -1] = 1.0; occupancy_matrix[0, -2] = 1.0; occupancy_matrix[0, -3] = 1.0
    occupancy_matrix[1,0] = 1.0; occupancy_matrix[1,1] = 1.0
    occupancy_matrix[1,-1] = 1.0; occupancy_matrix[1,-2] = 1.0
    occupancy_matrix[2,0] = 1.0; occupancy_matrix[2,-1] = 1.0
    # down
    occupancy_matrix[-1,0] = 1.0; occupancy_matrix[-1,1] = 1.0; occupancy_matrix[-1,2] = 1.0
    occupancy_matrix[-1, -1] = 1.0; occupancy_matrix[-1, -2] = 1.0; occupancy_matrix[-1, -3] = 1.0
    occupancy_matrix[-2,0] = 1.0; occupancy_matrix[-2,1] = 1.0
    occupancy_matrix[-2,-1] = 1.0; occupancy_matrix[-2,-2] = 1.0
    occupancy_matrix[-3,0] = 1.0; occupancy_matrix[-3,-1] = 1.0

    drone_target_pos = []
    start_grid = ()
    target_grid = ()
    small_reference_grid = [0, 0]
    small_reference_pos = [-0.9, -0.9]
    small_grid_size = grid_size / 2.0
    small_matrix_size = len(occupancy_matrix)
    
    for idx in range(num_drones + 1):
        while True:
            grid_idx = torch.randint(0, len(occupancy_matrix), (2,))
            x_grid = int(grid_idx[0])
            y_grid = int(grid_idx[1])
            
            x = (x_grid - small_reference_grid[0]) * small_grid_size + small_reference_pos[0]
            y = (y_grid - small_reference_grid[1]) * small_grid_size + small_reference_pos[1]

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < small_matrix_size and y_grid >= 0 and y_grid < small_matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    drone_target_pos.append(torch.tensor([x, y], device=device))
                    occupancy_matrix[x_grid, y_grid] = 1
                    if idx >= num_drones: # target
                        target_grid += (x_grid, y_grid)
                    else: # drones
                        start_grid += ((x_grid, y_grid), )
                    break
    drone_target_pos = torch.stack(drone_target_pos)
    objects_pos = torch.concat([drone_target_pos, cylinders_pos])
    drone_target_occupancy_matrix = occupancy_matrix
    
    return objects_pos, occupancy_matrix, drone_target_occupancy_matrix, \
        cylinder_occupancy_matrix, start_grid, target_grid

def rejection_sampling_with_validation(arena_size, cylinder_size, num_drones, num_cylinders, device, use_validation=True):
    if not use_validation:
        task_one, occupancy_matrix, drone_target_occupancy_matrix, \
            cylinder_occupancy_matrix, start_grid, target_grid = \
            rejection_sampling_all_obj_xy(arena_size=arena_size, 
                                                cylinder_size=cylinder_size, 
                                                num_drones=num_drones, 
                                                num_cylinders=num_cylinders, 
                                                device=device)
        return task_one, occupancy_matrix, drone_target_occupancy_matrix, cylinder_occupancy_matrix
    else:
        num_loop = 0
        while(num_loop <= 100):
            task_one, occupancy_matrix, drone_target_occupancy_matrix, \
                cylinder_occupancy_matrix, start_grid, target_grid = \
                rejection_sampling_all_obj_xy(arena_size=arena_size, 
                                                    cylinder_size=cylinder_size, 
                                                    num_drones=num_drones, 
                                                    num_cylinders=num_cylinders, 
                                                    device=device)
            if has_feasible_path(cylinder_occupancy_matrix, start_grid, target_grid):
                return task_one, occupancy_matrix, drone_target_occupancy_matrix, cylinder_occupancy_matrix
            num_loop += 1
        raise NotImplementedError

# plot
def plot_fessible_scenario():
    # 创建画布
    fig, ax = plt.subplots(figsize=(5, 5))

    # 绘制矩形
    rectangle = patches.Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rectangle)

    # 绘制同心圆
    circle = plt.Circle((0.5, 0.5), 0.5, edgecolor='black', facecolor='none')
    ax.add_patch(circle)

    # 创建5x5的格子
    for i in range(1, 5):
        ax.plot([0, 1], [i/5, i/5], color='red', linewidth=2)  # 横线
        ax.plot([i/5, i/5], [0, 1], color='red', linewidth=2)  # 竖线

    # 创建10x10的格子
    for i in range(1, 10):
        ax.plot([0, 1], [i/10, i/10], color='black', linewidth=1)  # 横线
        ax.plot([i/10, i/10], [0, 1], color='black', linewidth=1)  # 竖线

    # 设置坐标轴范围
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 隐藏坐标轴刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 显示图形
    plt.savefig('fessible.png')

def check_path():
    task_list = []
    fessible_list = []
    torch.manual_seed(1)
    num_cylinders = 3
    check_occupy = []
    for _ in range(10000):
        task_one, occupancy_matrix, drone_target_occupancy_matrix, \
            cylinder_occupancy_matrix, start_grid, target_grid = \
            rejection_sampling_all_obj_xy_debug(arena_size=1.0, 
                                                 cylinder_size=0.2, 
                                                 num_drones=4, 
                                                 num_cylinders=num_cylinders, 
                                                 device='cpu')
        task_list.append(task_one.numpy().reshape(-1))
        fessible_list.append(has_feasible_path(cylinder_occupancy_matrix, start_grid, target_grid))
        check_occupy.append(cylinder_occupancy_matrix)
    task_list = np.array(task_list)
    cylinder_poses = task_list[:, -2 * num_cylinders:]
    check_pose = np.array([0.0, 0.0, 0.4, 0.0, -0.4, 0.0])
    check_matrix = np.array(
                        [[0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., 0.],
                        [0., 0., 1., 0., 0.],
                        [0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0.]]
                        )
    check_occupy = np.array(check_occupy)
    cylinder_3_line = cylinder_poses[np.argwhere((check_occupy == check_matrix).reshape(10000, -1).sum(-1) == 25)]
    breakpoint()
    print('num_fessible', np.sum(fessible_list))

def check_dist(): # check drone, target and cylinder dist
    # task_list = np.load('/home/chenjy/OmniDrones/scripts/outputs/Disagreement_emptytransfer_0and1/03-04_18-08/wandb/run-20240304_180822-3e0txp70/files/tasks/tasks_2313.npy')
    # weights_list = np.load('/home/chenjy/OmniDrones/scripts/outputs/Disagreement_emptytransfer_0and1/03-04_18-08/wandb/run-20240304_180822-3e0txp70/files/tasks/weights_2313.npy')
    task_list = []
    cylinder_occupancy = np.zeros((10, 10))
    drone_target_occupancy = np.zeros((10, 10))
    for _ in range(10000):
        task_one, occupancy_matrix, drone_target_occupancy_matrix, \
            cylinder_occupancy_matrix = \
            rejection_sampling_with_validation(arena_size=1.0, 
                                                 cylinder_size=0.2, 
                                                 num_drones=4, 
                                                 num_cylinders=3, 
                                                 device='cpu')
        task_list.append(task_one.numpy().reshape(-1))
        cylinder_occupancy += cylinder_occupancy_matrix
        drone_target_occupancy += drone_target_occupancy_matrix
    task_list = np.array(task_list)
    drone_pos = task_list[:, :8]
    target_pos = task_list[:, 8:10]
    cylinder_pos = task_list[:, 10:]
    
    get_occupation_matrix(drone_pos[:, :2], matrix_size=10, grid_size=0.2, name='drone0')
    get_occupation_matrix(drone_pos[:, 2:4], matrix_size=10, grid_size=0.2, name='drone1')
    get_occupation_matrix(drone_pos[:, 4:6], matrix_size=10, grid_size=0.2, name='drone2')
    get_occupation_matrix(drone_pos[:, 6:8], matrix_size=10, grid_size=0.2, name='drone3')
    get_occupation_matrix(target_pos[:, :2], matrix_size=10, grid_size=0.2, name='target')
    get_occupation_matrix(cylinder_pos[:, :2], matrix_size=5, grid_size=0.4, name='cylinder1')
    get_occupation_matrix(cylinder_pos[:, 2:4], matrix_size=5, grid_size=0.4, name='cylinder2')
    plot_heatmap(drone_target_occupancy, 'drone_target_check')
    plot_heatmap(cylinder_occupancy, 'cylinder_check')
    
    # check disagreement
    # drone_pos = task_list[:, :12]
    # target_pos = task_list[:, 12:15]
    # cylinder_pos = task_list[:, 15:39]
    # cylinder_mask = task_list[:, 39:]
    # cylinder1 = cylinder_pos[cylinder_mask.sum(-1) == 1.0]
    # get_occupation_matrix(drone_pos[:, :3], 'drone0')
    # get_occupation_matrix(drone_pos[:, 3:6], 'drone1')
    # get_occupation_matrix(drone_pos[:, 6:9], 'drone2')
    # get_occupation_matrix(drone_pos[:, 9:12], 'drone3')
    # get_occupation_matrix(target_pos[:, :3], 'target')
    # get_occupation_matrix(cylinder1[:, :3], 'cylinder1')

if __name__ == '__main__':
    check_path()
    # check_dist()
    # check = generate_outside_cylinders_x_y(arena_size=1.0, 
    #                                        num_envs=10, 
    #                                        device='cpu')