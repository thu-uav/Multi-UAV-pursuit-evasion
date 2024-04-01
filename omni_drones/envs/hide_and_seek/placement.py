import numpy as np
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

def get_occupation_matrix(position_list, arena_size, matrix_size, grid_size, name):
    occupy = np.zeros((matrix_size, matrix_size))
    reference_grid = [0, 0]
    reference_pos = [-arena_size + grid_size / 2, -arena_size + grid_size / 2] # left corner
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
def generate_outside_cylinders_x_y(arena_size, num_envs, device, num_active=8):
    cylinder_inactive_pos_r = torch.tensor([2 * arena_size, 2 * arena_size,
                                            2 * arena_size, 2 * arena_size,
                                            2 * arena_size, 2 * arena_size,
                                            2 * arena_size, 2 * arena_size], device=device)
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
    return cylinders_pos[:num_active]

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
    num_threshold = 4
    for start_point in start_points:
        visited = set()
        if dfs(matrix, start_point, target, visited):
            reached_targets += 1
            if reached_targets >= num_threshold:
                return True
    return False

###################################################
# 5 active cylinders

# for evaluation
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

# cl_bound: generate drone and target in [-bound, bound], small scene
def generate_cylinder_xy(arena_size, cylinder_size, num_cylinders, device):
    # set cylinders by rejection sampling
    grid_size = 2 * cylinder_size
    matrix_size = int(2 * arena_size / grid_size)
    origin_grid = [int(matrix_size / 2), int(matrix_size / 2)]
    origin_pos = [0.0, 0.0] # left corner
    occupancy_matrix = np.zeros((matrix_size, matrix_size))

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
   
    return cylinders_pos, occupancy_matrix

def generate_drone_target_xy_after_cylinder(cylinder_size, num_drones, device, occupancy_matrix, cl_bound=5):
    # init
    grid_size = 2 * cylinder_size

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
    
    # cl_bound
    for idx in range(num_drones + 1):
        num_loop_inner = 0
        while num_loop_inner <= 10:
            # x_grid: [5 - cl_bound, 4 + cl_bound]
            # y_grid: [5 - cl_bound, 4 + cl_bound]
            grid_idx = torch.randint(5 - cl_bound, 5 + cl_bound, (2,))
            # grid_idx = torch.randint(0, len(occupancy_matrix), (2,))
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
            num_loop_inner += 1
    
    drone_target_pos = torch.stack(drone_target_pos)
    drone_target_occupancy_matrix = occupancy_matrix - cylinder_occupancy_matrix
    
    return drone_target_pos, occupancy_matrix, cylinder_occupancy_matrix, drone_target_occupancy_matrix, \
                start_grid, target_grid

def rejection_sampling_all_obj_xy_cl(arena_size, cylinder_size, num_drones, num_cylinders, device, cl_bound=5):        
    while True:
        cylinders_pos, occupancy_matrix = generate_cylinder_xy(arena_size, cylinder_size, num_cylinders, device)
        
        drone_target_pos, occupancy_matrix, cylinder_occupancy_matrix, drone_target_occupancy_matrix, \
            start_grid, target_grid = generate_drone_target_xy_after_cylinder(cylinder_size, num_drones, device, occupancy_matrix, cl_bound=cl_bound)
        
        if drone_target_pos.shape[0] == (num_drones + 1):
            break
    
    objects_pos = torch.concat([drone_target_pos, cylinders_pos])
    
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

def rejection_sampling_with_validation_cl(arena_size, cylinder_size, num_drones, num_cylinders, device, use_validation=True, cl_bound=5):
    if not use_validation:
        task_one, occupancy_matrix, drone_target_occupancy_matrix, \
            cylinder_occupancy_matrix, start_grid, target_grid = \
            rejection_sampling_all_obj_xy_cl(arena_size=arena_size, 
                                                cylinder_size=cylinder_size, 
                                                num_drones=num_drones, 
                                                num_cylinders=num_cylinders, 
                                                device=device,
                                                cl_bound=cl_bound)
        return task_one, occupancy_matrix, drone_target_occupancy_matrix, cylinder_occupancy_matrix
    else:
        num_loop = 0
        while(num_loop <= 100):
            task_one, occupancy_matrix, drone_target_occupancy_matrix, \
                cylinder_occupancy_matrix, start_grid, target_grid = \
                rejection_sampling_all_obj_xy_cl(arena_size=arena_size, 
                                                    cylinder_size=cylinder_size, 
                                                    num_drones=num_drones, 
                                                    num_cylinders=num_cylinders, 
                                                    device=device,
                                                    cl_bound=cl_bound)
            if has_feasible_path(cylinder_occupancy_matrix, start_grid, target_grid):
                return task_one, occupancy_matrix, drone_target_occupancy_matrix, cylinder_occupancy_matrix
            num_loop += 1
        raise NotImplementedError

###################################################

# 3 active cylinders
def rejection_sampling_all_obj_large_cylinder(arena_size, max_height, cylinder_size, num_drones, num_cylinders, device):
    # set cylinders by rejection sampling
    grid_size = 2 * arena_size / 3
    # arena_size = 1.2
    # cylinder_size = 0.4
    matrix_size = 3
    max_z = max_height / 2.0 # z = 0.0 or max_z
    origin_grid = [1, 1]
    occupancy_matrix = np.zeros((matrix_size, matrix_size))
    path_occupancy_matrix = np.zeros((matrix_size, matrix_size))
    cylinder_occupancy_matrix = np.zeros((matrix_size, matrix_size))
    
    # first randomize the grid pos of cylinders
    # occupy the matrix
    x_y_grid_list = [[0, 1], [1, 1], [1, 0], [1,2], [2, 1]]
    cylinders_pos = []
    for cylinder_idx in range(num_cylinders):
        while True:
            grid_idx = torch.randint(0, len(x_y_grid_list), (1,)).item()
            x_grid = int(x_y_grid_list[grid_idx][0])
            y_grid = int(x_y_grid_list[grid_idx][1])
            
            x = (x_grid - origin_grid[0]) * grid_size
            y = (y_grid - origin_grid[1]) * grid_size
            
            z = torch.randint(0, 2, (1,)).item() # 0 means low, 1 means high

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < matrix_size and y_grid >= 0 and y_grid < matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    cylinders_pos.append(torch.tensor([x, y, z * max_z], device=device))
                    path_occupancy_matrix[x_grid, y_grid] = z
                    occupancy_matrix[x_grid, y_grid] = 1
                    break
    if num_cylinders == 0:
        cylinders_pos = torch.tensor([], device=device)
    else:
        cylinders_pos = torch.stack(cylinders_pos)
    
    small_expand = 4
    occupancy_matrix = np.kron(occupancy_matrix, np.ones((small_expand, small_expand), dtype=occupancy_matrix.dtype))
    path_occupancy_matrix = np.kron(path_occupancy_matrix, np.ones((small_expand, small_expand), dtype=path_occupancy_matrix.dtype))
    # set disabled idx to 1.0
    # up
    occupancy_matrix[0,0] = 1.0; occupancy_matrix[0,1] = 1.0; occupancy_matrix[0,2] = 1.0; occupancy_matrix[0,3] = 1.0; occupancy_matrix[0,4] = 1.0
    occupancy_matrix[0,-1] = 1.0; occupancy_matrix[0,-2] = 1.0; occupancy_matrix[0,-3] = 1.0; occupancy_matrix[0,-4] = 1.0; occupancy_matrix[0,-5] = 1.0
    occupancy_matrix[1,0] = 1.0; occupancy_matrix[1,1] = 1.0; occupancy_matrix[1,2] = 1.0
    occupancy_matrix[1,-1] = 1.0; occupancy_matrix[1,-2] = 1.0; occupancy_matrix[1,-3] = 1.0
    occupancy_matrix[2,0] = 1.0; occupancy_matrix[2,1] = 1.0
    occupancy_matrix[2,-1] = 1.0; occupancy_matrix[2,-2] = 1.0
    occupancy_matrix[3,0] = 1.0; occupancy_matrix[3,-1] = 1.0
    occupancy_matrix[4,0] = 1.0; occupancy_matrix[4,-1] = 1.0

    path_occupancy_matrix[0,0] = 1.0; path_occupancy_matrix[0,1] = 1.0; path_occupancy_matrix[0,2] = 1.0; path_occupancy_matrix[0,3] = 1.0; path_occupancy_matrix[0,4] = 1.0
    path_occupancy_matrix[0,-1] = 1.0; path_occupancy_matrix[0,-2] = 1.0; path_occupancy_matrix[0,-3] = 1.0; path_occupancy_matrix[0,-4] = 1.0; path_occupancy_matrix[0,-5] = 1.0
    path_occupancy_matrix[1,0] = 1.0; path_occupancy_matrix[1,1] = 1.0; path_occupancy_matrix[1,2] = 1.0
    path_occupancy_matrix[1,-1] = 1.0; path_occupancy_matrix[1,-2] = 1.0; path_occupancy_matrix[1,-3] = 1.0
    path_occupancy_matrix[2,0] = 1.0; path_occupancy_matrix[2,1] = 1.0
    path_occupancy_matrix[2,-1] = 1.0; path_occupancy_matrix[2,-2] = 1.0
    path_occupancy_matrix[3,0] = 1.0; path_occupancy_matrix[3,-1] = 1.0
    path_occupancy_matrix[4,0] = 1.0; path_occupancy_matrix[4,-1] = 1.0
    # down
    occupancy_matrix[-1,0] = 1.0; occupancy_matrix[-1,1] = 1.0; occupancy_matrix[-1,2] = 1.0; occupancy_matrix[-1,3] = 1.0; occupancy_matrix[-1,4] = 1.0
    occupancy_matrix[-1,-1] = 1.0; occupancy_matrix[-1,-2] = 1.0; occupancy_matrix[-1,-3] = 1.0; occupancy_matrix[-1,-4] = 1.0; occupancy_matrix[-1,-5] = 1.0
    occupancy_matrix[-2,0] = 1.0; occupancy_matrix[-2,1] = 1.0; occupancy_matrix[-2,2] = 1.0
    occupancy_matrix[-2,-1] = 1.0; occupancy_matrix[-2,-2] = 1.0; occupancy_matrix[-2,-3] = 1.0
    occupancy_matrix[-3,0] = 1.0; occupancy_matrix[-3,1] = 1.0
    occupancy_matrix[-3,-1] = 1.0; occupancy_matrix[-3,-2] = 1.0
    occupancy_matrix[-4,0] = 1.0; occupancy_matrix[-4,-1] = 1.0
    occupancy_matrix[-5,0] = 1.0; occupancy_matrix[-5,-1] = 1.0

    path_occupancy_matrix[-1,0] = 1.0; path_occupancy_matrix[-1,1] = 1.0; path_occupancy_matrix[-1,2] = 1.0; path_occupancy_matrix[-1,3] = 1.0; path_occupancy_matrix[-1,4] = 1.0
    path_occupancy_matrix[-1,-1] = 1.0; path_occupancy_matrix[-1,-2] = 1.0; path_occupancy_matrix[-1,-3] = 1.0; path_occupancy_matrix[-1,-4] = 1.0; path_occupancy_matrix[-1,-5] = 1.0
    path_occupancy_matrix[-2,0] = 1.0; path_occupancy_matrix[-2,1] = 1.0; path_occupancy_matrix[-2,2] = 1.0
    path_occupancy_matrix[-2,-1] = 1.0; path_occupancy_matrix[-2,-2] = 1.0; path_occupancy_matrix[-2,-3] = 1.0
    path_occupancy_matrix[-3,0] = 1.0; path_occupancy_matrix[-3,1] = 1.0
    path_occupancy_matrix[-3,-1] = 1.0; path_occupancy_matrix[-3,-2] = 1.0
    path_occupancy_matrix[-4,0] = 1.0; path_occupancy_matrix[-4,-1] = 1.0
    path_occupancy_matrix[-5,0] = 1.0; path_occupancy_matrix[-5,-1] = 1.0

    cylinder_occupancy_matrix = occupancy_matrix.copy()
    path_cylinder_occupancy_matrix = path_occupancy_matrix.copy()

    drone_target_pos = []
    start_grid = ()
    target_grid = ()
    small_reference_grid = [0, 0]
    small_grid_size = grid_size / small_expand
    small_reference_pos = [-arena_size + small_grid_size / 2, -arena_size + small_grid_size / 2]
    small_matrix_size = len(occupancy_matrix)
    
    for idx in range(num_drones + 1):
        while True:
            grid_idx = torch.randint(0, len(occupancy_matrix), (2,))
            x_grid = int(grid_idx[0])
            y_grid = int(grid_idx[1])
            
            x = (x_grid - small_reference_grid[0]) * small_grid_size + small_reference_pos[0]
            y = (y_grid - small_reference_grid[1]) * small_grid_size + small_reference_pos[1]
            z = D.Uniform(
                torch.tensor([0.1]),
                torch.tensor([max_height])
            ).sample()

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < small_matrix_size and y_grid >= 0 and y_grid < small_matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    drone_target_pos.append(torch.tensor([x, y, z], device=device))
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
        cylinder_occupancy_matrix, path_cylinder_occupancy_matrix, start_grid, target_grid

# cl_bound: generate drone and target in [-bound, bound], large scene
# cl_bound: 3~6
def generate_cylinder_large(arena_size, max_height, cylinder_size, num_cylinders, device):
    # set cylinders by rejection sampling
    grid_size = 2 * arena_size / 3
    # arena_size = 1.2
    # cylinder_size = 0.4
    matrix_size = 3
    max_z = max_height / 2.0 # z = 0.0 or max_z
    origin_grid = [1, 1]
    occupancy_matrix = np.zeros((matrix_size, matrix_size))
    path_occupancy_matrix = np.zeros((matrix_size, matrix_size))
    
    # first randomize the grid pos of cylinders
    # occupy the matrix
    x_y_grid_list = [[0, 1], [1, 1], [1, 0], [1,2], [2, 1]]
    cylinders_pos = []
    for cylinder_idx in range(num_cylinders):
        while True:
            grid_idx = torch.randint(0, len(x_y_grid_list), (1,)).item()
            x_grid = int(x_y_grid_list[grid_idx][0])
            y_grid = int(x_y_grid_list[grid_idx][1])
            
            x = (x_grid - origin_grid[0]) * grid_size
            y = (y_grid - origin_grid[1]) * grid_size
            
            z = torch.randint(0, 2, (1,)).item() # 0 means low, 1 means high

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < matrix_size and y_grid >= 0 and y_grid < matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    cylinders_pos.append(torch.tensor([x, y, z * max_z], device=device))
                    path_occupancy_matrix[x_grid, y_grid] = z
                    occupancy_matrix[x_grid, y_grid] = 1
                    break
    if num_cylinders == 0:
        cylinders_pos = torch.tensor([], device=device)
    else:
        cylinders_pos = torch.stack(cylinders_pos)
    
    return cylinders_pos, occupancy_matrix, path_occupancy_matrix

def generate_drone_target_large_after_cylinder(arena_size, max_height, num_drones, device, occupancy_matrix, path_occupancy_matrix, cl_bound=6, height_bound=0.5):
    # set cylinders by rejection sampling
    grid_size = 2 * arena_size / 3
    
    # expand to 12 * 12, and set corner to 1.0
    small_expand = 4
    occupancy_matrix = np.kron(occupancy_matrix, np.ones((small_expand, small_expand), dtype=occupancy_matrix.dtype))
    path_occupancy_matrix = np.kron(path_occupancy_matrix, np.ones((small_expand, small_expand), dtype=path_occupancy_matrix.dtype))
    # set disabled idx to 1.0
    # up
    occupancy_matrix[0,0] = 1.0; occupancy_matrix[0,1] = 1.0; occupancy_matrix[0,2] = 1.0; occupancy_matrix[0,3] = 1.0; occupancy_matrix[0,4] = 1.0
    occupancy_matrix[0,-1] = 1.0; occupancy_matrix[0,-2] = 1.0; occupancy_matrix[0,-3] = 1.0; occupancy_matrix[0,-4] = 1.0; occupancy_matrix[0,-5] = 1.0
    occupancy_matrix[1,0] = 1.0; occupancy_matrix[1,1] = 1.0; occupancy_matrix[1,2] = 1.0
    occupancy_matrix[1,-1] = 1.0; occupancy_matrix[1,-2] = 1.0; occupancy_matrix[1,-3] = 1.0
    occupancy_matrix[2,0] = 1.0; occupancy_matrix[2,1] = 1.0
    occupancy_matrix[2,-1] = 1.0; occupancy_matrix[2,-2] = 1.0
    occupancy_matrix[3,0] = 1.0; occupancy_matrix[3,-1] = 1.0
    occupancy_matrix[4,0] = 1.0; occupancy_matrix[4,-1] = 1.0

    path_occupancy_matrix[0,0] = 1.0; path_occupancy_matrix[0,1] = 1.0; path_occupancy_matrix[0,2] = 1.0; path_occupancy_matrix[0,3] = 1.0; path_occupancy_matrix[0,4] = 1.0
    path_occupancy_matrix[0,-1] = 1.0; path_occupancy_matrix[0,-2] = 1.0; path_occupancy_matrix[0,-3] = 1.0; path_occupancy_matrix[0,-4] = 1.0; path_occupancy_matrix[0,-5] = 1.0
    path_occupancy_matrix[1,0] = 1.0; path_occupancy_matrix[1,1] = 1.0; path_occupancy_matrix[1,2] = 1.0
    path_occupancy_matrix[1,-1] = 1.0; path_occupancy_matrix[1,-2] = 1.0; path_occupancy_matrix[1,-3] = 1.0
    path_occupancy_matrix[2,0] = 1.0; path_occupancy_matrix[2,1] = 1.0
    path_occupancy_matrix[2,-1] = 1.0; path_occupancy_matrix[2,-2] = 1.0
    path_occupancy_matrix[3,0] = 1.0; path_occupancy_matrix[3,-1] = 1.0
    path_occupancy_matrix[4,0] = 1.0; path_occupancy_matrix[4,-1] = 1.0
    # down
    occupancy_matrix[-1,0] = 1.0; occupancy_matrix[-1,1] = 1.0; occupancy_matrix[-1,2] = 1.0; occupancy_matrix[-1,3] = 1.0; occupancy_matrix[-1,4] = 1.0
    occupancy_matrix[-1,-1] = 1.0; occupancy_matrix[-1,-2] = 1.0; occupancy_matrix[-1,-3] = 1.0; occupancy_matrix[-1,-4] = 1.0; occupancy_matrix[-1,-5] = 1.0
    occupancy_matrix[-2,0] = 1.0; occupancy_matrix[-2,1] = 1.0; occupancy_matrix[-2,2] = 1.0
    occupancy_matrix[-2,-1] = 1.0; occupancy_matrix[-2,-2] = 1.0; occupancy_matrix[-2,-3] = 1.0
    occupancy_matrix[-3,0] = 1.0; occupancy_matrix[-3,1] = 1.0
    occupancy_matrix[-3,-1] = 1.0; occupancy_matrix[-3,-2] = 1.0
    occupancy_matrix[-4,0] = 1.0; occupancy_matrix[-4,-1] = 1.0
    occupancy_matrix[-5,0] = 1.0; occupancy_matrix[-5,-1] = 1.0

    path_occupancy_matrix[-1,0] = 1.0; path_occupancy_matrix[-1,1] = 1.0; path_occupancy_matrix[-1,2] = 1.0; path_occupancy_matrix[-1,3] = 1.0; path_occupancy_matrix[-1,4] = 1.0
    path_occupancy_matrix[-1,-1] = 1.0; path_occupancy_matrix[-1,-2] = 1.0; path_occupancy_matrix[-1,-3] = 1.0; path_occupancy_matrix[-1,-4] = 1.0; path_occupancy_matrix[-1,-5] = 1.0
    path_occupancy_matrix[-2,0] = 1.0; path_occupancy_matrix[-2,1] = 1.0; path_occupancy_matrix[-2,2] = 1.0
    path_occupancy_matrix[-2,-1] = 1.0; path_occupancy_matrix[-2,-2] = 1.0; path_occupancy_matrix[-2,-3] = 1.0
    path_occupancy_matrix[-3,0] = 1.0; path_occupancy_matrix[-3,1] = 1.0
    path_occupancy_matrix[-3,-1] = 1.0; path_occupancy_matrix[-3,-2] = 1.0
    path_occupancy_matrix[-4,0] = 1.0; path_occupancy_matrix[-4,-1] = 1.0
    path_occupancy_matrix[-5,0] = 1.0; path_occupancy_matrix[-5,-1] = 1.0

    cylinder_occupancy_matrix = occupancy_matrix.copy()

    drone_target_pos = []
    start_grid = ()
    target_grid = ()
    small_reference_grid = [0, 0]
    small_grid_size = grid_size / small_expand
    small_reference_pos = [-arena_size + small_grid_size / 2, -arena_size + small_grid_size / 2]
    small_matrix_size = len(occupancy_matrix)
    
    # cl_bound
    for idx in range(num_drones + 1):
        while True:
            # x_grid: [6 - cl_bound, 5 + cl_bound]
            # y_grid: [6 - cl_bound, 5 + cl_bound]
            grid_idx = torch.randint(6 - cl_bound, 6 + cl_bound, (2,))
            # grid_idx = torch.randint(0, len(occupancy_matrix), (2,))
            x_grid = int(grid_idx[0])
            y_grid = int(grid_idx[1])
            
            x = (x_grid - small_reference_grid[0]) * small_grid_size + small_reference_pos[0]
            y = (y_grid - small_reference_grid[1]) * small_grid_size + small_reference_pos[1]
            z = D.Uniform(
                torch.tensor([0.5 * max_height - height_bound * max_height + 0.02]),
                torch.tensor([0.5 * max_height + height_bound * max_height - 0.02])
            ).sample()

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < small_matrix_size and y_grid >= 0 and y_grid < small_matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    drone_target_pos.append(torch.tensor([x, y, z], device=device))
                    occupancy_matrix[x_grid, y_grid] = 1
                    if idx >= num_drones: # target
                        target_grid += (x_grid, y_grid)
                    else: # drones
                        start_grid += ((x_grid, y_grid), )
                    break
    drone_target_pos = torch.stack(drone_target_pos)
    drone_target_occupancy_matrix = occupancy_matrix - cylinder_occupancy_matrix
    
    return drone_target_pos, occupancy_matrix, cylinder_occupancy_matrix, drone_target_occupancy_matrix, \
            path_occupancy_matrix, start_grid, target_grid

def rejection_sampling_all_obj_large_cylinder_cl(arena_size, max_height, cylinder_size, num_drones, num_cylinders, device, cl_bound=6, height_bound=0.5):
    while True:
        cylinders_pos, occupancy_matrix, path_occupancy_matrix = generate_cylinder_large(arena_size, max_height, cylinder_size, num_cylinders, device)
        
        drone_target_pos, occupancy_matrix, cylinder_occupancy_matrix, drone_target_occupancy_matrix, \
            path_cylinder_occupancy_matrix, start_grid, target_grid = generate_drone_target_large_after_cylinder(arena_size, max_height, num_drones, device, occupancy_matrix, path_occupancy_matrix, cl_bound=cl_bound, height_bound=height_bound)
        
        if drone_target_pos.shape[0] == (num_drones + 1):
            break
    
    objects_pos = torch.concat([drone_target_pos, cylinders_pos])

    return objects_pos, occupancy_matrix, drone_target_occupancy_matrix, \
        cylinder_occupancy_matrix, path_cylinder_occupancy_matrix, start_grid, target_grid

# add validation
def rejection_sampling_with_validation_large_cylinder(arena_size, max_height, cylinder_size, num_drones, num_cylinders, device, use_validation=True):
    if not use_validation:
        task_one, occupancy_matrix, drone_target_occupancy_matrix, \
            cylinder_occupancy_matrix, path_cylinder_occupancy_matrix, start_grid, target_grid = \
            rejection_sampling_all_obj_large_cylinder(arena_size=arena_size,
                                                    max_height=max_height,
                                                    cylinder_size=cylinder_size, 
                                                    num_drones=num_drones, 
                                                    num_cylinders=num_cylinders, 
                                                    device=device)
        return task_one, occupancy_matrix, drone_target_occupancy_matrix, cylinder_occupancy_matrix
    else:
        num_loop = 0
        while(num_loop <= 100):
            task_one, occupancy_matrix, drone_target_occupancy_matrix, \
                cylinder_occupancy_matrix, path_cylinder_occupancy_matrix, start_grid, target_grid = \
                rejection_sampling_all_obj_large_cylinder(arena_size=arena_size, 
                                                    max_height=max_height,
                                                    cylinder_size=cylinder_size, 
                                                    num_drones=num_drones, 
                                                    num_cylinders=num_cylinders, 
                                                    device=device)
            if has_feasible_path(path_cylinder_occupancy_matrix, start_grid, target_grid):
                return task_one, occupancy_matrix, drone_target_occupancy_matrix, cylinder_occupancy_matrix
            num_loop += 1
        raise NotImplementedError

def rejection_sampling_with_validation_large_cylinder_cl(arena_size, max_height, cylinder_size, num_drones, num_cylinders, device, use_validation=True, cl_bound=6, height_bound=0.5):
    if not use_validation:
        task_one, occupancy_matrix, drone_target_occupancy_matrix, \
            cylinder_occupancy_matrix, path_cylinder_occupancy_matrix, start_grid, target_grid = \
            rejection_sampling_all_obj_large_cylinder_cl(arena_size=arena_size,
                                                max_height=max_height, 
                                                cylinder_size=cylinder_size, 
                                                num_drones=num_drones, 
                                                num_cylinders=num_cylinders, 
                                                device=device,
                                                cl_bound=cl_bound,
                                                height_bound=height_bound)
        return task_one, occupancy_matrix, drone_target_occupancy_matrix, cylinder_occupancy_matrix
    else:
        num_loop = 0
        while(num_loop <= 100):
            task_one, occupancy_matrix, drone_target_occupancy_matrix, \
                cylinder_occupancy_matrix, path_cylinder_occupancy_matrix, start_grid, target_grid = \
                rejection_sampling_all_obj_large_cylinder_cl(arena_size=arena_size, 
                                                    max_height=max_height,
                                                    cylinder_size=cylinder_size, 
                                                    num_drones=num_drones, 
                                                    num_cylinders=num_cylinders, 
                                                    device=device,
                                                    cl_bound=cl_bound,
                                                    height_bound=height_bound)
            if has_feasible_path(path_cylinder_occupancy_matrix, start_grid, target_grid):
                return task_one, occupancy_matrix, drone_target_occupancy_matrix, cylinder_occupancy_matrix
            num_loop += 1
        raise NotImplementedError

###################################################

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

def plot_circle_square():
    # 创建图形
    fig, ax = plt.subplots()

    # 创建一个12x12的正方形
    square = plt.Rectangle((0, 0), 12, 12, color='lightgray', fill=True)

    # 创建一个内切圆
    circle = plt.Circle((6, 6), 6, color='red', fill=False)

    # 画内部的小格子
    for i in range(1, 12):
        ax.plot([i, i], [0, 12], color='black', linewidth=0.5)
        ax.plot([0, 12], [i, i], color='black', linewidth=0.5)

    # 画横向的线
    for i in range(1, 12):
        for j in range(1, 12):
            ax.plot([j - 0.5, j + 0.5], [i, i], color='black', linewidth=0.5)

    # 画纵向的线
    for i in range(1, 12):
        for j in range(1, 12):
            ax.plot([i, i], [j - 0.5, j + 0.5], color='black', linewidth=0.5)

    # 添加正方形和圆形到图形中
    ax.add_patch(square)
    ax.add_patch(circle)

    # 设置坐标轴范围
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)

    # 设置坐标轴标签
    plt.xlabel('X')
    plt.ylabel('Y')

    # 设置标题
    plt.title('Square with Inscribed Circle')

    # 显示图形
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('tmp.png')

def check_path():
    task_list = []
    fessible_list = []
    torch.manual_seed(1)
    num_cylinders = 3
    check_occupy = []
    for _ in range(2000):
        task_one, occupancy_matrix, drone_target_occupancy_matrix, \
            cylinder_occupancy_matrix, path_cylinder_occupancy_matrix, start_grid, target_grid = \
            rejection_sampling_all_obj_large_cylinder(arena_size=1.2, 
                                                 cylinder_size=0.4, 
                                                 num_drones=4, 
                                                 num_cylinders=num_cylinders, 
                                                 device='cpu')
        task_list.append(task_one.numpy().reshape(-1))
        fessible_list.append(has_feasible_path(path_cylinder_occupancy_matrix, start_grid, target_grid))
        check_occupy.append(cylinder_occupancy_matrix)
    task_list = np.array(task_list)
    print('num_fessible', np.sum(fessible_list))

def check_dist(): # check drone, target and cylinder dist
    # task_list = np.load('/home/chenjy/OmniDrones/scripts/outputs/Disagreement_emptytransfer_0and1/03-04_18-08/wandb/run-20240304_180822-3e0txp70/files/tasks/tasks_2313.npy')
    # weights_list = np.load('/home/chenjy/OmniDrones/scripts/outputs/Disagreement_emptytransfer_0and1/03-04_18-08/wandb/run-20240304_180822-3e0txp70/files/tasks/weights_2313.npy')
    task_list = []
    arena_size = 0.6
    grid_size = 2 * arena_size / 3
    small_grid_size = grid_size / 4
    cylinder_occupancy = np.zeros((12, 12))
    drone_target_occupancy = np.zeros((12, 12))
    for _ in range(10000):
        task_one, occupancy_matrix, drone_target_occupancy_matrix, \
            cylinder_occupancy_matrix = \
            rejection_sampling_with_validation_large_cylinder_cl(arena_size=arena_size, 
                                                 cylinder_size=0.3, 
                                                 num_drones=4, 
                                                 num_cylinders=2, 
                                                 device='cpu',
                                                 cl_bound=6)
        task_list.append(task_one.numpy().reshape(-1))
        cylinder_occupancy += cylinder_occupancy_matrix
        drone_target_occupancy += drone_target_occupancy_matrix
    task_list = np.array(task_list)
    drone_pos = task_list[:, :12]
    target_pos = task_list[:, 12:15]
    cylinder_pos = task_list[:, 15:]
    
    get_occupation_matrix(drone_pos[:, :3], arena_size=arena_size, matrix_size=12, grid_size=small_grid_size, name='drone0')
    get_occupation_matrix(drone_pos[:, 3:6], arena_size=arena_size, matrix_size=12, grid_size=small_grid_size, name='drone1')
    get_occupation_matrix(drone_pos[:, 6:9], arena_size=arena_size, matrix_size=12, grid_size=small_grid_size, name='drone2')
    get_occupation_matrix(drone_pos[:, 9:12], arena_size=arena_size, matrix_size=12, grid_size=small_grid_size, name='drone3')
    get_occupation_matrix(target_pos[:, :3], arena_size=arena_size, matrix_size=12, grid_size=small_grid_size, name='target')
    get_occupation_matrix(cylinder_pos[:, :3], arena_size=arena_size, matrix_size=3, grid_size=grid_size, name='cylinder1')
    # get_occupation_matrix(cylinder_pos[:, 3:6], arena_size=1.2, matrix_size=3, grid_size=0.8, name='cylinder2')
    # plot_heatmap(drone_target_occupancy, 'drone_target_check')
    # plot_heatmap(cylinder_occupancy, 'cylinder_check')

if __name__ == '__main__':
    # plot_circle_square()
    # check_path()
    check_dist()
    # check = generate_outside_cylinders_x_y(arena_size=1.0, 
    #                                        num_envs=10, 
    #                                        device='cpu')