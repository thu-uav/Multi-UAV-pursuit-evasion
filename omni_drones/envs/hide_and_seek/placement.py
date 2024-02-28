import numpy as np
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import copy

def plot_heatmap(matrix):
    # matrix [n, m]
    plt.imshow(matrix, cmap='viridis', aspect='auto', origin='lower')
    plt.colorbar()
    plt.title('Heatmap')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig('tmp.png')

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

# def rejection_sampling_all_obj_xy(arena_size, cylinder_size, num_drones, num_cylinders, device):
#     # set cylinders by rejection sampling
#     grid_size = 2 * cylinder_size
#     matrix_size = int(2 * arena_size / grid_size)
#     origin_grid = [int(matrix_size / 2), int(matrix_size / 2)]
#     origin_pos = [-arena_size, +arena_size] # left corner
#     occupancy_matrix = np.zeros((matrix_size, matrix_size))
#     cylinder_occupancy_matrix = np.zeros((matrix_size, matrix_size))
    
#     # first randomize the grid pos of cylinders
#     # occupy the matrix
#     cylinders_pos = []
#     for cylinder_idx in range(num_cylinders):
#         while True:
#             x_grid = torch.randint(0, matrix_size, (1,))
#             y_grid = torch.randint(0, matrix_size, (1,))
            
#             x = (x_grid - origin_grid[0]) * grid_size
#             y = (y_grid - origin_grid[1]) * grid_size

#             # Check if the new object overlaps with existing objects
#             if x_grid >= 0 and x_grid < matrix_size and y_grid >= 0 and y_grid < matrix_size:
#                 if occupancy_matrix[x_grid, y_grid] == 0:
#                     cylinders_pos.append(torch.tensor([x, y], device=device))
#                     occupancy_matrix[x_grid, y_grid] = 1
#                     break
#     cylinders_pos = torch.stack(cylinders_pos)
#     cylinder_occupancy_matrix = copy.deepcopy(occupancy_matrix)
    
#     # pos dist
#     angle_dist = D.Uniform(
#         torch.tensor([0.0], device=device),
#         torch.tensor([2 * torch.pi], device=device)
#     )
#     r_dist = D.Uniform(
#         torch.tensor([0.0], device=device),
#         torch.tensor([arena_size - 0.2], device=device)
#     )
#     drone_target_pos = []
#     for _ in range(num_drones + 1):
#         while True:
#             # Generate random angle and radius within the circular area
#             angle = angle_dist.sample()
#             r = r_dist.sample()

#             # Convert polar coordinates to Cartesian coordinates
#             x = r * torch.cos(angle)
#             y = r * torch.sin(angle)

#             # Convert coordinates to grid units
#             x_grid = int((x - origin_pos[0]) / grid_size)
#             y_grid = int((origin_pos[1] - y) / grid_size)

#             # Check if the new object overlaps with existing objects
#             if x_grid >= 0 and x_grid < matrix_size and y_grid >= 0 and y_grid < matrix_size:
#                 if occupancy_matrix[x_grid, y_grid] == 0:
#                     drone_target_pos.append(torch.tensor([x, y], device=device))
#                     occupancy_matrix[x_grid, y_grid] = 1
#                     break
#     drone_target_pos = torch.stack(drone_target_pos)
#     objects_pos = torch.concat([drone_target_pos, cylinders_pos])
    
#     return objects_pos, occupancy_matrix

def rejection_sampling_all_obj_xy(arena_size, cylinder_size, num_drones, num_cylinders, device):
    # set cylinders by rejection sampling
    grid_size = 2 * cylinder_size
    matrix_size = int(2 * arena_size / grid_size)
    origin_grid = [int(matrix_size / 2), int(matrix_size / 2)]
    origin_pos = [-arena_size, +arena_size] # left corner
    occupancy_matrix = np.zeros((matrix_size, matrix_size))
    cylinder_occupancy_matrix = np.zeros((matrix_size, matrix_size))
    
    # first randomize the grid pos of cylinders
    # occupy the matrix
    cylinders_pos = []
    for cylinder_idx in range(num_cylinders):
        while True:
            x_grid = torch.randint(0, matrix_size, (1,))
            y_grid = torch.randint(0, matrix_size, (1,))
            
            x = (x_grid - origin_grid[0]) * grid_size
            y = (y_grid - origin_grid[1]) * grid_size

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < matrix_size and y_grid >= 0 and y_grid < matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    cylinders_pos.append(torch.tensor([x, y], device=device))
                    occupancy_matrix[x_grid, y_grid] = 1
                    break
    cylinders_pos = torch.stack(cylinders_pos)
    cylinder_occupancy_matrix = copy.deepcopy(occupancy_matrix)
    
    # pos dist
    angle_dist = D.Uniform(
        torch.tensor([0.0], device=device),
        torch.tensor([2 * torch.pi], device=device)
    )
    r_dist = D.Uniform(
        torch.tensor([0.0], device=device),
        torch.tensor([arena_size - 0.2], device=device)
    )
    drone_target_pos = []
    extra_size = 0.15
    for _ in range(num_drones + 1):
        while True:
            # Generate random angle and radius within the circular area
            angle = angle_dist.sample()
            r = r_dist.sample()

            # Convert polar coordinates to Cartesian coordinates
            x = r * torch.cos(angle)
            y = r * torch.sin(angle)

            # Convert coordinates to grid units
            x_grid = int((x - origin_pos[0]) / grid_size)
            y_grid = int((origin_pos[1] - y) / grid_size)
            
            current_grid_pos = [x_grid * grid_size + origin_pos[0], 
                                origin_pos[1] - y_grid * grid_size]

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < matrix_size and y_grid >= 0 and y_grid < matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    extra_x = (2.0 * torch.rand(1)[0] - 1.0) * extra_size
                    extra_y = (2.0 * torch.rand(1)[0] - 1.0) * extra_size
                    real_x = current_grid_pos[0] + extra_x
                    real_y = current_grid_pos[1] + extra_y
                    drone_target_pos.append(torch.tensor([real_x, real_y], device=device))
                    occupancy_matrix[x_grid, y_grid] = 1
                    break
    drone_target_pos = torch.stack(drone_target_pos)
    objects_pos = torch.concat([drone_target_pos, cylinders_pos])
    
    return objects_pos, occupancy_matrix

def rejection_sampling(arena_size, cylinder_size, num_cylinders, device):
    # set cylinders by rejection sampling
    grid_size = 2 * cylinder_size
    matrix_size = int(2 * arena_size / grid_size)
    origin_pos = [-arena_size, +arena_size] # left corner
    occupancy_matrix = np.zeros((matrix_size, matrix_size))
    # pos dist
    angle_dist = D.Uniform(
        torch.tensor([0.0], device=device),
        torch.tensor([2 * torch.pi], device=device)
    )
    r_dist = D.Uniform(
        torch.tensor([0.0], device=device),
        torch.tensor([arena_size - 0.2], device=device)
    )
    objects_pos = []
    for obj_idx in range(num_cylinders):
        while True:
            # Generate random angle and radius within the circular area
            angle = angle_dist.sample()
            r = r_dist.sample()

            # Convert polar coordinates to Cartesian coordinates
            x = r * torch.cos(angle)
            y = r * torch.sin(angle)

            # Convert coordinates to grid units
            x_grid = int((x - origin_pos[0]) / grid_size)
            y_grid = int((origin_pos[1] - y) / grid_size)

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < matrix_size and y_grid >= 0 and y_grid < matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    objects_pos.append(torch.tensor([x, y, 1.0], device=device))
                    occupancy_matrix[x_grid, y_grid] = 1
                    break

    objects_pos = torch.stack(objects_pos)
    return objects_pos