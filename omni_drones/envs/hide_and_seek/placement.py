import numpy as np
import torch
import torch.distributions as D
import matplotlib.pyplot as plt
import copy

def get_occupation_matrix(position_list, name):
    matrix_size = 5
    occupy = np.zeros((matrix_size, matrix_size))
    grid_size = 0.4
    origin_grid = [int(matrix_size / 2), int(matrix_size / 2)]
    origin_pos = [0.0, 0.0] # left corner
    for pos in position_list:
        x_grid = round((pos[0] - origin_pos[0]) / grid_size) + origin_grid[0]
        y_grid = round((pos[1] - origin_pos[1]) / grid_size) + origin_grid[1]
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
    cylinder_occupancy_matrix = occupancy_matrix.copy()

    drone_target_pos = []
    extra_size = 0.15
    for idx in range(num_drones + 1):
        while True:
            # Generate random angle and radius within the circular area
            grid_idx = torch.randint(0, len(x_y_grid_list), (1,)).item()
            x_grid = int(x_y_grid_list[grid_idx][0])
            y_grid = int(x_y_grid_list[grid_idx][1])
            
            x = (x_grid - origin_grid[0]) * grid_size
            y = (y_grid - origin_grid[1]) * grid_size

            # Check if the new object overlaps with existing objects
            if x_grid >= 0 and x_grid < matrix_size and y_grid >= 0 and y_grid < matrix_size:
                if occupancy_matrix[x_grid, y_grid] == 0:
                    extra_x = (2.0 * torch.rand(1)[0] - 1.0) * extra_size
                    extra_y = (2.0 * torch.rand(1)[0] - 1.0) * extra_size
                    real_x = x + extra_x
                    real_y = y + extra_y
                    drone_target_pos.append(torch.tensor([real_x, real_y], device=device))
                    occupancy_matrix[x_grid, y_grid] = 1
                    if idx == 0:
                        drone_target_occupancy_matrix = occupancy_matrix - cylinder_occupancy_matrix
                    break
    drone_target_pos = torch.stack(drone_target_pos)
    objects_pos = torch.concat([drone_target_pos, cylinders_pos])
    # drone_target_occupancy_matrix = occupancy_matrix - cylinder_occupancy_matrix
    
    return objects_pos, drone_target_occupancy_matrix

# only for empty scenario
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
        torch.tensor([arena_size], device=device)
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

def main():
    # task_list = np.load('/home/chenjy/OmniDrones/scripts/outputs/Disagreement_emptytransfer_0and1/03-04_18-08/wandb/run-20240304_180822-3e0txp70/files/tasks/tasks_2313.npy')
    # weights_list = np.load('/home/chenjy/OmniDrones/scripts/outputs/Disagreement_emptytransfer_0and1/03-04_18-08/wandb/run-20240304_180822-3e0txp70/files/tasks/weights_2313.npy')
    task_list = []
    # cylinder_occupancy = np.zeros((5, 5))
    drone_target_occupancy = np.zeros((5, 5))
    for _ in range(10000):
        task_one, drone_target_occupancy_matrix = rejection_sampling_all_obj_xy(arena_size=1.0, 
                                                 cylinder_size=0.2, 
                                                 num_drones=4, 
                                                 num_cylinders=2, 
                                                 device='cpu')
        task_list.append(task_one.numpy().reshape(-1))
        drone_target_occupancy += drone_target_occupancy_matrix
        # cylinder_occupancy += cylinder_occupancy_matrix
    task_list = np.array(task_list)
    drone_pos = task_list[:, :8]
    target_pos = task_list[:, 8:10]
    cylinder_pos = task_list[:, 10:]
    
    get_occupation_matrix(drone_pos[:, :2], 'drone0')
    get_occupation_matrix(drone_pos[:, 2:4], 'drone1')
    get_occupation_matrix(drone_pos[:, 4:6], 'drone2')
    get_occupation_matrix(drone_pos[:, 6:8], 'drone3')
    get_occupation_matrix(target_pos[:, :2], 'target')
    get_occupation_matrix(cylinder_pos[:, :2], 'cylinder1')
    get_occupation_matrix(cylinder_pos[:, 2:4], 'cylinder2')
    # plot_heatmap(drone_target_occupancy, 'drone0_check')
    # plot_heatmap(cylinder_occupancy, 'cylinder1_check')
    
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
    main()