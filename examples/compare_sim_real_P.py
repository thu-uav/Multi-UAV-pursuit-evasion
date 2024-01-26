import os

from typing import Dict, Optional
import torch
from functorch import vmap
import torch.optim as optim
from scipy import optimize
import time

import hydra
from omegaconf import OmegaConf
from omni_drones import init_simulation_app
from tensordict import TensorDict
import pandas as pd
import pdb
import numpy as np
import yaml
from skopt import Optimizer
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse
import matplotlib.pyplot as plt
import numpy as np

rosbags = [
    # '/home/jiayu/OmniDrones/realdata/crazyflie/cf1_figure8.csv',
    # '/home/jiayu/OmniDrones/realdata/crazyflie/cf7_figure8.csv',
    # '/home/jiayu/OmniDrones/realdata/crazyflie/cf9_figure8.csv',
    # '/home/jiayu/OmniDrones/realdata/crazyflie/cf7_hover1.csv',
    # '/home/jiayu/OmniDrones/realdata/crazyflie/cf7_hover2.csv',
    # '/home/jiayu/OmniDrones/realdata/crazyflie/cf9_hover1.csv',
    '/home/jiayu/OmniDrones/realdata/crazyflie/cf9_figure8.csv'
]

@hydra.main(version_base=None, config_path=".", config_name="real2sim")
def main(cfg):
    """
        preprocess real data
        real_data: [batch_size, T, dimension]
    """
    df = pd.read_csv(rosbags[0], skip_blank_lines=True)
    df = np.array(df)
    use_preprocess = True
    if use_preprocess:
        preprocess_df = []
        for df_one in df:
            if df_one[-1] > 0: # req motor thrust > 0 means fly
                preprocess_df.append(df_one)
        preprocess_df = np.array(preprocess_df)
    else:
        preprocess_df = df
    episode_len = preprocess_df.shape[0] # hover
    # episode_len = 1400 # TODO, apply to all trajectories
    real_data = []
    T = 1
    skip = 1
    for i in range(0, episode_len-T, skip):
        _slice = slice(i, i+T)
        real_data.append(preprocess_df[_slice])
    real_data = np.array(real_data)

    # start sim
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.controllers import RateController, PIDRateController
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.utils.torch import euler_to_quaternion, quaternion_to_euler
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg

    average_dt = 0.02
    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=average_dt,
        rendering_dt=average_dt,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    drone: MultirotorBase = MultirotorBase.REGISTRY[cfg.drone_model]()
    n = 1 # parrallel envs
    translations = torch.zeros(n, 3)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 0.5
    drone.spawn(translations=translations)
    scene_utils.design_scene()

    """
        evaluate in Omnidrones
        params: suggested params
        real_data: [batch_size, T, dimension]
        sim: omnidrones core
        drone: crazyflie
        controller: the predefined controller
    """
    # # origin
    # params = [
    #     0.03, 1.4e-5, 1.4e-5, 2.17e-5, 0.043,
    #     2.88e-8, 2315, 7.24e-10, 0.2, 
    #     # time const
    #     0.0178,
    #     # controller
    #     0.0052, 0.0052, 0.00025
    # ]
    
    # # opt for dynamics (kf, km, Tm)
    # params = [
    #     0.03, 1.4e-5, 1.4e-5, 2.17e-5, 0.043,
    #     2e-08, 2315, 7.24e-11, 0.2, 
    #     0.01,
    #     # controller
    #     0.0052, 0.0052, 0.00025
    # ]
    
    # opt for dynamics (kf, km, Tm) + controller (gain)
    params = [
        0.03, 1.4e-5, 1.4e-5, 2.17e-5, 0.043,
        2e-08, 2315, 7.24e-11, 0.2, 
        0.01,
        # controller
        0.00052, 0.00052, 2.5e-05
    ]
    
    tunable_parameters = {
        'mass': params[0],
        'inertia_xx': params[1],
        'inertia_yy': params[2],
        'inertia_zz': params[3],
        'arm_lengths': params[4],
        'force_constants': params[5],
        'max_rotation_velocities': params[6],
        'moment_constants': params[7],
        'drag_coef': params[8],
        'time_constant': params[9],
        'gain': params[10:]
    }
    
    # reset sim
    sim.reset()
    drone.initialize_byTunablePara(tunable_parameters=tunable_parameters)
    controller = RateController(9.81, drone.params).to(sim.device)
    # controller = PIDRateController(9.81, drone.params).to(sim.device)
    controller.set_byTunablePara(tunable_parameters=tunable_parameters)
    controller = controller.to(sim.device)
    
    max_thrust = controller.max_thrusts.sum(-1)

    def set_drone_state(pos, quat, vel, ang_vel):
        pos = pos.to(device=sim.device).float()
        quat = quat.to(device=sim.device).float()
        vel = vel.to(device=sim.device).float()
        ang_vel = ang_vel.to(device=sim.device).float()
        drone.set_world_poses(pos, quat)
        whole_vel = torch.cat([vel, ang_vel], dim=-1)
        drone.set_velocities(whole_vel)
        # flush the buffer so that the next getter invocation 
        # returns up-to-date values
        sim._physics_sim_view.flush() 
    
    '''
    df:
    Index(['pos.time', 'pos.x', 'pos.y', 'pos.z', (1:4)
        'quat.time', 'quat.w', 'quat.x','quat.y', 'quat.z', (5:9)
        'vel.time', 'vel.x', 'vel.y', 'vel.z', (10:13)
        'omega.time','omega.r', 'omega.p', 'omega.y', (14:17)
        'real_rate.time', 'real_rate.r', 'real_rate.p', 'real_rate.y', 
        'real_rate.thrust', (18:22)
        'target_rate.time','target_rate.r', 'target_rate.p', 'target_rate.y', 
        'target_rate.thrust',(23:27)
        'motor.time', 'motor.m1', 'motor.m2', 'motor.m3', 'motor.m4',(28:32)
        'motor.time', 'motor.m1req', 'motor.m2req', 'motor.m3req', 'motor.m4req',(33:37)
        ]
        dtype='object')
    '''
    sim_pos_list = []
    sim_quat_list = []
    sim_vel_list = []
    sim_body_rate_list = []
    sim_angvel_list = []
    sim_motor_thrust_ratio = []

    real_pos_list = []
    real_quat_list = []
    real_vel_list = []
    real_body_rate_list = []
    real_angvel_list = []
    real_motor_thrust_ratio = []
    
    target_body_rate_list = []
    
    # action
    sim_req_motor_thrust_ratio_list = []
    real_req_motor_thrust_ratio_list = []

    use_real_action = False
    trajectory_len = real_data.shape[0] - 1
    
    for i in range(trajectory_len):
        real_pos = torch.tensor(real_data[i, :, 1:4])
        real_quat = torch.tensor(real_data[i, :, 5:9])
        real_vel = torch.tensor(real_data[i, :, 10:13])
        real_rate = torch.tensor(real_data[i, :, 18:21])
        real_next_rate = torch.tensor(real_data[i + 1, :, 18:21])
        next_real_motor_thrust = torch.tensor(real_data[i + 1, :, 28:32]) / 2**16 # gamma_i**2, means f_i / f_max
        req_real_motor_thrust = torch.tensor(real_data[i, :, 33:37])
        real_rate[:, 1] = -real_rate[:, 1]
        real_next_rate[:, 1] = -real_next_rate[:, 1]
        # get angvel
        real_ang_vel = quat_rotate(real_quat, real_rate)
        set_drone_state(real_pos, real_quat, real_vel, real_ang_vel)
        # if i == 0:
            # set_drone_state(real_pos, real_quat, real_vel, real_ang_vel)
        
        # save
        real_pos_list.append(real_pos.numpy())
        real_quat_list.append(real_quat.numpy())
        real_vel_list.append(real_vel.numpy())
        real_body_rate_list.append(real_next_rate.numpy())
        real_angvel_list.append(real_ang_vel.numpy())
        real_motor_thrust_ratio.append(next_real_motor_thrust.numpy())

        drone_state = drone.get_state()[..., :13].reshape(-1, 13)
        # get current_rate
        pos, rot, linvel, angvel = drone_state.split([3, 4, 3, 3], dim=1)
        current_rate = quat_rotate_inverse(rot, angvel)
        target_thrust = torch.tensor(real_data[i, :, 26]).to(device=sim.device).float()
        target_rate = torch.tensor(real_data[i, :, 23:26]).to(device=sim.device).float()
        real_rate = real_rate.to(device=sim.device).float()
        real_next_rate = real_next_rate.to(device=sim.device).float()
        target_rate[:, 1] = -target_rate[:, 1]
        # current_rate[:, 1] = -current_rate[:, 1]
        action = controller.sim_step(
            current_rate=current_rate,
            target_rate=target_rate / 180 * torch.pi,
            # target_rate=real_next_rate,
            target_thrust=target_thrust.unsqueeze(1) / (2**16) * max_thrust
        )
        
        sim_req_motor_thrust_ratio_list.append(action.detach().to('cpu').numpy())
        
        real_action = req_real_motor_thrust.to(sim.device) / (2**16) * 2 - 1
        
        real_req_motor_thrust_ratio_list.append(real_action.detach().to('cpu').numpy())
        
        if use_real_action:
            # drone.apply_action(real_action)
            _, next_sim_motor_thrust, torques = drone.apply_action_foropt(real_action)
        else:
            # drone.apply_action(action)
            _, next_sim_motor_thrust, torques = drone.apply_action_foropt(action)
        sim.step(render=True)

        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue

        # get simulated drone state
        sim_state = drone.get_state().squeeze(0).cpu()
        sim_pos = sim_state[..., :3]
        sim_quat = sim_state[..., 3:7]
        sim_vel = sim_state[..., 7:10]
        sim_omega = sim_state[..., 10:13]
        next_body_rate = quat_rotate_inverse(sim_quat, sim_omega)

        sim_pos_list.append(sim_pos.cpu().detach().numpy())
        sim_quat_list.append(sim_quat.cpu().detach().numpy())
        sim_vel_list.append(sim_vel.cpu().detach().numpy())
        sim_body_rate_list.append(next_body_rate.cpu().detach().numpy())
        sim_angvel_list.append(sim_omega.cpu().detach().numpy())
        sim_motor_thrust_ratio.append((next_sim_motor_thrust / drone.KF).squeeze(1).cpu().detach().numpy())

        # get body_rate and thrust & compare
        target_body_rate = (target_rate / 180 * torch.pi).cpu()
        target_thrust = target_thrust.unsqueeze(1) / (2**16) * max_thrust
        target_body_rate_list.append(target_body_rate.detach().numpy())
    
    real_body_rate_list = np.array(real_body_rate_list)
    target_body_rate_list = np.array(target_body_rate_list)
    sim_body_rate_list = np.array(sim_body_rate_list)
    sim_req_motor_thrust_ratio_list = np.array(sim_req_motor_thrust_ratio_list)
    real_req_motor_thrust_ratio_list = np.array(real_req_motor_thrust_ratio_list)
    sim_motor_thrust_ratio = np.array(sim_motor_thrust_ratio)
    real_motor_thrust_ratio = np.array(real_motor_thrust_ratio)
        
    sim_pos_list = np.array(sim_pos_list)
    sim_quat_list = np.array(sim_quat_list)
    sim_vel_list = np.array(sim_vel_list)
    sim_body_rate_list = np.array(sim_body_rate_list)
    sim_angvel_list = np.array(sim_angvel_list)
    
    real_pos_list = np.array(real_pos_list)
    real_quat_list = np.array(real_quat_list)
    real_vel_list = np.array(real_vel_list)
    real_body_rate_list = np.array(real_body_rate_list)
    real_angvel_list = np.array(real_angvel_list)
    
    steps = np.arange(0, sim_pos_list.shape[0])
    
    # error
    if use_real_action:
        fig, axs = plt.subplots(8, 4, figsize=(20, 12))  # 8 * 4 
    else:
        fig, axs = plt.subplots(9, 4, figsize=(20, 12))  # 9 * 4 
    fig.subplots_adjust()
    # x error
    axs[0, 0].scatter(steps, sim_pos_list[:, 0, 0], s=5, c='red', label='sim')
    axs[0, 0].scatter(steps, real_pos_list[:, 0, 0], s=5, c='green', label='real')
    axs[0, 0].set_xlabel('steps')
    axs[0, 0].set_ylabel('m')
    axs[0, 0].set_title('sim/real_X')
    axs[0, 0].legend()
    # y error
    axs[0, 1].scatter(steps, sim_pos_list[:, 0, 1], s=5, c='red', label='sim')
    axs[0, 1].scatter(steps, real_pos_list[:, 0, 1], s=5, c='green', label='real')
    axs[0, 1].set_xlabel('steps')
    axs[0, 1].set_ylabel('m')
    axs[0, 1].set_title('sim/real_Y')
    axs[0, 1].legend()
    # z error
    axs[0, 2].scatter(steps[:], sim_pos_list[:, 0, 2], s=5, c='red', label='sim')
    axs[0, 2].scatter(steps[:], real_pos_list[:, 0, 2], s=5, c='green', label='real')
    axs[0, 2].set_xlabel('steps')
    axs[0, 2].set_ylabel('m')
    axs[0, 2].set_title('sim/real_Z')
    axs[0, 2].legend()
    pos_error = np.square(sim_pos_list - real_pos_list)
    # print('sim_real/X_error', np.mean(pos_error, axis=0)[0,0])
    # print('sim_real/Y_error', np.mean(pos_error, axis=0)[0,1])
    # print('sim_real/Z_error', np.mean(pos_error, axis=0)[0,2])
    print('sim_real/Pos_error', np.mean(np.sum(pos_error, axis=-1)))
    # print('#'*55)
    
    # quat1 error
    axs[1, 0].scatter(steps[:], sim_quat_list[:, 0, 0], s=5, c='red', label='sim')
    axs[1, 0].scatter(steps[:], real_quat_list[:, 0, 0], s=5, c='green', label='real')
    axs[1, 0].set_xlabel('steps')
    axs[1, 0].set_ylabel('rad')
    axs[1, 0].set_title('sim/real_quat1')
    axs[1, 0].legend()
    # quat2 error
    axs[1, 1].scatter(steps[:], sim_quat_list[:, 0, 1], s=5, c='red', label='sim')
    axs[1, 1].scatter(steps[:], real_quat_list[:, 0, 1], s=5, c='green', label='real')
    axs[1, 1].set_xlabel('steps')
    axs[1, 1].set_ylabel('rad')
    axs[1, 1].set_title('sim/real_quat2')
    axs[1, 1].legend()
    # quat3 error
    axs[1, 2].scatter(steps[:], sim_quat_list[:, 0, 2], s=5, c='red', label='sim')
    axs[1, 2].scatter(steps[:], real_quat_list[:, 0, 2], s=5, c='green', label='real')
    axs[1, 2].set_xlabel('steps')
    axs[1, 2].set_ylabel('rad')
    axs[1, 2].set_title('sim/real_quat3')
    axs[1, 2].legend()
    # quat4 error
    axs[1, 3].scatter(steps[:], sim_quat_list[:, 0, 3], s=5, c='red', label='sim')
    axs[1, 3].scatter(steps[:], real_quat_list[:, 0, 3], s=5, c='green', label='real')
    axs[1, 3].set_xlabel('steps')
    axs[1, 3].set_ylabel('rad')
    axs[1, 3].set_title('sim/real_quat4')
    axs[1, 3].legend()
    quat_error = np.square(sim_quat_list - real_quat_list)
    # print('sim_real/Quat1_error', np.mean(quat_error, axis=0)[0,0])
    # print('sim_real/Quat2_error', np.mean(quat_error, axis=0)[0,1])
    # print('sim_real/Quat3_error', np.mean(quat_error, axis=0)[0,2])
    # print('sim_real/Quat4_error', np.mean(quat_error, axis=0)[0,3])
    print('sim_real/Quat_error', np.mean(np.sum(quat_error, axis=-1)))
    # print('#'*55)

    # vel x error
    axs[2, 0].scatter(steps[:], sim_vel_list[:, 0, 0], s=5, c='red', label='sim')
    axs[2, 0].scatter(steps[:], real_vel_list[:, 0, 0], s=5, c='green', label='real')
    axs[2, 0].set_xlabel('steps')
    axs[2, 0].set_ylabel('m/s')
    axs[2, 0].set_title('sim/real_velx')
    axs[2, 0].legend()
    # vel y error
    axs[2, 1].scatter(steps[:], sim_vel_list[:, 0, 1], s=5, c='red', label='sim')
    axs[2, 1].scatter(steps[:], real_vel_list[:, 0, 1], s=5, c='green', label='real')
    axs[2, 1].set_xlabel('steps')
    axs[2, 1].set_ylabel('m/s')
    axs[2, 1].set_title('sim/real_vely')
    axs[2, 1].legend()
    # vel z error
    axs[2, 2].scatter(steps[:], sim_vel_list[:, 0, 2], s=5, c='red', label='sim')
    axs[2, 2].scatter(steps[:], real_vel_list[:, 0, 2], s=5, c='green', label='real')
    axs[2, 2].set_xlabel('steps')
    axs[2, 2].set_ylabel('m/s')
    axs[2, 2].set_title('sim/real_velz')
    axs[2, 2].legend()
    vel_error = np.square(sim_vel_list - real_vel_list)
    # print('sim_real/Velx_error', np.mean(vel_error, axis=0)[0,0])
    # print('sim_real/Vely_error', np.mean(vel_error, axis=0)[0,1])
    # print('sim_real/Velz_error', np.mean(vel_error, axis=0)[0,2])
    print('sim_real/Vel_error', np.mean(np.sum(vel_error, axis=-1)))
    # print('#'*55)

    # angvel x error
    axs[3, 0].scatter(steps[:], sim_angvel_list[:, 0, 0], s=5, c='red', label='sim')
    axs[3, 0].scatter(steps[:], real_angvel_list[:, 0, 0], s=5, c='green', label='real')
    axs[3, 0].set_xlabel('steps')
    axs[3, 0].set_ylabel('rad/s')
    axs[3, 0].set_title('sim/real_angvelx')
    axs[3, 0].legend()
    # angvel y error
    axs[3, 1].scatter(steps[:], sim_angvel_list[:, 0, 1], s=5, c='red', label='sim')
    axs[3, 1].scatter(steps[:], real_angvel_list[:, 0, 1], s=5, c='green', label='real')
    axs[3, 1].set_xlabel('steps')
    axs[3, 1].set_ylabel('rad/s')
    axs[3, 1].set_title('sim/real_angvely')
    axs[3, 1].legend()
    # angvel z error
    axs[3, 2].scatter(steps[:], sim_angvel_list[:, 0, 2], s=5, c='red', label='sim')
    axs[3, 2].scatter(steps[:], real_angvel_list[:, 0, 2], s=5, c='green', label='real')
    axs[3, 2].set_xlabel('steps')
    axs[3, 2].set_ylabel('rad/s')
    axs[3, 2].set_title('sim/real_angvelz')
    axs[3, 2].legend()
    angvel_error = np.square(sim_angvel_list - real_angvel_list)
    # print('sim_real/Angvelx_error', np.mean(angvel_error, axis=0)[0,0])
    # print('sim_real/Angvely_error', np.mean(angvel_error, axis=0)[0,1])
    # print('sim_real/Angvelz_error', np.mean(angvel_error, axis=0)[0,2])
    print('sim_real/Angvel_error', np.mean(np.sum(angvel_error, axis=-1)))
    # # print('#'*55)
    
    # body rate x error
    axs[4, 0].scatter(steps[:], sim_body_rate_list[:, 0, 0], s=5, c='red', label='sim')
    axs[4, 0].scatter(steps[:], real_body_rate_list[:, 0, 0], s=5, c='green', label='real')
    axs[4, 0].set_xlabel('steps')
    axs[4, 0].set_ylabel('rad/s')
    axs[4, 0].set_title('sim/real_bodyratex')
    axs[4, 0].legend()
    # body rate y error
    axs[4, 1].scatter(steps[:], sim_body_rate_list[:, 0, 1], s=5, c='red', label='sim')
    axs[4, 1].scatter(steps[:], real_body_rate_list[:, 0, 1], s=5, c='green', label='real')
    axs[4, 1].set_xlabel('steps')
    axs[4, 1].set_ylabel('rad/s')
    axs[4, 1].set_title('sim/real_bodyratey')
    axs[4, 1].legend()
    # body rate z error
    axs[4, 2].scatter(steps[:], sim_body_rate_list[:, 0, 2], s=5, c='red', label='sim')
    axs[4, 2].scatter(steps[:], real_body_rate_list[:, 0, 2], s=5, c='green', label='real')
    axs[4, 2].set_xlabel('steps')
    axs[4, 2].set_ylabel('rad/s')
    axs[4, 2].set_title('sim/real_bodyratez')
    axs[4, 2].legend()
    bodyrate_error = np.square(sim_body_rate_list - real_body_rate_list)
    # print('sim_real/Bodyratex_error', np.mean(bodyrate_error, axis=0)[0,0])
    # print('sim_real/Bodyratey_error', np.mean(bodyrate_error, axis=0)[0,1])
    # print('sim_real/Bodyratez_error', np.mean(bodyrate_error, axis=0)[0,2])
    print('sim_real/Bodyrate_error', np.mean(np.sum(bodyrate_error, axis=-1)))
    # print('#'*55)
    
    # sim & target
    axs[5, 0].scatter(steps[:], sim_body_rate_list[:, 0, 0], s=5, c='red', label='sim')
    axs[5, 0].scatter(steps[:], target_body_rate_list[:, 0, 0], s=5, c='green', label='target')
    axs[5, 0].set_xlabel('steps')
    axs[5, 0].set_ylabel('rad/s')
    axs[5, 0].set_title('sim/target_bodyratex')
    axs[5, 0].legend()
    # body rate y error
    axs[5, 1].scatter(steps[:], sim_body_rate_list[:, 0, 1], s=5, c='red', label='sim')
    axs[5, 1].scatter(steps[:], target_body_rate_list[:, 0, 1], s=5, c='green', label='target')
    axs[5, 1].set_xlabel('steps')
    axs[5, 1].set_ylabel('rad/s')
    axs[5, 1].set_title('sim/target_bodyratey')
    axs[5, 1].legend()
    # body rate z error
    axs[5, 2].scatter(steps[:], sim_body_rate_list[:, 0, 2], s=5, c='red', label='sim')
    axs[5, 2].scatter(steps[:], target_body_rate_list[:, 0, 2], s=5, c='green', label='target')
    axs[5, 2].set_xlabel('steps')
    axs[5, 2].set_ylabel('rad/s')
    axs[5, 2].set_title('sim/target_bodyratez')
    axs[5, 2].legend()
    target_bodyrate_error = np.square(sim_body_rate_list - target_body_rate_list)
    # print('sim_target/Bodyratex_error', np.mean(target_bodyrate_error, axis=0)[0,0])
    # print('sim_target/Bodyratey_error', np.mean(target_bodyrate_error, axis=0)[0,1])
    # print('sim_target/Bodyratez_error', np.mean(target_bodyrate_error, axis=0)[0,2])
    print('sim_target/Bodyrate_error', np.mean(np.sum(target_bodyrate_error, axis=-1)))
    # print('#'*55)

    # real & target
    axs[6, 0].scatter(steps[:], real_body_rate_list[:, 0, 0], s=5, c='red', label='real')
    axs[6, 0].scatter(steps[:], target_body_rate_list[:, 0, 0], s=5, c='green', label='target')
    axs[6, 0].set_xlabel('steps')
    axs[6, 0].set_ylabel('rad/s')
    axs[6, 0].set_title('real/target_bodyratex')
    axs[6, 0].legend()
    # body rate y error
    axs[6, 1].scatter(steps[:], real_body_rate_list[:, 0, 1], s=5, c='red', label='real')
    axs[6, 1].scatter(steps[:], target_body_rate_list[:, 0, 1], s=5, c='green', label='target')
    axs[6, 1].set_xlabel('steps')
    axs[6, 1].set_ylabel('rad/s')
    axs[6, 1].set_title('real/target_bodyratey')
    axs[6, 1].legend()
    # body rate z error
    axs[6, 2].scatter(steps[:], real_body_rate_list[:, 0, 2], s=5, c='red', label='real')
    axs[6, 2].scatter(steps[:], target_body_rate_list[:, 0, 2], s=5, c='green', label='target')
    axs[6, 2].set_xlabel('steps')
    axs[6, 2].set_ylabel('rad/s')
    axs[6, 2].set_title('real/target_bodyratez')
    axs[6, 2].legend()
    real_target_bodyrate_error = np.square(real_body_rate_list - target_body_rate_list)
    # print('sim_target/Bodyratex_error', np.mean(target_bodyrate_error, axis=0)[0,0])
    # print('sim_target/Bodyratey_error', np.mean(target_bodyrate_error, axis=0)[0,1])
    # print('sim_target/Bodyratez_error', np.mean(target_bodyrate_error, axis=0)[0,2])
    print('real_target/Bodyrate_error', np.mean(np.sum(real_target_bodyrate_error, axis=-1)))
    # print('#'*55)
    
    # motor thrust ratio list
    axs[7, 0].scatter(steps[:], sim_motor_thrust_ratio[:, 0, 0], s=5, c='red', label='sim')
    axs[7, 0].scatter(steps[:], real_motor_thrust_ratio[:, 0, 0], s=5, c='green', label='real')
    axs[7, 0].set_xlabel('steps')
    axs[7, 0].set_ylabel('ratio')
    axs[7, 0].set_title('sim/real_motor1_thrust_ratio')
    axs[7, 0].legend()
    
    axs[7, 1].scatter(steps[:], sim_motor_thrust_ratio[:, 0, 1], s=5, c='red', label='sim')
    axs[7, 1].scatter(steps[:], real_motor_thrust_ratio[:, 0, 1], s=5, c='green', label='real')
    axs[7, 1].set_xlabel('steps')
    axs[7, 1].set_ylabel('ratio')
    axs[7, 1].set_title('sim/real_motor2_thrust_ratio')
    axs[7, 1].legend()
    
    axs[7, 2].scatter(steps[:], sim_motor_thrust_ratio[:, 0, 2], s=5, c='red', label='sim')
    axs[7, 2].scatter(steps[:], real_motor_thrust_ratio[:, 0, 2], s=5, c='green', label='real')
    axs[7, 2].set_xlabel('steps')
    axs[7, 2].set_ylabel('ratio')
    axs[7, 2].set_title('sim/real_motor3_thrust_ratio')
    axs[7, 2].legend()

    axs[7, 3].scatter(steps[:], sim_motor_thrust_ratio[:, 0, 3], s=5, c='red', label='sim')
    axs[7, 3].scatter(steps[:], real_motor_thrust_ratio[:, 0, 3], s=5, c='green', label='real')
    axs[7, 3].set_xlabel('steps')
    axs[7, 3].set_ylabel('ratio')
    axs[7, 3].set_title('sim/real_motor4_thrust_ratio')
    axs[7, 3].legend()
    
    motor_thrust_error = np.square(sim_motor_thrust_ratio - real_motor_thrust_ratio)
    print('sim_real/motor_thrust_ratio_error', np.mean(np.sum(motor_thrust_error, axis=-1)))
    
    # req motor thrust ratio error
    if not use_real_action: # check controller
        axs[8, 0].scatter(steps[:], sim_req_motor_thrust_ratio_list[:, 0, 0], s=5, c='red', label='sim_controller')
        axs[8, 0].scatter(steps[:], real_req_motor_thrust_ratio_list[:, 0, 0], s=5, c='green', label='real_controller')
        axs[8, 0].set_xlabel('steps')
        axs[8, 0].set_ylabel('ratio')
        axs[8, 0].set_title('sim/real_req_motor1_thrust_ratio')
        axs[8, 0].legend()
        
        axs[8, 1].scatter(steps[:], sim_req_motor_thrust_ratio_list[:, 0, 1], s=5, c='red', label='sim_controller')
        axs[8, 1].scatter(steps[:], real_req_motor_thrust_ratio_list[:, 0, 1], s=5, c='green', label='real_controller')
        axs[8, 1].set_xlabel('steps')
        axs[8, 1].set_ylabel('ratio')
        axs[8, 1].set_title('sim/real_req_motor2_thrust_ratio')
        axs[8, 1].legend()
        
        axs[8, 2].scatter(steps[:], sim_req_motor_thrust_ratio_list[:, 0, 2], s=5, c='red', label='sim_controller')
        axs[8, 2].scatter(steps[:], real_req_motor_thrust_ratio_list[:, 0, 2], s=5, c='green', label='real_controller')
        axs[8, 2].set_xlabel('steps')
        axs[8, 2].set_ylabel('ratio')
        axs[8, 2].set_title('sim/real_req_motor3_thrust_ratio')
        axs[8, 2].legend()

        axs[8, 3].scatter(steps[:], sim_req_motor_thrust_ratio_list[:, 0, 3], s=5, c='red', label='sim_controller')
        axs[8, 3].scatter(steps[:], real_req_motor_thrust_ratio_list[:, 0, 3], s=5, c='green', label='real_controller')
        axs[8, 3].set_xlabel('steps')
        axs[8, 3].set_ylabel('ratio')
        axs[8, 3].set_title('sim/real_req_motor4_thrust_ratio')
        axs[8, 3].legend()
        
        req_motor_thrust_ratio_error = np.square(sim_req_motor_thrust_ratio_list - real_req_motor_thrust_ratio_list)
        # print('sim_real/motor1_error', np.mean(motor_thrust_error, axis=0)[0,0])
        # print('sim_real/motor2_error', np.mean(motor_thrust_error, axis=0)[0,1])
        # print('sim_real/motor3_error', np.mean(motor_thrust_error, axis=0)[0,2])
        # print('sim_real/motor4_error', np.mean(motor_thrust_error, axis=0)[0,3])
        print('sim_real/req_motor_thrust_ratio_error', np.mean(np.sum(req_motor_thrust_ratio_error, axis=-1)))
        # print('sim_real/motor_error', np.mean(motor_thrust_error))
    
    plt.tight_layout()
    plt.savefig('comparison_sim_real')

    # plot trajectory
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection='3d')
    ax_3d.scatter(sim_pos_list[:, 0, 0], sim_pos_list[:, 0, 1], sim_pos_list[:, 0, 2], s=5, label='sim')
    ax_3d.scatter(real_pos_list[:, 0, 0], real_pos_list[:, 0, 1], real_pos_list[:, 0, 2], s=5, label='real')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend()
    plt.savefig('simopt')
    
    simulation_app.close()

if __name__ == "__main__":
    main()