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
    '/home/jiayu/OmniDrones/realdata/crazyflie/8_100hz_light.csv',
    # '/home/cf/ros2_ws/rosbags/takeoff.csv',
    # '/home/cf/ros2_ws/rosbags/square.csv',
    # '/home/cf/ros2_ws/rosbags/rl.csv',
]

CRAZYFLIE_PARAMS = [
    'mass',
    'inertia_xx',
    'inertia_yy',
    'inertia_zz',
    'arm_lengths',
    'force_constants',
    'max_rotation_velocities',
    'moment_constants',
    # 'rotor_angles',
    'drag_coef',
    'time_constant',
    'gain',
]

@hydra.main(version_base=None, config_path=".", config_name="real2sim")
def main(cfg):
    """
        preprocess real data
        real_data: [batch_size, T, dimension]
    """
    df = pd.read_csv(rosbags[0], skip_blank_lines=True)
    df = np.array(df)
    # preprocess, motor > 0
    use_preprocess = True
    if use_preprocess:
        preprocess_df = []
        for df_one in df:
            if df_one[-1] > 0:
                preprocess_df.append(df_one)
        preprocess_df = np.array(preprocess_df)
    else:
        preprocess_df = df
    episode_length = preprocess_df.shape[0]
    real_data = []
    # T = 20
    # skip = 5
    T = 1
    skip = 1
    for i in range(0, episode_length-T, skip):
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

    average_dt = 0.01
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
    # origin
    params = [
        0.03,
        1.4e-5,
        1.4e-5,
        2.17e-5,
        0.043,
        2.88e-8,
        2315,
        7.24e-10,
        0.2,
        0.43,
        0.0052,
        0.0052,
        0.00025
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
    # controller = RateController(9.81, drone.params).to(sim.device)
    controller = PIDRateController(9.81, drone.params).to(sim.device)
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
        'motor.time', 'motor.m1', 'motor.m2', 'motor.m3', 'motor.m4'],(28:32)
        dtype='object')
    '''
    sim_pos_list = []
    sim_quat_list = []
    sim_vel_list = []
    sim_body_rate_list = []
    sim_angvel_list = []

    real_pos_list = []
    real_quat_list = []
    real_vel_list = []
    real_body_rate_list = []
    real_angvel_list = []
    
    target_body_rate_list = []
    
    # for i in range(real_data.shape[0]):
    for i in range(real_data.shape[0] - 1):
        real_pos = torch.tensor(real_data[i, :, 1:4])
        real_quat = torch.tensor(real_data[i, :, 5:9])
        real_vel = torch.tensor(real_data[i, :, 10:13])
        real_rate = torch.tensor(real_data[i, :, 18:21])
        real_next_rate = torch.tensor(real_data[i + 1, :, 18:21])
        real_rate[:, 1] = -real_rate[:, 1]
        real_next_rate[:, 1] = -real_next_rate[:, 1]
        # get angvel
        real_ang_vel = quat_rotate(real_quat, real_rate)
        if i == 0:
            set_drone_state(real_pos, real_quat, real_vel, real_ang_vel)
        
        # save
        real_pos_list.append(real_pos.numpy())
        real_quat_list.append(real_quat.numpy())
        real_vel_list.append(real_vel.numpy())
        real_body_rate_list.append(real_rate.numpy())
        real_angvel_list.append(real_ang_vel.numpy())

        drone_state = drone.get_state()[..., :13].reshape(-1, 13)
        # get current_rate
        pos, rot, linvel, angvel = drone_state.split([3, 4, 3, 3], dim=1)
        current_rate = quat_rotate_inverse(rot, angvel)
        target_thrust = torch.tensor(real_data[i, :, 26]).to(device=sim.device).float()
        target_rate = torch.tensor(real_data[i, :, 23:26]).to(device=sim.device).float()
        real_rate = real_rate.to(device=sim.device).float()
        real_next_rate = real_next_rate.to(device=sim.device).float()
        target_rate[:, 1] = -target_rate[:, 1]
        action = controller.sim_step(
            current_rate=current_rate,
            target_rate=target_rate / 180 * torch.pi,
            # target_rate=real_next_rate,
            target_thrust=target_thrust.unsqueeze(1) / (2**16) * max_thrust
        )
        
        drone.apply_action(action)
        # _, thrust, torques = drone.apply_action_foropt(action)
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

        # get body_rate and thrust & compare
        target_body_rate = (target_rate / 180 * torch.pi).cpu()
        target_thrust = target_thrust.unsqueeze(1) / (2**16) * max_thrust
        target_body_rate_list.append(target_body_rate.detach().numpy())

    # now run optimization
    print('*'*55)
    
    steps = np.arange(0, real_data.shape[0]-1)
    real_body_rate_list = np.array(real_body_rate_list)
    target_body_rate_list = np.array(target_body_rate_list)
    sim_body_rate_list = np.array(sim_body_rate_list)
    
    sim_pos_list = np.array(sim_pos_list)
    real_pos_list = np.array(real_pos_list)
    # plot trajectory
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection='3d')
    ax_3d.scatter(sim_pos_list[:, 0, 0], sim_pos_list[:, 0, 1], sim_pos_list[:, 0, 2], s=5, label='sim')
    # ax_3d.scatter(real_pos_list[:, 0, 0], real_pos_list[:, 0, 1], real_pos_list[:, 0, 2], s=5, label='real')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend()
    # plt.savefig('RateController_tracktarget_trajectory')
    plt.savefig('PIDController_tracktarget_trajectory')

    # sim track target
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))  # 1 * 3    
    # sim
    axs[0].scatter(steps, sim_body_rate_list[:, 0, 0], s=5, c='red', label='sim')
    axs[0].scatter(steps, target_body_rate_list[:, 0, 0], s=5, c='green', label='target')
    axs[0].set_xlabel('steps')
    axs[0].set_ylabel('rad/s')
    axs[0].set_title('sim/target_body_rate_x')
    axs[0].legend()
    
    axs[1].scatter(steps, sim_body_rate_list[:, 0, 1], s=5, c='red', label='sim')
    axs[1].scatter(steps, target_body_rate_list[:, 0, 1], s=5, c='green', label='target')
    axs[1].set_xlabel('steps')
    axs[1].set_ylabel('rad/s')
    axs[1].set_title('sim/target_body_rate_y')
    axs[1].legend()
    
    axs[2].scatter(steps, sim_body_rate_list[:, 0, 2], s=5, c='red', label='sim')
    axs[2].scatter(steps, target_body_rate_list[:, 0, 2], s=5, c='green', label='target')
    axs[2].set_xlabel('steps')
    axs[2].set_ylabel('rad/s')
    axs[2].set_title('sim/target_body_rate_z')
    axs[2].legend()
    
    error = np.square(sim_body_rate_list - target_body_rate_list)
    print('sim_target/body_rateX_error', np.mean(error, axis=0)[0,0])
    print('sim_target/body_rateY_error', np.mean(error, axis=0)[0,1])
    print('sim_target/body_rateZ_error', np.mean(error, axis=0)[0,2])
    print('sim_target/body_rate_error', np.mean(error))
    
    # # sim track real
    # fig, axs = plt.subplots(1, 3, figsize=(10, 6))  # 1 * 3    
    # axs[0].scatter(steps, sim_body_rate_list[:, 0, 0], s=5, c='red', label='sim')
    # axs[0].scatter(steps, real_body_rate_list[:, 0, 0], s=5, c='green', label='real')
    # axs[0].set_xlabel('steps')
    # axs[0].set_ylabel('rad/s')
    # axs[0].set_title('sim/real_body_rate_x')
    # axs[0].legend()
    
    # axs[1].scatter(steps, sim_body_rate_list[:, 0, 1], s=5, c='red', label='sim')
    # axs[1].scatter(steps, real_body_rate_list[:, 0, 1], s=5, c='green', label='real')
    # axs[1].set_xlabel('steps')
    # axs[1].set_ylabel('rad/s')
    # axs[1].set_title('sim/real_body_rate_y')
    # axs[1].legend()
    
    # axs[2].scatter(steps, sim_body_rate_list[:, 0, 2], s=5, c='red', label='sim')
    # axs[2].scatter(steps, real_body_rate_list[:, 0, 2], s=5, c='green', label='real')
    # axs[2].set_xlabel('steps')
    # axs[2].set_ylabel('rad/s')
    # axs[2].set_title('sim/real_body_rate_z')
    # axs[2].legend()
    
    # error = np.square(sim_body_rate_list - real_body_rate_list)
    # print('sim_real/body_rateX_error', np.mean(error, axis=0)[0,0])
    # print('sim_real/body_rateY_error', np.mean(error, axis=0)[0,1])
    # print('sim_real/body_rateZ_error', np.mean(error, axis=0)[0,2])
    # print('sim_real/body_rate_error', np.mean(error))
    
    # # real
    # axs[1,0].scatter(steps, real_body_rate_list[:, 0, 0], s=5, c='red', label='real')
    # axs[1,0].scatter(steps, target_body_rate_list[:, 0, 0], s=5, c='green', label='target')
    # axs[1,0].set_xlabel('steps')
    # axs[1,0].set_ylabel('rad/s')
    # axs[1,0].set_title('real/target_body_rate_x')
    # axs[1,0].legend()
    
    # axs[1,1].scatter(steps, real_body_rate_list[:, 0, 1], s=5, c='red', label='real')
    # axs[1,1].scatter(steps, target_body_rate_list[:, 0, 1], s=5, c='green', label='target')
    # axs[1,1].set_xlabel('steps')
    # axs[1,1].set_ylabel('rad/s')
    # axs[1,1].set_title('real/target_body_rate_y')
    # axs[1,1].legend()
    
    # axs[1,2].scatter(steps, real_body_rate_list[:, 0, 2], s=5, c='red', label='real')
    # axs[1,2].scatter(steps, target_body_rate_list[:, 0, 2], s=5, c='green', label='target')
    # axs[1,2].set_xlabel('steps')
    # axs[1,2].set_ylabel('rad/s')
    # axs[1,2].set_title('real/target_body_rate_z')
    # axs[1,2].legend()
    
    # plt.savefig('RateController_trackTarget')
    plt.savefig('PIDController_trackTarget')
    
    simulation_app.close()

if __name__ == "__main__":
    main()