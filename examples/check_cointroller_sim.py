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

def loss_function(obs_sim, obs_real) -> float:
    r"""Computes the distance between observations from sim and real."""

    # angles
    e_rpy = 10 * (obs_sim['quat'] - obs_real['quat'])

    # angle rates
    err_rpy_dot = obs_sim['omega'] - obs_real['omega']

    # position - errors are smaller than angle errors
    e_xyz = 100 * (obs_sim['pos'] - obs_real['pos'])

    # linear velocity
    e_xyz_dot = obs_sim['vel'] - obs_real['vel']

    # Build norms of error vector:
    err = np.hstack((e_rpy.detach().cpu().numpy(), \
        e_xyz.detach().cpu().numpy(), \
        e_xyz_dot.detach().cpu().numpy(), \
        err_rpy_dot.detach().cpu().numpy()))
    L1 = np.linalg.norm(err, ord=1)
    L2 = np.linalg.norm(err, ord=2)
    L = L1 + L2
    return L

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
    exp_name = 'lossThrust_tuneGain'
    # exp_name = 'lossBodyrate_tuneGain_kf'
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
    from omni_drones.controllers import RateController
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
    n = real_data.shape[0] # parrallel envs
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
    controller = RateController(9.81, drone.params).to(sim.device)
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
    
    for i in range(real_data.shape[0]):
        real_pos = torch.tensor(real_data[i, 1:4])
        real_quat = torch.tensor(real_data[i, 5:9])
        real_vel = torch.tensor(real_data[i, 10:13])
        real_rate = torch.tensor(real_data[i, 18:21])
        real_rate[:, 1] = -real_rate[:, 1]
        # get angvel
        real_ang_vel = quat_rotate(real_quat, real_rate)
        real_pos_list.append(real_pos)
        real_quat_list.append(real_quat)
        real_vel_list.append(real_vel)
        real_body_rate_list.append(real_rate)
        real_angvel_list.append(real_ang_vel)
        if i == 0:
            set_drone_state(real_pos, real_quat, real_vel, real_ang_vel)

        drone_state = drone.get_state()[..., :13].reshape(-1, 13)
        # get current_rate
        pos, rot, linvel, angvel = drone_state.split([3, 4, 3, 3], dim=1)
        current_rate = quat_rotate_inverse(rot, angvel)
        target_thrust = torch.tensor(real_data[i, 26]).to(device=sim.device).float()
        target_rate = torch.tensor(real_data[i, 23:26]).to(device=sim.device).float()
        # TODO: check error of current_rate and real_rate, why are they diff ?
        # real_rate = torch.tensor(shuffled_real_data[:, i, 18:21]).to(device=sim.device).float()
        # real_rate[:, 1] = -real_rate[:, 1]
        real_rate = real_rate.to(device=sim.device).float()
        target_rate[:, 1] = -target_rate[:, 1]
        # pdb.set_trace()
        action = controller.sim_step(
            current_rate=current_rate,
            target_rate=target_rate / 180 * torch.pi,
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
        sim_state = drone.get_state().squeeze().cpu()
        sim_pos = sim_state[..., :3]
        sim_quat = sim_state[..., 3:7]
        sim_vel = sim_state[..., 7:10]
        sim_omega = sim_state[..., 10:13]
        next_body_rate = quat_rotate_inverse(sim_quat, sim_omega)

        sim_pos_list.append(sim_pos)
        sim_quat_list.append(sim_quat)
        sim_vel_list.append(sim_vel)
        sim_body_rate_list.append(next_body_rate)
        sim_angvel_list.append(sim_omega)

        # get body_rate and thrust & compare
        target_body_rate = (target_rate / 180 * torch.pi).cpu()
        target_thrust = target_thrust.unsqueeze(1) / (2**16) * max_thrust
        target_body_rate_list.append(target_body_rate)

    # now run optimization
    print('*'*55)
    
    steps = np.arange(0, real_data.shape[0])
    real_body_rate_list = np.array(real_body_rate_list)
    target_body_rate_list = np.array(target_body_rate_list)
    sim_body_rate_list = np.array(sim_body_rate_list)

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.scatter(steps, sim_body_rate_list[:, 0], s=5, label='sim')
    ax1.scatter(steps, target_body_rate_list[:, 0], s=5, label='target')
    ax2 = fig.add_subplot()
    ax2.scatter(steps, sim_body_rate_list[:, 1], s=5, label='sim')
    ax2.scatter(steps, target_body_rate_list[:, 1], s=5, label='target')
    ax3 = fig.add_subplot()
    ax3.scatter(steps, sim_body_rate_list[:, 2], s=5, label='sim')
    ax3.scatter(steps, target_body_rate_list[:, 2], s=5, label='target')
    
    ax1.set_xlabel('steps')
    ax1.set_ylabel('rad/s')
    ax1.set_title('body_rate_x')
    ax1.legend()
    ax2.set_xlabel('steps')
    ax2.set_ylabel('rad/s')
    ax2.set_title('body_rate_y')
    ax2.legend()
    ax3.set_xlabel('steps')
    ax3.set_ylabel('rad/s')
    ax3.set_title('body_rate_z')
    ax3.legend()
    
    plt.savefig('sim_tracking')
    
    simulation_app.close()

if __name__ == "__main__":
    main()