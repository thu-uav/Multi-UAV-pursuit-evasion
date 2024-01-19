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
    # '/home/jiayu/OmniDrones/realdata/crazyflie/train_figure8.csv',
    # '/home/jiayu/OmniDrones/realdata/crazyflie/cf1_figure8.csv',
    '/home/jiayu/OmniDrones/realdata/crazyflie/cf7_figure8.csv',
    # '/home/jiayu/OmniDrones/realdata/crazyflie/cf9_figure8.csv',
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
    # episode_length = preprocess_df.shape[0]
    episode_length = 1400
    real_data = []
    # real_date: [episode_length, num_trajectory, dim]
    T = 1
    for i in range(0, episode_length-T):
        _slice = slice(i, i+T)
        real_data.append(preprocess_df[_slice])
    real_data = np.array(real_data)
    
    # add real next_body_rate
    next_body_rate = real_data[1:,:,18:21]
    # add real next motor_thrust
    next_motor_thrust = real_data[1:,:,28:32]
    # add real next_vel
    next_vel = real_data[1:,:,10:13]
    real_data = np.concatenate([real_data[:-1],next_body_rate], axis=-1)
    real_data = np.concatenate([real_data, next_motor_thrust], axis=-1)
    real_data = np.concatenate([real_data, next_vel], axis=-1)

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
    n = real_data.shape[1] # parrallel envs, num_trajetories
    translations = torch.zeros(n, 3)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 0.5
    drone.spawn(translations=translations)
    scene_utils.design_scene()
    
    # apply_action, if True, opt for rotor
    # if False, opt for controller
    use_real_action = False

    def evaluate(params, real_data):
        """
            evaluate in Omnidrones
            params: suggested params
            real_data: [batch_size, T, dimension]
            sim: omnidrones core
            drone: crazyflie
            controller: the predefined controller
        """
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
            # 'gain': params[10:]
            'pid_kp': params[10:13],
            'pid_kd': params[13:15],
            'pid_ki': params[15:18],
            'iLimit': params[18:21],
        }
        
        # reset sim
        sim.reset()
        drone.initialize_byTunablePara(tunable_parameters=tunable_parameters)
        # controller = RateController(9.81, drone.params).to(sim.device)
        controller = PIDRateController(9.81, drone.params).to(sim.device)
        controller.set_byTunablePara(tunable_parameters=tunable_parameters)
        controller = controller.to(sim.device)
        max_thrust = controller.max_thrusts.sum(-1)
        
        # # shuffle index and split into batches
        # shuffled_idx = torch.randperm(real_data.shape[0])
        # # shuffled_idx = np.arange(0,real_data.shape[0])
        # shuffled_real_data = real_data[shuffled_idx]
        
        loss = torch.tensor(0.0, dtype=torch.float)
        # update simulation parameters
        """
            1. set parameters into sim
            2. update parameters
            3. export para to yaml 
        """
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
            'next_real_rate.r', 'next_real_rate.p', 'next_real_rate.y', (33:35)
            'next_motor1', 'next_motor2', 'next_motor3', 'next_motor4', (35:39)
            'next_vel.x', 'next_vel.y', 'next_vel.z', (39:42)]
            dtype='object')
        '''
        # for i in range(max(1, real_data.shape[1]-1)):
        
        sim_action_list = []
        real_action_list = []
        
        sim_body_rate_list = []
        real_body_rate_list = []

        sim_vel_list = []
        real_vel_list = []
        
        for i in range(real_data.shape[0]):
            pos = torch.tensor(real_data[i, :, 1:4])
            quat = torch.tensor(real_data[i, :, 5:9])
            vel = torch.tensor(real_data[i, :, 10:13])
            real_rate = torch.tensor(real_data[i, :, 18:21])
            next_real_rate = torch.tensor(real_data[i, :, 32:35])
            next_real_motor_thrust = torch.tensor(real_data[i, :, 35:39])
            next_vel = torch.tensor(real_data[i, :, 39:42])
            real_rate[:, 1] = -real_rate[:, 1]
            next_real_rate[:, 1] = -next_real_rate[:, 1]
            # get angvel
            ang_vel = quat_rotate(quat, real_rate)
            # if i == 0 :
            set_drone_state(pos, quat, vel, ang_vel)

            drone_state = drone.get_state()[..., :13].reshape(-1, 13)
            # get current_rate
            pos, rot, linvel, angvel = drone_state.split([3, 4, 3, 3], dim=1)
            current_rate = quat_rotate_inverse(rot, angvel)
            target_thrust = torch.tensor(real_data[i, :, 26]).to(device=sim.device).float()
            target_rate = torch.tensor(real_data[i, :, 23:26]).to(device=sim.device).float()
            real_rate = real_rate.to(device=sim.device).float()
            next_real_rate = next_real_rate.to(device=sim.device).float()
            target_rate[:, 1] = -target_rate[:, 1]
            action = controller.sim_step(
                current_rate=current_rate,
                target_rate=target_rate / 180 * torch.pi,
                target_thrust=target_thrust.unsqueeze(1) / (2**16) * max_thrust
            )
            
            sim_action_list.append(action.detach().to('cpu').numpy())
            
            real_action = next_real_motor_thrust.to(sim.device) / (2**16) * max_thrust * 2 - 1
            real_action_list.append(real_action.detach().to('cpu').numpy())
            
            if use_real_action:
                drone.apply_action(real_action)
            else:
                drone.apply_action(action)
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

            sim_body_rate_list.append(next_body_rate.cpu().detach().numpy())
            real_body_rate_list.append(next_real_rate.cpu().detach().numpy())
            
            sim_vel_list.append(sim_vel.cpu().detach().numpy())
            real_vel_list.append(next_vel.cpu().detach().numpy())
        
        sim_action_list = np.array(sim_action_list).reshape(-1, 4)
        real_action_list = np.array(real_action_list).reshape(-1, 4)
        
        sim_body_rate_list = np.array(sim_body_rate_list).reshape(-1, 3)
        real_body_rate_list = np.array(real_body_rate_list).reshape(-1, 3)
        
        sim_vel_list = np.array(sim_vel_list).reshape(-1, 3)
        real_vel_list = np.array(real_vel_list).reshape(-1, 3)
        
        if use_real_action:
            # opt for rotor
            loss = np.mean(np.square(sim_body_rate_list - real_body_rate_list))
            loss += 0.0 * np.mean(np.square(sim_vel_list - real_vel_list))
        else:
            # opt for controller
            loss = np.mean(np.square(sim_action_list - real_action_list))
            # loss = np.mean(np.square(sim_body_rate_list - real_body_rate_list))

        return loss

    # PID
    params = [
        0.03, 1.4e-5, 1.4e-5, 2.17e-5, 0.043,
        2.88e-8, 2315, 7.24e-10, 0.2, 0.43,
        # controller
        250.0, 250.0, 120.0, # kp
        2.5, 2.5, # kd
        500.0, 500.0, 16.7, # ki
        33.3, 33.3, 166.7 # ilimit
    ]

    """
        'mass': params[0],
        'inertia_xx': params[1],
        'inertia_yy': params[2],
        'inertia_zz': params[3],
        'arm_lengths': params[4],
        'force_constants': params[5], # kf
        'max_rotation_velocities': params[6],
        'moment_constants': params[7], # km
        'drag_coef': params[8],
        'time_constant': params[9], # tau
        # 'gain': params[10:]
        'pid_kp': params[10:13],
        'pid_kd': params[13:15],
        'pid_ki': params[15:18],
        'iLimit': params[18:21],
    """
    params_mask = np.array([0] * len(params))
    if use_real_action:
        # update rotor params
        params_mask[5] = 1
        params_mask[7] = 1
        # params_mask[9] = 1
    else:
        # update controller params
        # params_mask[1] = 1
        # params_mask[2] = 1
        # params_mask[3] = 1
        # params_mask[5] = 1
        # params_mask[7] = 1
        params_mask[10:] = 1

    params_range = []
    lower = 0.1
    upper = 10.0
    count = 0
    for param, mask in zip(params, params_mask):
        if mask == 1:
            params_range.append((lower * param, upper * param))
        count += 1
    opt = Optimizer(
        dimensions=params_range,
        base_estimator='gp',  # Gaussian Process is a common choice
        n_initial_points=10,   # Number of initial random points to sample
        random_state=0        # Set a random seed for reproducibility
        )

    # set up objective function
    def func(suggested_para, real_data) -> float:
        """A simple callable function that evaluates the objective (fitness)."""
        return evaluate(suggested_para, real_data)

    # now run optimization
    print('*'*55)
    losses = []
    rate_error = []
    epochs = []

    for epoch in range(100):
        print(f'Start with epoch: {epoch}')
        
        x = np.array(opt.ask(), dtype=float)
        # set real params
        set_idx = 0
        for idx, mask in enumerate(params_mask):
            if mask == 1:
                params[idx] = x[set_idx]
                set_idx += 1
        grad = func(params, real_data)
        res = opt.tell(x.tolist(), grad.item())
        
        # TODO: export paras to yaml

        # do the logging and save to disk
        print('Epoch', epoch + 1)
        print(f'CurrentParam/{x.tolist()}')
        print(f'Best/{res.x}')
        print('Best/Loss', res.fun)
        losses.append(grad)
        epochs.append(epoch)
    
    simulation_app.close()

if __name__ == "__main__":
    main()