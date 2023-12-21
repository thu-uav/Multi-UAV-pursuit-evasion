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
    '/home/jiayu/OmniDrones/realdata/crazyflie/8_100hz_cjy.csv',
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
            'gain': params[10:]
        }
        
        # reset sim
        sim.reset()
        drone.initialize_byTunablePara(tunable_parameters=tunable_parameters)
        controller = RateController(9.81, drone.params).to(sim.device)
        controller.set_byTunablePara(tunable_parameters=tunable_parameters)
        controller = controller.to(sim.device)
        
        # shuffle index and split into batches
        shuffled_idx = torch.randperm(real_data.shape[0])
        # shuffled_idx = np.arange(0,real_data.shape[0])
        shuffled_real_data = real_data[shuffled_idx]
        loss = torch.tensor(0.0, dtype=torch.float)
        target_sim_rate_error = torch.tensor(0.0, dtype=torch.float)
        pos_error = torch.tensor(0.0, dtype=torch.float)
        quat_error = torch.tensor(0.0, dtype=torch.float)
        vel_error = torch.tensor(0.0, dtype=torch.float)
        omega_error = torch.tensor(0.0, dtype=torch.float)
        gamma = 0.95
        mse = torch.nn.functional.mse_loss
        
        # tunable_parameters = drone.tunable_parameters()
        max_thrust = controller.max_thrusts.sum(-1)

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
            'motor.time', 'motor.m1', 'motor.m2', 'motor.m3', 'motor.m4'],(28:32)
            dtype='object')
        '''
        for i in range(max(1, real_data.shape[1]-1)):
            pos = torch.tensor(shuffled_real_data[:, i, 1:4])
            quat = torch.tensor(shuffled_real_data[:, i, 5:9])
            vel = torch.tensor(shuffled_real_data[:, i, 10:13])
            # body_rate = torch.tensor(shuffled_real_data[:, i, 14:17]) / 180 * torch.pi
            real_rate = torch.tensor(shuffled_real_data[:, i, 18:21])
            real_rate[:, 1] = -real_rate[:, 1]
            # get angvel
            # ang_vel = quat_rotate(quat, body_rate)
            ang_vel = quat_rotate(quat, real_rate)
            if i == 0 :
                set_drone_state(pos, quat, vel, ang_vel)

            drone_state = drone.get_state()[..., :13].reshape(-1, 13)
            # get current_rate
            pos, rot, linvel, angvel = drone_state.split([3, 4, 3, 3], dim=1)
            # current_rate = quat_rotate_inverse(rot, angvel)
            target_thrust = torch.tensor(shuffled_real_data[:, i, 26]).to(device=sim.device).float()
            target_rate = torch.tensor(shuffled_real_data[:, i, 23:26]).to(device=sim.device).float()
            # TODO: check error of current_rate and real_rate, why are they diff ?
            # real_rate = torch.tensor(shuffled_real_data[:, i, 18:21]).to(device=sim.device).float()
            # real_rate[:, 1] = -real_rate[:, 1]
            real_rate = real_rate.to(device=sim.device).float()
            target_rate[:, 1] = -target_rate[:, 1]
            # pdb.set_trace()
            action = controller.sim_step(
                current_rate=real_rate,
                target_rate=target_rate / 180 * torch.pi,
                target_thrust=target_thrust.unsqueeze(1) / (2**16) * max_thrust
            )
            
            # drone.apply_action(action)
            _, thrust, torques = drone.apply_action_foropt(action)
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

            # get body_rate and thrust & compare
            target_body_rate = (target_rate / 180 * torch.pi).cpu()
            target_thrust = target_thrust.unsqueeze(1) / (2**16) * max_thrust
            
            # loss: thrust error
            # real_motor_thrust = torch.tensor(shuffled_real_data[:, i, 28:32]).to(device=sim.device).float() / (2**16) * max_thrust
            # mask = ~torch.isnan(thrust.squeeze(0))
            # thrust_no_nan = thrust.squeeze(0)[mask]
            # real_motor_thrust_no_nan = real_motor_thrust[mask]
            # loss += mse(thrust_no_nan.to('cpu'), real_motor_thrust_no_nan.to('cpu'))
            
            # TODO: mask NaN, why?
            # body rate error
            mask = ~torch.isnan(next_body_rate)
            next_body_rate_no_nan = next_body_rate[mask]
            target_body_rate_no_nan = target_body_rate[mask]
            loss += mse(next_body_rate_no_nan.to('cpu'), target_body_rate_no_nan.to('cpu'))
            
            # report
            target_sim_rate_error += mse(next_body_rate_no_nan, target_body_rate_no_nan)
            # target_gt_thrust_error += mse(gt_thrust, target_thrust)

        return loss, target_sim_rate_error

    # sim, drone, controller, simulation_app = init_sim(cfg, n_envs=real_data.shape[0])

    # start from the yaml
    # params = drone.tunable_parameters().detach().tolist()
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

    """
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
    """
    params_mask = np.array([0] * 13)
    params_mask[5] = 1
    params_mask[7] = 1
    params_mask[10:] = 1

    params_range = []
    lower = 0.01
    upper = 100.0
    for param, mask in zip(params, params_mask):
        if mask == 1:
            params_range.append((lower * param, upper * param))
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
        # x = np.array([0.0052, 0.0052, 0.00025])
        # set real params
        set_idx = 0
        for idx, mask in enumerate(params_mask):
            if mask == 1:
                params[idx] = x[set_idx]
                set_idx += 1
        grad, target_sim_rate_error = func(params, real_data)
        res = opt.tell(x.tolist(), grad.item())
        
        # TODO: export paras to yaml

        # do the logging and save to disk
        print('Epoch', epoch + 1)
        print(f'CurrentParam/{x.tolist()}')
        print(f'Best/{res.x}')
        print('Best/Loss', res.fun, \
            'Body rate/error', target_sim_rate_error)
        losses.append(grad)
        rate_error.append(target_sim_rate_error)
        epochs.append(epoch)

    losses = torch.tensor(losses).detach().numpy()
    rate_error = torch.tensor(rate_error).detach().numpy()
    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(epochs, losses, s=5, label='loss')
    # ax.scatter(epochs, rate_error, s=5, label='rate error')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    plt.savefig('training_curve_{}'.format(exp_name))
    
    simulation_app.close()

if __name__ == "__main__":
    main()