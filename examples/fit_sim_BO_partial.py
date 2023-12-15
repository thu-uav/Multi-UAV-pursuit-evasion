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

rosbags = [
    '/home/jiayu/OmniDrones/realdata/crazyflie/takeoff.csv',
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

def init_sim(cfg, n_envs):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))
    average_dt = 0.01

    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.controllers import RateController
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.utils.torch import euler_to_quaternion, quaternion_to_euler
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=average_dt,
        rendering_dt=average_dt,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    drone: MultirotorBase = MultirotorBase.REGISTRY[cfg.drone_model]()
    n = n_envs # parrallel envs
    translations = torch.zeros(n, 3)
    translations[:, 1] = torch.arange(n)
    translations[:, 2] = 0.5
    drone.spawn(translations=translations)

    scene_utils.design_scene()

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(960, 720),
        data_types=["rgb", "distance_to_camera"],
    )
    # camera for visualization
    camera_vis = Camera(camera_cfg)

    sim.reset()
    camera_vis.initialize("/OmniverseKit_Persp")
    drone.initialize()
    drone.base_link.set_masses(torch.tensor([0.03785]).to(sim.device))

    controller = RateController(9.81, drone.params).to(sim.device)
    return sim, drone, controller, simulation_app

def evaluate(params, real_data, sim, drone, controller):
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
    drone.initialize_byTunablePara(tunable_parameters=tunable_parameters)
    controller.set_byTunablePara(tunable_parameters=tunable_parameters)
    controller.to(sim.device)
    mse = torch.nn.functional.mse_loss
    '''
        df:
        Index(['pos.time', 'pos.x', 'pos.y', 'pos.z', 
                'quat.time', 'quat.w', 'quat.x', 'quat.y', 'quat.z',
                'vel.time', 'vel.x', 'vel.y', 'vel.z', 
                'omega.time','omega.r', 'omega.p', 'omega.y', 
                'cmd.time', 'cmd.thrust', 'cmd.r_rate',
                'cmd.p_rate', 'cmd.y_rate'],
    '''
    # shuffle index and split into batches
    shuffled_idx = torch.randperm(real_data.shape[0])
    shuffled_real_data = real_data[shuffled_idx]
    loss = torch.tensor(0.0, dtype=torch.float)
    pos_error = torch.tensor(0.0, dtype=torch.float)
    quat_error = torch.tensor(0.0, dtype=torch.float)
    vel_error = torch.tensor(0.0, dtype=torch.float)
    omega_error = torch.tensor(0.0, dtype=torch.float)
    gamma = 0.95
    
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
    
    for i in range(real_data.shape[1]-1):
        pos = torch.tensor(shuffled_real_data[:, i, 1:4])
        quat = torch.tensor(shuffled_real_data[:, i, 5:9])
        vel = torch.tensor(shuffled_real_data[:, i, 10:13])
        ang_vel = torch.tensor(shuffled_real_data[:, i, 14:17])
        if i == 0 :
            set_drone_state(pos, quat, vel, ang_vel)

        drone_state = drone.get_state()[..., :13]
        cmd_thrust = torch.tensor(shuffled_real_data[:, i, 18]).to(device=sim.device).float()
        cmd_rate = torch.tensor(shuffled_real_data[:, i, 19:]).to(device=sim.device).float()
        action = controller(
            drone_state.squeeze(0), 
            target_rate=cmd_rate / 180 * torch.pi,
            target_thrust=cmd_thrust.unsqueeze(1) / (2**16) * max_thrust
        )
        pdb.set_trace()
        
        drone.apply_action(action)
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

        # get real state & compare
        real_pos = torch.tensor(shuffled_real_data[:, i+1, 1:4])
        real_quat = torch.tensor(shuffled_real_data[:, i+1, 5:9])
        real_vel = torch.tensor(shuffled_real_data[:, i+1, 10:13])
        real_omega = torch.tensor(shuffled_real_data[:, i+1, 14:17])
        
        obs_sim = {
            'pos': sim_pos, 'quat': sim_quat,
            'vel': sim_vel, 'omega': sim_omega,
        }
        obs_real = {
            'pos': real_pos, 'quat': real_quat,
            'vel': real_vel, 'omega': real_omega,
        }
        
        # pos_error += torch.sqrt(mse(sim_pos, real_pos))
        # quat_error += torch.sqrt(mse(sim_quat, real_quat))
        # vel_error += torch.sqrt(mse(sim_vel, real_vel))
        # omega_error += torch.sqrt(mse(sim_omega, real_omega))
        loss += (gamma **i) * loss_function(obs_sim, obs_real)

    # TODO: update loss
    # loss = pos_error + quat_error + vel_error + omega_error
    # loss = pos_error + quat_error

    return loss, pos_error, quat_error, vel_error, omega_error

@hydra.main(version_base=None, config_path=".", config_name="real2sim")
def main(cfg):
    """
        preprocess real data
        real_data: [batch_size, T, dimension]
    """
    df = pd.read_csv(rosbags[0])
    episode_length = df.index.stop
    df = np.array(df)
    real_data = []
    T = 20
    skip = 5
    for i in range(0, episode_length-T, skip):
        _slice = slice(i, i+T)
        real_data.append(df[_slice])
    real_data = np.array(real_data)

    r"""Apply Adams' Stochastic Gradient Descend based on finite-differences."""
    sim, drone, controller, simulation_app = init_sim(cfg, n_envs=real_data.shape[0])

    # start from the yaml
    params = drone.tunable_parameters().detach().tolist()

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
    params_mask[9] = 1
    params_mask[10:] = 1

    # TODO : update the range
    params_range = []
    lower = 0.1
    upper = 100.0
    for param, mask in zip(params, params_mask):
        if mask == 1:
            params_range.append((lower * param, upper * param))
    opt = Optimizer(params_range)

    # set up objective function
    def func(suggested_para, real_data, sim, drone, controller) -> float:
        """A simple callable function that evaluates the objective (fitness)."""
        return evaluate(suggested_para, real_data, sim, drone, controller)

    # now run optimization
    print('*'*55)

    for epoch in range(500):
        print(f'Start with epoch: {epoch}')
        # grad = optimize.approx_fprime(params.detach().numpy(), func, \
        #     eps, real_data, sim, drone, controller)

        suggested_para = np.array(opt.ask(), dtype=float)
        # set real params
        set_idx = 0
        for idx, mask in enumerate(params_mask):
            if mask == 1:
                params[idx] = suggested_para[set_idx]
                set_idx += 1
        grad, pos_error, quat_error, vel_error, omega_error = func(params, real_data, sim, drone, controller)
        res = opt.tell(suggested_para.tolist(), grad.item())
        
        # TODO: export paras to yaml

        # do the logging and save to disk
        print('Epoch', epoch + 1)
        print(f'Param/{suggested_para.tolist()}')
        print(f'Best/{res.x}')
        print('Best/Loss', res.fun, 'Pos/Loss', pos_error, \
            'Quat/Loss', quat_error, 'Vel/Loss', vel_error, \
            'Omega/Loss', omega_error)
    
    simulation_app.close()

if __name__ == "__main__":
    main()