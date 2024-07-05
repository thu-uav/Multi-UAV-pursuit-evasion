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
import quaternion
import numpy as np
import yaml
from skopt import Optimizer
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse, quaternion_to_euler, euler_to_quaternion
import matplotlib.pyplot as plt
import numpy as np

rosbags = [
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_large_data1.csv',
]

def exclude_battery_compensation(PWMs, voltages):
    r"""Make PWM motor signals as if battery is 100% charged."""
    percentage = PWMs / 65535
    volts = percentage * voltages

    a = -0.0006239
    b = 0.088
    c = -volts
    c_min = b**2/(4*a)
    D = np.clip(b ** 2 - 4 * a * c, c_min, np.inf)
    thrust = (-b + np.sqrt(D)) / (2 * a)
    PWMs_cleaned = np.clip(thrust / 60, 0, 1) * 65535
    return PWMs_cleaned

@hydra.main(version_base=None, config_path=".", config_name="real2sim")
def main(cfg):
    """
        preprocess real data
        real_data: [batch_size, T, dimension]
    """
    df = pd.read_csv(rosbags[0], skip_blank_lines=True)
    # preprocess, motor > 0
    preprocess_df = df[(df[['motor.m1']].to_numpy()[:,0] > 0)]
    preprocess_df = preprocess_df[:1600]
    pos = preprocess_df[['pos.x', 'pos.y', 'pos.z']].to_numpy()
    vel = preprocess_df[['vel.x', 'vel.y', 'vel.z']].to_numpy()
    # quat = preprocess_df[['quat.w', 'quat.x', 'quat.y', 'quat.z']].to_numpy()
    rpy = preprocess_df[['rpy.r', 'rpy.p', 'rpy.y']].to_numpy() / 180.0 * torch.pi
    rpy[..., 1] = - rpy[..., 1] # crazyflie
    quat = euler_to_quaternion(torch.from_numpy(rpy)).numpy()
    rate_rpy = preprocess_df[['rate_rpy.r', 'rate_rpy.p', 'rate_rpy.y']].to_numpy() / 1000.0
    rate_rpy[..., 1] = - rate_rpy[..., 1] # crazyflie
    PWMs = preprocess_df[['motor.m1', 'motor.m2', 'motor.m3', 'motor.m4']].to_numpy()
    # voltages = preprocess_df[['bat']].to_numpy()
    # exclude_battery_compensation_flag = False
    # if exclude_battery_compensation_flag:
    #     PWMs = exclude_battery_compensation(PWMs, voltages)
    action = PWMs / (2**16) * 2 - 1.0
    
    # start sim
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.controllers import RateController, PIDRateController
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg
    import omni_drones.utils.kit as kit_utils
    from omni.isaac.cloner import GridCloner
    from omni.isaac.core.utils import prims as prim_utils, stage as stage_utils

    dt = 0.01
    g = 9.81
    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=dt,
        rendering_dt=dt,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    num_envs = 1
    # drone: MultirotorBase = MultirotorBase.REGISTRY[cfg.drone_model]()
    # n = 1 # parrallel envs
    # translations = torch.zeros(n, 3)
    # translations[:, 1] = torch.arange(n)
    # translations[:, 2] = 0.5
    # drone.spawn(translations=translations)
    # scene_utils.design_scene()

    # create cloner for duplicating the scenes
    env_ns = "/World/envs"
    template_env_ns = "/World/envs/env_0"
    cloner = GridCloner(spacing=8)
    cloner.define_base_env("/World/envs")
    # create the xform prim to hold the template environment
    if not prim_utils.is_prim_path_valid(template_env_ns):
        prim_utils.define_prim(template_env_ns)
    # setup single scene
    # scene_utils.design_scene()
    drone_model = MultirotorBase.REGISTRY['crazyflie']
    cfg = drone_model.cfg_cls(force_sensor=False)
    drone: MultirotorBase = drone_model(cfg=cfg)
    kit_utils.create_ground_plane(
        "/World/defaultGroundPlane",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    drone.spawn(translations=[(0.0, 0.0, 1.5)])
    global_prim_paths =  ["/World/defaultGroundPlane"] # # global_prim_paths = _design_scene()
    # check if any global prim paths are defined
    if global_prim_paths is None:
        global_prim_paths = list()
    envs_prim_paths = cloner.generate_paths(
        env_ns + "/env", num_envs
    )
    envs_positions = cloner.clone(
        source_prim_path=template_env_ns,
        prim_paths=envs_prim_paths,
        replicate_physics=False,
    )
    # convert environment positions to torch tensor
    envs_positions = torch.tensor(
        envs_positions, dtype=torch.float, device='cuda:0'
    )

    # filter collisions within each environment instance
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path,
        "/World/collisions",
        prim_paths=envs_prim_paths,
        global_paths=global_prim_paths,
    )

    """
        evaluate in Omnidrones
        params: suggested params
        real_data: [batch_size, dimension]
        sim: omnidrones core
        drone: crazyflie
        controller: the predefined controller
    """
    # backbone
    params = [
        0.03, 1.4e-5, 1.4e-5, 2.17e-5, 0.043,
        2.350347298350041e-08, 2315, 7.24e-10, 0.2, 0.023255813953488372,
        # controller
        250.0, 250.0, 120.0, # kp
        2.5, 2.5, 2.5, # kd
        500.0, 500.0, 16.7, # ki
        33.3, 33.3, 166.7 # ilimit
    ]
    
    # simopt
    params[0] = 0.03
    params[5] = 2.1965862601402255e-08
    params[8] = 0.08765309988122862
    params[9] = 0.028945283545750024

    
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
    # env_mask = torch.ones(num_envs, dtype=bool, device=sim.device)
    # env_ids = env_mask.nonzero().squeeze(-1)
    # drone._reset_idx(env_ids)
    # controller = RateController(9.81, drone.params).to(sim.device)
    controller = PIDRateController(dt, g, drone.params).to(sim.device)
    # controller.set_byTunablePara(tunable_parameters=tunable_parameters)
    controller = controller.to(sim.device)
    
    max_thrust = controller.max_thrusts.sum(-1)

    def set_drone_state(pos, quat, vel, rate_rpy):
        pos = pos.to(device=sim.device).float()
        quat = quat.to(device=sim.device).float()
        vel = vel.to(device=sim.device).float()
        rate_rpy = rate_rpy.to(device=sim.device).float()
        # convert body rate from body frame to world frame
        ang_vel = quat_rotate(quat, rate_rpy)
        
        drone.set_world_poses(pos + envs_positions, quat)
        whole_vel = torch.cat([vel, ang_vel], dim=-1)
        drone.set_velocities(whole_vel)
        # flush the buffer so that the next getter invocation 
        # returns up-to-date values
        sim._physics_sim_view.flush() 
    
    sim_pos_list = []
    sim_quat_list = []
    sim_vel_list = []
    sim_angvel_list = []

    real_pos_list = []
    real_quat_list = []
    real_vel_list = []
    real_angvel_list = []

    trajectory_len = preprocess_df.shape[0] - 1
    T = 2
    prep_step = 0
    
    for i in range(trajectory_len):
        real_pos = torch.tensor(pos[i]).to(sim.device)
        real_quat = torch.tensor(quat[i]).to(sim.device)
        real_vel = torch.tensor(vel[i]).to(sim.device)
        real_rate_rpy = torch.tensor(rate_rpy[i]).to(sim.device)
        real_action = torch.tensor(action[i]).to(sim.device)

        if i % (T - 1) == 0:
            set_drone_state(real_pos.unsqueeze(0), real_quat.unsqueeze(0), real_vel.unsqueeze(0), real_rate_rpy.unsqueeze(0))
        
        # log
        next_real_pos = torch.tensor(pos[i + 1])
        next_real_quat = torch.tensor(quat[i + 1])
        next_real_vel = torch.tensor(vel[i + 1])
        next_real_rate_rpy = torch.tensor(rate_rpy[i + 1])
        next_real_ang_vel = quat_rotate(next_real_quat.unsqueeze(0), next_real_rate_rpy.unsqueeze(0)).squeeze(0)
        
        real_pos_list.append(next_real_pos.numpy())
        real_quat_list.append(next_real_quat.numpy())
        real_vel_list.append(next_real_vel.numpy())
        real_angvel_list.append(next_real_ang_vel.numpy())
        
        drone.apply_action(real_action)
        
        # _, thrust, torques = drone.apply_action_foropt(action)
        sim.step(render=True)

        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.render()
            continue

        # get simulated drone state
        sim_state = drone.get_state().squeeze(0).cpu()
        sim_pos = sim_state[..., :3] - envs_positions.cpu()
        sim_quat = sim_state[..., 3:7]
        sim_vel = sim_state[..., 7:10]
        sim_omega = sim_state[..., 10:13]

        sim_pos_list.append(sim_pos.cpu().detach().numpy())
        sim_quat_list.append(sim_quat.cpu().detach().numpy())
        sim_vel_list.append(sim_vel.cpu().detach().numpy())
        sim_angvel_list.append(sim_omega.cpu().detach().numpy())

    # now run optimization
    # print('*'*55)
            
    sim_pos_list = np.array(sim_pos_list)[prep_step:]
    sim_quat_list = np.array(sim_quat_list)[prep_step:]
    sim_vel_list = np.array(sim_vel_list)[prep_step:]
    sim_angvel_list = np.array(sim_angvel_list)[prep_step:]
    
    real_pos_list = np.array(real_pos_list)[prep_step:]
    real_quat_list = np.array(real_quat_list)[prep_step:]
    real_vel_list = np.array(real_vel_list)[prep_step:]
    real_angvel_list = np.array(real_angvel_list)[prep_step:]
    
    steps = np.arange(0, sim_pos_list.shape[0])
    
    # error
    fig, axs = plt.subplots(4, 4, figsize=(20, 12))  # 7 * 4 
    fig.subplots_adjust()
    # x error
    axs[0, 0].scatter(steps, sim_pos_list[:, 0, 0], s=1, c='red', label='sim')
    axs[0, 0].scatter(steps, real_pos_list[:, 0], s=1, c='green', label='real')
    axs[0, 0].set_xlabel('steps')
    axs[0, 0].set_ylabel('m')
    axs[0, 0].set_title('sim/real_X')
    axs[0, 0].legend()
    # y error
    axs[0, 1].scatter(steps, sim_pos_list[:, 0, 1], s=1, c='red', label='sim')
    axs[0, 1].scatter(steps, real_pos_list[:, 1], s=1, c='green', label='real')
    axs[0, 1].set_xlabel('steps')
    axs[0, 1].set_ylabel('m')
    axs[0, 1].set_title('sim/real_Y')
    axs[0, 1].legend()
    # z error
    axs[0, 2].scatter(steps[:], sim_pos_list[:, 0, 2], s=1, c='red', label='sim')
    axs[0, 2].scatter(steps[:], real_pos_list[:, 2], s=1, c='green', label='real')
    axs[0, 2].set_xlabel('steps')
    axs[0, 2].set_ylabel('m')
    axs[0, 2].set_title('sim/real_Z')
    axs[0, 2].legend()
    # norm
    min_pos_list = np.min(np.min(sim_pos_list, axis=0), axis=0)
    max_pos_list = np.max(np.max(sim_pos_list, axis=0), axis=0)
    sim_pos_list = (sim_pos_list - min_pos_list) / (max_pos_list - min_pos_list)
    real_pos_list = (real_pos_list - min_pos_list) / (max_pos_list - min_pos_list)
    pos_error = sim_pos_list[:,0] - real_pos_list
    pos_error1 = np.linalg.norm(pos_error, axis=-1, ord=1)
    pos_error2 = np.linalg.norm(pos_error, axis=-1, ord=2)
    print('sim_real/Pos_error', np.mean(pos_error1 + pos_error2))
    # print('#'*55)
    
    # quat1 error
    axs[1, 0].scatter(steps[:], sim_quat_list[:, 0, 0], s=1, c='red', label='sim')
    axs[1, 0].scatter(steps[:], real_quat_list[:, 0], s=1, c='green', label='real')
    axs[1, 0].set_xlabel('steps')
    axs[1, 0].set_ylabel('rad')
    axs[1, 0].set_title('sim/real_quat1')
    axs[1, 0].legend()
    # quat2 error
    axs[1, 1].scatter(steps[:], sim_quat_list[:, 0, 1], s=1, c='red', label='sim')
    axs[1, 1].scatter(steps[:], real_quat_list[:, 1], s=1, c='green', label='real')
    axs[1, 1].set_xlabel('steps')
    axs[1, 1].set_ylabel('rad')
    axs[1, 1].set_title('sim/real_quat2')
    axs[1, 1].legend()
    # quat3 error
    axs[1, 2].scatter(steps[:], sim_quat_list[:, 0, 2], s=1, c='red', label='sim')
    axs[1, 2].scatter(steps[:], real_quat_list[:, 2], s=1, c='green', label='real')
    axs[1, 2].set_xlabel('steps')
    axs[1, 2].set_ylabel('rad')
    axs[1, 2].set_title('sim/real_quat3')
    axs[1, 2].legend()
    # quat4 error
    axs[1, 3].scatter(steps[:], sim_quat_list[:, 0, 3], s=1, c='red', label='sim')
    axs[1, 3].scatter(steps[:], real_quat_list[:, 3], s=1, c='green', label='real')
    axs[1, 3].set_xlabel('steps')
    axs[1, 3].set_ylabel('rad')
    axs[1, 3].set_title('sim/real_quat4')
    axs[1, 3].legend()
    min_quat_list = np.min(np.min(sim_quat_list, axis=0), axis=0)
    max_quat_list = np.max(np.max(sim_quat_list, axis=0), axis=0)
    sim_quat_list = (sim_quat_list - min_quat_list) / (max_quat_list - min_quat_list)
    real_quat_list = (real_quat_list - min_quat_list) / (max_quat_list - min_quat_list)
    quat_error = (sim_quat_list[:,0] - real_quat_list)[..., :2]
    quat_error1 = np.linalg.norm(quat_error, axis=-1, ord=1)
    quat_error2 = np.linalg.norm(quat_error, axis=-1, ord=2)
    print('sim_real/Quat_error', np.mean(quat_error1 + quat_error2))
    # print('#'*55)

    # vel x error
    axs[2, 0].scatter(steps[:], sim_vel_list[:, 0, 0], s=1, c='red', label='sim')
    axs[2, 0].scatter(steps[:], real_vel_list[:, 0], s=1, c='green', label='real')
    axs[2, 0].set_xlabel('steps')
    axs[2, 0].set_ylabel('m/s')
    axs[2, 0].set_title('sim/real_velx')
    axs[2, 0].legend()
    # vel y error
    axs[2, 1].scatter(steps[:], sim_vel_list[:, 0, 1], s=1, c='red', label='sim')
    axs[2, 1].scatter(steps[:], real_vel_list[:, 1], s=1, c='green', label='real')
    axs[2, 1].set_xlabel('steps')
    axs[2, 1].set_ylabel('m/s')
    axs[2, 1].set_title('sim/real_vely')
    axs[2, 1].legend()
    # vel z error
    axs[2, 2].scatter(steps[:], sim_vel_list[:, 0, 2], s=1, c='red', label='sim')
    axs[2, 2].scatter(steps[:], real_vel_list[:, 2], s=1, c='green', label='real')
    axs[2, 2].set_xlabel('steps')
    axs[2, 2].set_ylabel('m/s')
    axs[2, 2].set_title('sim/real_velz')
    axs[2, 2].legend()
    min_vel_list = np.min(np.min(sim_vel_list, axis=0), axis=0)
    max_vel_list = np.max(np.max(sim_vel_list, axis=0), axis=0)
    sim_vel_list = (sim_vel_list - min_vel_list) / (max_vel_list - min_vel_list)
    real_vel_list = (real_vel_list - min_vel_list) / (max_vel_list - min_vel_list)
    vel_error = sim_vel_list[:,0] - real_vel_list
    vel_error1 = np.linalg.norm(vel_error, axis=-1, ord=1)
    vel_error2 = np.linalg.norm(vel_error, axis=-1, ord=2)
    print('sim_real/Vel_error', np.mean(vel_error1 + vel_error2))
    # print('#'*55)

    # angvel x error
    axs[3, 0].scatter(steps[:], sim_angvel_list[:, 0, 0], s=1, c='red', label='sim')
    axs[3, 0].scatter(steps[:], real_angvel_list[:, 0], s=1, c='green', label='real')
    axs[3, 0].set_xlabel('steps')
    axs[3, 0].set_ylabel('rad/s')
    axs[3, 0].set_title('sim/real_angvelx')
    axs[3, 0].legend()
    # angvel y error
    axs[3, 1].scatter(steps[:], sim_angvel_list[:, 0, 1], s=1, c='red', label='sim')
    axs[3, 1].scatter(steps[:], real_angvel_list[:, 1], s=1, c='green', label='real')
    axs[3, 1].set_xlabel('steps')
    axs[3, 1].set_ylabel('rad/s')
    axs[3, 1].set_title('sim/real_angvely')
    axs[3, 1].legend()
    # angvel z error
    axs[3, 2].scatter(steps[:], sim_angvel_list[:, 0, 2], s=1, c='red', label='sim')
    axs[3, 2].scatter(steps[:], real_angvel_list[:, 2], s=1, c='green', label='real')
    axs[3, 2].set_xlabel('steps')
    axs[3, 2].set_ylabel('rad/s')
    axs[3, 2].set_title('sim/real_angvelz')
    axs[3, 2].legend()
    min_angvel_list = np.min(np.min(sim_angvel_list, axis=0), axis=0)
    max_angvel_list = np.max(np.max(sim_angvel_list, axis=0), axis=0)
    sim_angvel_list = (sim_angvel_list - min_angvel_list) / (max_angvel_list - min_angvel_list)
    real_angvel_list = (real_angvel_list - min_angvel_list) / (max_angvel_list - min_angvel_list)
    angvel_error = (sim_angvel_list[:,0] - real_angvel_list)[..., :2]
    angvel_error1 = np.linalg.norm(angvel_error, axis=-1, ord=1)
    angvel_error2 = np.linalg.norm(angvel_error, axis=-1, ord=2)
    print('sim_real/Angvel_error', np.mean(angvel_error1 + angvel_error2))
    # # print('#'*55)
    
    plt.tight_layout()
    plt.savefig('comparison_sim_real.png')
    
    # plot trajectory
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(projection='3d')
    ax_3d.scatter(sim_pos_list[:, 0, 0], sim_pos_list[:, 0, 1], sim_pos_list[:, 0, 2], s=1, label='sim')
    ax_3d.scatter(real_pos_list[:, 0], real_pos_list[:, 1], real_pos_list[:, 2], s=1, label='real')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.legend()
    plt.savefig('trajectory.png')
    
    simulation_app.close()

if __name__ == "__main__":
    main()