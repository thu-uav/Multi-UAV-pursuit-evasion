import os

from typing import Dict, Optional
import torch
from functorch import vmap

import hydra
from omegaconf import OmegaConf
from omni_drones import init_simulation_app
from tensordict import TensorDict
import pandas as pd
import pdb

rosbags = [
    '/root/OmniDrones/realdata/crazyflie/takeoff.csv',
    # '/home/cf/ros2_ws/rosbags/takeoff.csv',
    # '/home/cf/ros2_ws/rosbags/square.csv',
    # '/home/cf/ros2_ws/rosbags/rl.csv',
]

@hydra.main(version_base=None, config_path=".", config_name="real2sim")
def main(cfg):
    pdb.set_trace()
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    average_dt = 0.01
    df = pd.read_csv(rosbags[0])
    episode_length = df.index.stop

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
    n = 1
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
    max_thrust = controller.max_thrusts.sum(-1)

    def set_drone_state(pos, quat, vel, ang_vel):
        pos = pos.to(device=sim.device).float()
        quat = quat.to(device=sim.device).float()
        vel = vel.to(device=sim.device).float()
        ang_vel = ang_vel.to(device=sim.device).float()
        drone.set_world_poses(pos, quat)
        whole_vel = torch.cat([vel, ang_vel])
        drone.set_velocities(whole_vel)
        # flush the buffer so that the next getter invocation 
        # returns up-to-date values
        sim._physics_sim_view.flush() 

    frames_vis = []

    pos_error = []
    quat_error = []
    vel_error = []
    omega_error = []
    pos_change = []
    quat_change = []
    vel_change = []
    omega_change = []
    pos_abs_error = []
    vel_abs_error = []
    quat_abs_error = []
    omega_abs_error = []
    mse = torch.nn.functional.mse_loss

    sim_poses = []
    real_poses = []

    for i in range(episode_length):
        # set real drone state
        current_state = df.loc[i]
        pos = torch.tensor([current_state['pos.x'], current_state['pos.y'], current_state['pos.z']])
        quat = torch.tensor([current_state['quat.w'], current_state['quat.x'], current_state['quat.y'], current_state['quat.z']])
        vel = torch.tensor([current_state['vel.x'], current_state['vel.y'], current_state['vel.z']])
        ang_vel = torch.tensor([current_state['omega.r'], current_state['omega.p'], current_state['omega.y']]) # in radius / s
        if i == 0:
            set_drone_state(pos, quat, vel, ang_vel)
        else:
            set_drone_state(pos, quat, vel, ang_vel)

        drone_state = drone.get_state()[..., :13]
        cmd_thrust = torch.tensor([current_state['cmd.thrust']]).to(device=sim.device).float()
        cmd_rate = torch.tensor([current_state['cmd.r_rate'], current_state['cmd.p_rate'], current_state['cmd.y_rate']]).to(device=sim.device).float()
        pdb.set_trace()
        # print(cmd_thrust.unsqueeze(0) / (2**16) * max_thrust)
        action = controller(
            drone_state.squeeze(0), 
            target_rate=cmd_rate.unsqueeze(0) / 180 * torch.pi,
            target_thrust=cmd_thrust.unsqueeze(0) / (2**16) * max_thrust
        )

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
        next_state = df.loc[i]
        real_pos = torch.tensor([next_state['pos.x'], next_state['pos.y'], next_state['pos.z']])
        real_quat = torch.tensor([next_state['quat.w'], next_state['quat.x'], next_state['quat.y'], next_state['quat.z']])
        real_vel = torch.tensor([next_state['vel.x'], next_state['vel.y'], next_state['vel.z']])
        real_omega = torch.tensor([next_state['omega.r'], next_state['omega.p'], next_state['omega.y']]) # in radius / s

        pos_error.append(torch.sqrt(mse(sim_pos, real_pos)).item())
        pos_change.append(torch.sum(torch.abs(pos - real_pos)))
        pos_abs_error.append(torch.sum(torch.abs(sim_pos - real_pos)))

        vel_error.append(torch.sqrt(mse(sim_vel, real_vel)).item())
        vel_change.append(torch.sum(torch.abs(vel - real_vel)))
        vel_abs_error.append(torch.sum(torch.abs(sim_vel - real_vel)))

        quat_error.append(torch.sqrt(mse(sim_quat, real_quat)).item())
        quat_change.append(torch.sum(torch.abs(quat - real_quat)))
        quat_abs_error.append(torch.sum(torch.abs(sim_quat - real_quat)))

        omega_error.append(torch.sqrt(mse(sim_omega, real_omega)).item())
        omega_change.append(torch.sum(torch.abs(ang_vel - real_omega)))
        omega_abs_error.append(torch.sum(torch.abs(sim_omega - real_omega)))

        sim_poses.append(sim_pos)
        real_poses.append(real_pos)

        # frames_vis.append(camera_vis.get_images().cpu())

    pos_error = torch.tensor(pos_error)
    print("pos mse error", "mean", torch.mean(pos_error).item(), "std", torch.std(pos_error).item())
    pos_abs_error = torch.tensor(pos_abs_error)
    pos_change = torch.tensor(pos_change)
    pos_relative_error = pos_abs_error / pos_change
    print("pos relative error", "mean", torch.mean(pos_relative_error).item(), "std", torch.std(pos_relative_error).item())

    vel_error = torch.tensor(vel_error)
    print("vel mse error", "mean", torch.mean(vel_error).item(), "std", torch.std(vel_error).item())
    vel_abs_error = torch.tensor(vel_abs_error)
    vel_change = torch.tensor(vel_change)
    vel_relative_error = vel_abs_error / vel_change
    print("vel relative error", "mean", torch.mean(vel_relative_error).item(), "std", torch.std(vel_relative_error).item())

    quat_error = torch.tensor(quat_error)
    print("quat mse error", "mean", torch.mean(quat_error).item(), "std", torch.std(quat_error).item())
    quat_abs_error = torch.tensor(quat_abs_error)
    quat_change = torch.tensor(quat_change)
    quat_relative_error = quat_abs_error / quat_change
    print("quat relative error", "mean", torch.mean(quat_relative_error).item(), "std", torch.std(quat_relative_error).item())

    omega_error = torch.tensor(omega_error)
    print("omega mse error", "mean", torch.mean(omega_error).item(), "std", torch.std(omega_error).item())
    omega_abs_error = torch.tensor(omega_abs_error)
    omega_change = torch.tensor(omega_change)
    omega_relative_error = omega_abs_error / omega_change
    print("omega relative error", "mean", torch.mean(omega_relative_error).item(), "std", torch.std(omega_relative_error).item())

    print(episode_length)

    # from torchvision.io import write_video

    # for image_type, arrays in torch.stack(frames_vis).items():
    #     print(f"Writing {image_type} of shape {arrays.shape}.")
    #     for _, arrays_drone in enumerate(arrays.unbind(1)):
    #         if image_type == "rgb":
    #             arrays_drone = arrays_drone.permute(0, 2, 3, 1)[..., :3]
    #             write_video(f"rgb.mp4", arrays_drone, fps=1/average_dt * 2)
    #         elif image_type == "distance_to_camera":
    #             continue


    # # plot trajectory
    # import matplotlib.pyplot as plt
    # import numpy as np

    # real_poses = torch.stack(real_poses).detach().cpu().numpy()
    # sim_poses = torch.stack(sim_poses).detach().cpu().numpy()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(sim_poses[:, 0], sim_poses[:, 1], sim_poses[:, 2], s=5, label='sim')
    # ax.scatter(real_poses[:, 0], real_poses[:, 1], real_poses[:, 2], s=5, label='real')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()

    # plt.savefig('cf_goodrl')

    simulation_app.close()

if __name__ == "__main__":
    main()
