import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import hydra
from omegaconf import OmegaConf
from omni_drones import init_simulation_app
from omni_drones.utils.torch import quat_rotate, quat_rotate_inverse, euler_to_quaternion, quaternion_to_euler

# define residualnet
class ResidualNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)
        
    def forward(self, x):
        # x: [batch, chunk, dim]
        batch = x.shape[0]
        chunk = x.shape[1]
        x_dim = x.shape[-1]
        x = x.reshape(-1, x_dim).float()
        residual = torch.tanh(self.fc1(x))
        residual = torch.tanh(self.fc2(residual))
        residual = self.fc3(residual)
        return residual.reshape(batch, chunk, -1)

rosbags = [
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_large_data1.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_large_data3.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_large_data5.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_small_data1.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_small_data2.csv',
    '/home/jiayu/OmniDrones/simopt/real_data/dataset/data/cf15_small_data4.csv',
]

T = 35
prep_step = 5
skip = 1

@hydra.main(version_base=None, config_path=".", config_name="real2sim")
def main(cfg):
    """
        preprocess real data
        real_data: [batch_size, T, dimension]
    """
    real_pos_data = []
    real_vel_data = []
    real_quat_data = []
    real_rate_rpy_data = []
    real_action_data = []
    real_prep_action_data = []
    for idx in range(len(rosbags)):
        df = pd.read_csv(rosbags[idx], skip_blank_lines=True)
        preprocess_df = df[(df[['motor.m1']].to_numpy()[:,0] > 0)]
        preprocess_df = preprocess_df[:1600] # only figure 8
        pos = preprocess_df[['pos.x', 'pos.y', 'pos.z']].to_numpy()
        vel = preprocess_df[['vel.x', 'vel.y', 'vel.z']].to_numpy()
        rpy = preprocess_df[['rpy.r', 'rpy.p', 'rpy.y']].to_numpy() / 180.0 * torch.pi
        rpy[..., 1] = - rpy[..., 1] # crazyflie
        quat = euler_to_quaternion(torch.from_numpy(rpy)).numpy()
        rate_rpy = preprocess_df[['rate_rpy.r', 'rate_rpy.p', 'rate_rpy.y']].to_numpy() / 1000.0
        rate_rpy[..., 1] = - rate_rpy[..., 1] # crazyflie
        PWMs = preprocess_df[['motor.m1', 'motor.m2', 'motor.m3', 'motor.m4']].to_numpy()
        # voltages = preprocess_df[['bat']].to_numpy()
        # exclude_battery_compensation_flag = False # True: maybe nan
        # if exclude_battery_compensation_flag:
        #     PWMs = exclude_battery_compensation(PWMs, voltages)
        action = PWMs / (2**16) * 2 - 1.0
        
        episode_length = preprocess_df.shape[0]
        pos_one = []
        vel_one = []
        quat_one = []
        rate_rpy_one = []
        action_one = []
        prep_action_one = []
        for i in range(prep_step, episode_length - T, skip):
            _slice = slice(i, i + T)
            pos_one.append(pos[_slice])
            vel_one.append(vel[_slice])
            quat_one.append(quat[_slice])
            rate_rpy_one.append(rate_rpy[_slice])
            action_one.append(action[_slice])
            slc = slice(i-prep_step, i)
            prep_action_one.append(action[slc])
        pos_one = np.array(pos_one)
        vel_one = np.array(vel_one)
        quat_one = np.array(quat_one)
        rate_rpy_one = np.array(rate_rpy_one)
        action_one = np.array(action_one)
        prep_action_one = np.array(prep_action_one)
        real_pos_data.append(pos_one)
        real_vel_data.append(vel_one)
        real_quat_data.append(quat_one)
        real_rate_rpy_data.append(rate_rpy_one)
        real_action_data.append(action_one)
        real_prep_action_data.append(prep_action_one)
    real_pos_data = np.concatenate(real_pos_data, axis=0)
    real_vel_data = np.concatenate(real_vel_data, axis=0)
    real_quat_data = np.concatenate(real_quat_data, axis=0)
    real_rate_rpy_data = np.concatenate(real_rate_rpy_data, axis=0)
    real_action_data = np.concatenate(real_action_data, axis=0)
    real_prep_action_data = np.concatenate(real_prep_action_data, axis=0)
    real_data = {
        'pos': real_pos_data,
        'vel': real_vel_data,
        'quat': real_quat_data,
        'rate_rpy': real_rate_rpy_data,
        'action': real_action_data,
        'prep_action': real_prep_action_data,
    }
    # num_envs = real_pos_data.shape[0]
    num_envs = batch_size = 64
    data_size = real_pos_data.shape[0]
    #******************************************************#

    # start sim
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    import omni_drones.utils.scene as scene_utils
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.controllers import RateController, PIDRateController
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg
    from omni.isaac.cloner import GridCloner
    from omni.isaac.core.utils import prims as prim_utils, stage as stage_utils
    import omni_drones.utils.kit as kit_utils

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
    #******************************************************#
    
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
    
    def evaluate(params, real_data, predictions): # real_data: mini-batch
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
        
        sim.reset()
        drone.initialize_byTunablePara(tunable_parameters=tunable_parameters)
        controller = PIDRateController(dt, g, drone.params).to(sim.device)
        controller = controller.to(sim.device)
        
        shuffled_pos = real_data['pos']
        shuffled_vel = real_data['vel']
        shuffled_quat = real_data['quat']
        shuffled_rate_rpy = real_data['rate_rpy']
        shuffled_action = real_data['action']
        shuffled_prep_action = real_data['prep_action']
        
        chunk_length = pos.shape[1]
        
        # update simulation parameters
        """
            1. set parameters into sim
            2. update parameters
            3. export para to yaml 
        """

        sim_pos_list = []
        real_pos_list = []
        
        sim_quat_list = []
        real_quat_list = []

        sim_vel_list = []
        real_vel_list = []

        sim_ang_vel_list = []
        real_ang_vel_list = []
        
        # reset sim and execute prep_step
        for j in range(shuffled_prep_action.shape[1]):
            real_prep_action = torch.tensor(shuffled_prep_action[:, j]).to(sim.device)
            drone.apply_action(real_prep_action.unsqueeze(1))
            sim.step(render=True)
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.render()
                continue
        
        for i in range(chunk_length - 1):
            real_pos = torch.tensor(shuffled_pos[:, i]).to(sim.device)
            real_quat = torch.tensor(shuffled_quat[:, i]).to(sim.device)
            real_vel = torch.tensor(shuffled_vel[:, i]).to(sim.device)
            real_rate_rpy = torch.tensor(shuffled_rate_rpy[:, i]).to(sim.device)
            real_action = torch.tensor(shuffled_action[:, i]).to(sim.device)
            res = torch.tensor(predictions[:, i])
            # if i == 0:
            set_drone_state(real_pos, real_quat, real_vel, real_rate_rpy)
            
            drone.apply_action_residual(real_action.unsqueeze(1), res.unsqueeze(1))

            sim.step(render=True)
            
            if sim.is_stopped():
                break
            if not sim.is_playing():
                sim.render()
                continue

            # get simulated drone state
            next_sim_state = drone.get_state().squeeze(1).cpu()
            next_sim_pos = next_sim_state[..., :3] - envs_positions.cpu() # get env pos
            next_sim_quat = next_sim_state[..., 3:7]
            next_sim_vel = next_sim_state[..., 7:10]
            next_sim_ang_vel = next_sim_state[..., 10:13]

            # next real states, ground truth
            next_real_pos = torch.tensor(shuffled_pos[:, i + 1])
            next_real_quat = torch.tensor(shuffled_quat[:, i + 1])
            next_real_vel = torch.tensor(shuffled_vel[:, i + 1])
            next_real_rate_rpy = torch.tensor(shuffled_rate_rpy[:, i + 1])
            next_real_ang_vel = quat_rotate(next_real_quat, next_real_rate_rpy)

            sim_pos_list.append(next_sim_pos.cpu().detach().numpy())
            sim_quat_list.append(next_sim_quat.cpu().detach().numpy())
            sim_vel_list.append(next_sim_vel.cpu().detach().numpy())
            sim_ang_vel_list.append(next_sim_ang_vel.cpu().detach().numpy())
            
            real_pos_list.append(next_real_pos.cpu().detach().numpy())
            real_quat_list.append(next_real_quat.cpu().detach().numpy())
            real_vel_list.append(next_real_vel.cpu().detach().numpy())
            real_ang_vel_list.append(next_real_ang_vel.cpu().detach().numpy())
                
        sim_pos_list = np.array(sim_pos_list)
        sim_quat_list = np.array(sim_quat_list)
        sim_vel_list = np.array(sim_vel_list)
        sim_ang_vel_list = np.array(sim_ang_vel_list)

        real_pos_list = np.array(real_pos_list)
        real_quat_list = np.array(real_quat_list)
        real_vel_list = np.array(real_vel_list)
        real_ang_vel_list = np.array(real_ang_vel_list)
        
        # # normalization
        # min_pos_list = np.min(np.min(sim_pos_list, axis=0), axis=0)[np.newaxis, np.newaxis, :]
        # max_pos_list = np.max(np.max(sim_pos_list, axis=0), axis=0)[np.newaxis, np.newaxis, :]
        # sim_pos_list = (sim_pos_list - min_pos_list) / (max_pos_list - min_pos_list)
        # real_pos_list = (real_pos_list - min_pos_list) / (max_pos_list - min_pos_list)
        # min_vel_list = np.min(np.min(sim_vel_list, axis=0), axis=0)[np.newaxis, np.newaxis, :]
        # max_vel_list = np.max(np.max(sim_vel_list, axis=0), axis=0)[np.newaxis, np.newaxis, :]
        # sim_vel_list = (sim_vel_list - min_vel_list) / (max_vel_list - min_vel_list)
        # real_vel_list = (real_vel_list - min_vel_list) / (max_vel_list - min_vel_list)
        # min_quat_list = np.min(np.min(sim_quat_list, axis=0), axis=0)[np.newaxis, np.newaxis, :]
        # max_quat_list = np.max(np.max(sim_quat_list, axis=0), axis=0)[np.newaxis, np.newaxis, :]
        # sim_quat_list = (sim_quat_list - min_quat_list) / (max_quat_list - min_quat_list)
        # real_quat_list = (real_quat_list - min_quat_list) / (max_quat_list - min_quat_list)
        # min_ang_vel_list = np.min(np.min(sim_ang_vel_list, axis=0), axis=0)[np.newaxis, np.newaxis, :]
        # max_ang_vel_list = np.max(np.max(sim_ang_vel_list, axis=0), axis=0)[np.newaxis, np.newaxis, :]
        # sim_ang_vel_list = (sim_ang_vel_list - min_ang_vel_list) / (max_ang_vel_list - min_ang_vel_list)
        # real_ang_vel_list = (real_ang_vel_list - min_ang_vel_list) / (max_ang_vel_list - min_ang_vel_list)
        
        error_rpy = (sim_quat_list - real_quat_list)[..., :2]
        error_rpy_dot = (sim_ang_vel_list - real_ang_vel_list)[..., :2]
        error_xyz = 100 * (sim_pos_list - real_pos_list)
        error_xyz_dot = 50 * (sim_vel_list - real_vel_list)
        
        error = np.concatenate([error_rpy, error_rpy_dot, error_xyz, error_xyz_dot], axis=-1)
        # error = np.concatenate([error_xyz_dot], axis=-1)

        L1_loss = np.linalg.norm(error, axis=-1, ord=1)
        L2_loss = np.linalg.norm(error, axis=-1, ord=2)
        L = np.mean(L1_loss + L2_loss, axis=-1)
        breakpoint()
        
        loss = torch.tensor(0.0, dtype=torch.float)
        # gamma = 0.95 # discounted factor
        gamma = 1.0 # accumulative error, estimate the real throttle
        # opt trajectory = sim_length
        for i in range(chunk_length - 1):
            loss += L[i] * gamma**i
        return loss / (chunk_length - 1)

    # PID
    params = [
        0.03, 1.4e-5, 1.4e-5, 2.17e-5, 0.043,
        2.2148508468341153e-08, 2315, 7.24e-10, 0.2,
        0.02618507955424977, # Tm
        # controller
        250.0, 250.0, 120.0, # kp
        2.5, 2.5, 2.5, # kd
        500.0, 500.0, 16.7, # ki
        33.3, 33.3, 166.7 # ilimit
    ]
    for idx in range(len(params)):
        params[idx] = float(params[idx])

    #******************************************************#
    # opt for residual network
    # init residual network
    input_size = 17
    output_size = 8
    model = ResidualNetwork(input_size, output_size).to(sim.device)
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # dataset = TensorDataset(torch.cat((s_t, a_t), dim=1), s_next_prime - s_t)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # for inputs, targets in dataloader:
        shuffled_idx = torch.randperm(data_size)
        # shuffled_data
        shuffled_pos = real_data['pos'][shuffled_idx]
        shuffled_quat = real_data['quat'][shuffled_idx]
        shuffled_vel = real_data['vel'][shuffled_idx]
        shuffled_rate_rpy = real_data['rate_rpy'][shuffled_idx]
        shuffled_action = real_data['action'][shuffled_idx]
        shuffled_prep_action = real_data['prep_action'][shuffled_idx]
        x_shuffled = torch.from_numpy(np.concatenate([shuffled_pos, shuffled_quat, shuffled_vel, shuffled_rate_rpy, shuffled_action], axis=-1)).to(sim.device)
        for idx in range(data_size // batch_size):
            batch_data = x_shuffled[idx * batch_size: (idx + 1) * batch_size]
            batch_real_data = {
                'pos': shuffled_pos[idx * batch_size: (idx + 1) * batch_size],
                'quat': shuffled_quat[idx * batch_size: (idx + 1) * batch_size],
                'vel': shuffled_vel[idx * batch_size: (idx + 1) * batch_size],
                'rate_rpy': shuffled_rate_rpy[idx * batch_size: (idx + 1) * batch_size],
                'action': shuffled_action[idx * batch_size: (idx + 1) * batch_size],
                'prep_action': shuffled_prep_action[idx * batch_size: (idx + 1) * batch_size]
            }
            
            # prediction: f_res, torque_res
            predictions = model(batch_data)
            
            loss = evaluate(params, batch_real_data, predictions)
            
            # opt
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    main()