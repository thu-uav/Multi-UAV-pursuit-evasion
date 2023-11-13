#!$ISAACSIM_PATH/python.sh

import logging
import os
import time

import hydra
import torch
import numpy as np
import wandb
from functorch import vmap
from omegaconf import OmegaConf

import sys
dir_name = os.path.dirname(os.path.abspath(__file__))
OMNIDRONES_ENV_DIR = os.path.dirname(dir_name)
sys.path.append(OMNIDRONES_ENV_DIR)

from omni_drones import CONFIG_PATH, init_simulation_app
# from omni_drones.utils.torchrl.transforms import (
#     DepthImageNorm,
#     LogOnEpisode, 
#     # FromMultiDiscreteAction, 
#     # FromDiscreteAction,
#     # flatten_composite,
#     # VelController,
# )
from omni_drones.utils.wandb import init_wandb

from setproctitle import setproctitle
from torchrl.envs.transforms import (
    TransformedEnv, 
    InitTracker, 
    Compose,
    CatTensors,
    StepCounter,
)
from tensordict import TensorDict
import torch

from tqdm import tqdm

import rospy
# from quadrotor_msgs.msg.PositionCommand import PositionCommand
# from nav_msgs.msg import Odometry, GridCells
# from geometry_msgs.msg import PoseStamped , TransformStamped, PointStamped, Point
from snapstack_msgs.msg import State, Goal
from visualization_msgs.msg import Marker
from mader_msgs.msg import DynTraj
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_about_axis, quaternion_multiply, random_quaternion

from tf2_msgs.msg import TFMessage
from tf.transformations import *

GRAVITY = 9.81
FITTING_NUM = 3

class Every:
    def __init__(self, func, steps):
        self.func = func
        self.steps = steps
        self.i = 0

    def __call__(self, *args, **kwargs):
        if self.i % self.steps == 0:
            self.func(*args, **kwargs)
        self.i += 1


def getRvizPosMarker(statemsg: State, idx: int, marker_color, r): 
    marker = Marker()
    marker.header.frame_id="world"
    marker.header.stamp = rospy.Time.now()
    
    marker.id = idx
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = statemsg.pos.x
    marker.pose.position.y = statemsg.pos.y
    marker.pose.position.z = statemsg.pos.z
    marker.pose.orientation.x = 0.
    marker.pose.orientation.y = 0.
    marker.pose.orientation.z = 0.
    marker.pose.orientation.w = 1.
    
    marker.scale.x = marker.scale.y = marker.scale.z = r
    marker.color.r, marker.color.g, marker.color.b = marker_color
    marker.color.a = 1.
    marker.lifetime = rospy.Duration()
    return marker


def fill_statemsg(statemsg : State, state):
        statemsg.pos.x = state[0]
        statemsg.pos.y = state[1]
        statemsg.pos.z = state[2]
        statemsg.quat.x = state[4]
        statemsg.quat.y = state[5]
        statemsg.quat.z = state[6]
        statemsg.quat.w = state[3]
        statemsg.vel.x = state[7]
        statemsg.vel.y = state[8]
        statemsg.vel.z = state[9]
        return statemsg


def solve_param(xyz_list, t_list, t_throw=None):
    t_0=t_list[-1]
    if t_throw is None:
        t_throw = t_0
    t_array = np.array([[1., (t-t_throw)] for t in t_list])
    b_array = np.array([[x, y, z+0.5*GRAVITY*(t_list[i]-t_throw)**2] for i, [x, y, z] in enumerate(xyz_list)])
    result = np.linalg.lstsq(t_array, b_array)[0]
    pos, vel = result
    return pos, vel, t_0


def fitting_parabola(x, y, z, vx, vy, vz, t_0, t_ref = 0.):
    tt = '(t - T_REF - ' + str(float(t_0)) + ')'
    # tt = 't'
    x_string = str(float(x)) + ' + (' + str(float(vx)) + ') * ' + tt
    y_string = str(float(y)) + ' + (' + str(float(vy)) + ') * ' + tt
    z_string = str(float(z)) + ' + (' + str(float(vz)) + ') * ' + tt + ' - 0.5 * '+ str(float(GRAVITY)) + ' *' + tt + '**2' 
    # pass
    tc = '(t - ' + str(float(t_ref)) + '-' + str(float(t_0))  + ')'
    x_string_c = str(float(x)) + ' + (' + str(float(vx)) + ') * ' + tc
    y_string_c = str(float(y)) + ' + (' + str(float(vy)) + ') * ' + tc
    z_string_c = str(float(z)) + ' + (' + str(float(vz)) + ') * ' + tc + ' - 0.5 * '+ str(float(GRAVITY)) + ' * pow(' + tc + ', 2)' 
    return [x_string, y_string, z_string], [x_string_c, y_string_c, z_string_c]



@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.wandb.mode = "disabled"
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from omni.isaac.core.utils import extensions, stage
    extensions.enable_extension("omni.isaac.ros_bridge")
    simulation_app.update()

    # Note that this is not the system level rospy, but one compiled for omniverse
    import rosgraph
    from rosgraph_msgs.msg import Clock
    import rospy

    from omni_drones.envs.isaac_env import IsaacEnv

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    cfg.env.num_envs = 1
    base_env = env_class(cfg, headless=cfg.headless)

    def log(info):
        print(OmegaConf.to_yaml(info))
        # run.log(info)


    transforms = [InitTracker()] #, logger]


    from omni_drones.controllers import LeePositionController
    controller = LeePositionController(
                9.81, 
                base_env.drone.params
            ).to(base_env.device)
    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    frames = []
    frame_ind = []

    def record_frame():
        frame = env.base_env.render(mode="rgb_array")
        frames.append(frame)


    base_env.enable_render(True)
    env.eval()
    env.reset()

    num_drones = env.base_env.drone.n

    global td, target_pos, target_vel, target_acc, target_yaw, simulation_time
    target_pos = torch.zeros(num_drones, 3, device=env.device)
    target_vel = torch.zeros_like(target_pos)
    target_acc = torch.zeros_like(target_pos)
    target_yaw = torch.zeros(num_drones, 1, device=env.device)
    target_pos = td[("info", 'drone_state')][..., :3].squeeze(dim=0)
    simulation_time = time.time()
    td = env.reset()
    
    class RosDrone:
        def __init__(self, idx) -> None:
            self.idx = idx
            # rospy.Subscriber('/SQ01s/goal', Goal, change_action_1)
            self.subscriber = rospy.Subscriber(f'/SQ0{self.idx}s/goal', Goal, self.call_back)
            pass
       
        
        def call_back(self, position: Goal):
            global is_running
            is_running = True
            target_pos[self.idx][0] = position.p.x
            target_pos[self.idx][1] = position.p.y
            target_pos[self.idx][2] = position.p.z
            target_vel[self.idx][0] = position.v.x
            target_vel[self.idx][1] = position.v.y
            target_vel[self.idx][2] = position.v.z
            target_acc[self.idx][0] = position.a.x
            target_acc[self.idx][1] = position.a.y
            target_acc[self.idx][2] = position.a.z
            target_yaw[self.idx][:] = position.psi
            pass
    

    global ball_xyz_list, ball_t_list
    ball_xyz_list = []
    ball_t_list = []
    # ball_r = env.ball
    def step(cnt):
        global td, target_pos, target_vel, target_acc, target_yaw, is_running
        root_state = td[("info", 'drone_state')].squeeze(0)
        global ball_xyz_list, ball_t_list

        for i in range(num_drones):
            state = State()
            state = fill_statemsg(statemsg=state, state=td[("info", 'drone_state')][0][i])
            pub_state_list[i].publish(state)


        if not is_running:
            time.sleep(0.1)
            return cnt
    

        action = controller(root_state=root_state, 
                            target_pos=target_pos,
                            target_vel=target_vel,
                            target_acc=target_acc,
                            target_yaw=target_yaw,
                            )
        if base_env.num_envs == 1:
            action = action.unsqueeze(0)
        
        td = td.update({("agents", "action"): action})
        td = env.step(td)
        # print(td.keys())
        
        
        record_frame()

        for i in range(num_drones):
            state = State()
            state = fill_statemsg(state, td[("info", 'drone_state')][0][i])
            pub_state_list[i].publish(state)
            # print(f"state.vel = {state.vel}")
            # print(f"state.acc = {state.a}")
        
        t = rospy.get_time()
        t_ros = rospy.Time.now()
        
        ball_pos = env.ball.get_world_poses().squeeze()
        print(ball_pos.shape, ball_pos)
        # raise NotImplementedError()
        if not ball_pos[1] == -20. and ball_pos[2] > 0.1:
            ball_xyz_list.append([float(ball_pos[i]) for i in range(len(ball_pos))])
            ball_t_list.append(t - T_REF)
        if len(ball_xyz_list) > FITTING_NUM:
            dynamic_trajectory_msg = DynTraj()
            ball_xyz_list = ball_xyz_list[-FITTING_NUM:]
            ball_t_list = ball_t_list[-FITTING_NUM:]
            assert(len(ball_t_list) == len(ball_xyz_list))
            # print("before solving")
            solve_pos, solve_vel, t_0 = solve_param(ball_xyz_list, ball_t_list)
            # print("after solving")
            [x_string, y_string, z_string], [x_string_c, y_string_c, z_string_c] \
                = fitting_parabola(x=solve_pos[0], y=solve_pos[1], z =solve_pos[2], vx=solve_vel[0], vy=solve_vel[1], vz=solve_vel[2], t_0 = t_0, t_ref = T_REF)
            # print("before eval")
            x = eval(x_string)
            y = eval(y_string)
            z = eval(z_string)
            # import omni.usd
            # stage = omni.usd.get_context().get_stage()
            ball_pos, ball_rot = env.get_env_poses(env.ball.get_world_poses())
            print(ball_pos)
            print(f"x={x}, y={y}, z={z}")
            print(f"xyz_list = {ball_xyz_list}")
            print(f"t_list = {ball_t_list}")
            dynamic_trajectory_msg.bbox = [0.15, 0.15, 0.15]
            dynamic_trajectory_msg.is_agent = False
            dynamic_trajectory_msg.header.stamp = t_ros
            dynamic_trajectory_msg.function = [x_string_c, y_string_c, z_string_c]
            # print(f"func = {dynamic_trajectory_msg.function}")
            
            dynamic_trajectory_msg.pos.x = x
            dynamic_trajectory_msg.pos.y = y
            dynamic_trajectory_msg.pos.z = z
            dynamic_trajectory_msg.id = 4000 + 1
            pubTraj.publish(dynamic_trajectory_msg)
        
                
        # send_obstacles(td[('next','drone.obs', 'abs_ball')][0][0][0][0])
        # print(cnt)
        cnt = cnt + 1

        if len(frames) > 800:
            video_array = np.stack(frames).transpose(0, 3, 1, 2)
            print(video_array.shape)
            video = wandb.Video(
                video_array, fps=1 / cfg.sim.dt, format="mp4"
            )
            wandb.log({"video": video})

            simulation_app.close()
            exit(0)
        
        return cnt
    
    
    rospy.init_node('sim', anonymous = True)
    state = State()
    subscriber_list = []
    for i in range(num_drones):
        subscriber = RosDrone(idx=i)
        subscriber_list.append(subscriber)
        
    pub_state_list : list[rospy.Publisher] = []
    pub_rviz_pos_list : list[rospy.Publisher] = []
    marker_colors = torch.rand((num_drones, 3))
    marker_r = 0.3

    # pub_state_list.append(rospy.Publisher('/SQ05s/state', State, queue_size=10, latch=True))
    
    for i in range(num_drones):
        pub_state_list.append(rospy.Publisher('/SQ0' + str(i) + 's/state', State, queue_size=10, latch=True))
        pub_rviz_pos_list.append(rospy.Publisher('/SQ0' + str(i) + 's/pos', Marker, queue_size=10))


    for i in range(num_drones):
        state = fill_statemsg(state, td[("info", 'drone_state')][0][i])
        pub_state_list[i].publish(state)
        marker = getRvizPosMarker(state, idx=i, 
                                marker_color=marker_colors[i], 
                                r=marker_r)
        pub_rviz_pos_list[i].publish(marker)

    pubTraj = rospy.Publisher('/trajs', DynTraj, queue_size=1)
    
    print("all nodes has been initialized")

    global is_running
    is_running = False
    cnt = 0
    
    T_REF = rospy.get_time()
    while True:
        cnt = step(cnt)


if __name__ == "__main__":
    main()