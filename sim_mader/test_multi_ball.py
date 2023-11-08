#!/home/zanghongzhi/.local/share/ov/pkg/isaac_sim-2022.2.0/python.sh

import os

from typing import Dict, Optional
import torch
import numpy as np
from functorch import vmap

import sys
dir_name = os.path.dirname(os.path.abspath(__file__))
OMNIDRONES_ENV_DIR = os.path.dirname(dir_name)
sys.path.append(OMNIDRONES_ENV_DIR)

import hydra
from omegaconf import OmegaConf
from omni_drones import CONFIG_PATH, init_simulation_app
from tensordict import TensorDict
from geometry_msgs.msg import PointStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_about_axis, quaternion_multiply, random_quaternion
import rospy
from snapstack_msgs.msg import State, Goal
from visualization_msgs.msg import Marker, MarkerArray
from mader_msgs.msg import DynTraj
import tf
from omni_drones.utils.poisson_disk import poisson_disk_sampling
from math import sin 



RADIUS = 1.0
HEIGHT = 1.5

FORMATION = [
    [0., 0., HEIGHT],
    [0., 1., HEIGHT], 
    [0.951, 0.309, HEIGHT],
    [0.588, -0.809, HEIGHT],
    [-0.588, -0.809, HEIGHT],
    [-0.951, 0.309, HEIGHT]
]


def wave_in_z(x,y,z,scale=1.2, offset=0., slower=0.4):
    tt='t/' + str(slower)+'+';
    x_string=str(float(x));
    y_string=str(float(y))
    z_string=str(scale)+'*(sin( '+tt +str(offset)+'))' + '+' + str(float(z));                     
    return [x_string, y_string, z_string]


def wave(x,y,z, scale = [0., 0., 1.2], offset=[0., 0., 0.], slower=0.5):
    tt='t/' + str(slower)+'+';
    x_string=str(scale[0])+'*(sin( '+tt +str(offset[0])+'))' + '+' +str(float(x));
    y_string=str(scale[1])+'*(sin( '+tt +str(offset[1])+'))' + '+' +str(float(y))
    z_string=str(scale[2])+'*(sin( '+tt +str(offset[2])+'))' + '+' + str(float(z));                     
    return [x_string, y_string, z_string]


def parabola(x, y, z, vx, vy, vz, period):
    tt = 't %' + str(float(period))
    x_string = str(float(x)) + str(float(vx)) + '*' + tt
    y_string = str(float(y)) + str(float(vy)) + '*' + tt
    z_string = str(float(z)) + str(float(vz)) + '*' + tt - '0.5 * 0.981 *' + tt + '**2' 
    # pass
    return [x_string, y_string, z_string]


def get_yaw_from_goal(goal: Goal):
    thrust=np.array([goal.a.x, goal.a.y, goal.a.z + 9.81]); 
    thrust_normalized=thrust/np.linalg.norm(thrust);

    a=thrust_normalized[0];
    b=thrust_normalized[1];
    c=thrust_normalized[2];
    qabc = random_quaternion();
    tmp=(1/np.sqrt(2*(1+c)));
    #From http://docs.ros.org/en/jade/api/tf/html/python/transformations.html, the documentation says
    #"Quaternions ix+jy+kz+w are represented as [x, y, z, w]."
    qabc[3] = tmp*(1+c) #w
    qabc[0] = tmp*(-b)  #x
    qabc[1] = tmp*(a)   #y
    qabc[2] = 0         #z

    qpsi = random_quaternion();
    qpsi[3] = np.cos(goal.psi/2.0);  #w
    qpsi[0] = 0;                       #x 
    qpsi[1] = 0;                       #y
    qpsi[2] = np.sin(goal.psi/2.0);  #z

    w_q_b=quaternion_multiply(qabc,qpsi)

    _, _, yaw = euler_from_quaternion(w_q_b, axes='rxyz')

    return yaw


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
    # pub_rviz_pos_list[i-1].publish(marker)


def getBallRvizMarker(idx, pos, height, ball_radius):
    marker=Marker();
    marker.id=idx;
    marker.ns="mesh";
    marker.header.frame_id="world"
    marker.type=Marker.SPHERE;
    marker.action=marker.ADD;

    marker.pose.position.x=pos[0]
    marker.pose.position.y=pos[1]
    marker.pose.position.z=height
    marker.pose.orientation.x=0.0;
    marker.pose.orientation.y=0.0;
    marker.pose.orientation.z=0.0;
    marker.pose.orientation.w=1.0;
    marker.color.r, marker.color.g, marker.color.b = (0.4, 0.8, 0.)
    marker.color.a = 1.
    # marker.mesh_use_embedded_materials=True
    # marker.mesh_resource = "package://mader/meshes/ConcreteDamage01b/model4.dae"
       
    marker.lifetime = rospy.Duration.from_sec(0.0);
    marker.scale.x = marker.scale.y = marker.scale.z = ball_radius*2
    return marker


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    from omni.isaac.core.objects import DynamicSphere, DynamicCuboid
    from omni_drones.envs.forest import Forest
    import omni_drones.utils.scene as scene_utils
    import omni_drones.utils.kit as kit_utils
    from omni_drones.views import RigidPrimView
    from omni.isaac.core.simulation_context import SimulationContext
    from omni_drones.controllers import LeePositionController
    from omni_drones.robots.drone import MultirotorBase
    from omni_drones.utils.torch import euler_to_quaternion, quaternion_to_euler
    from omni_drones.sensors.camera import Camera, PinholeCameraCfg
    import dataclasses

    sim = SimulationContext(
        stage_units_in_meters=1.0,
        physics_dt=0.01,
        rendering_dt=0.01,
        sim_params=cfg.sim,
        backend="torch",
        device=cfg.sim.device,
    )
    n = 6

    drone_model = "Hummingbird"
    drone_cls = MultirotorBase.REGISTRY[drone_model]
    drone = drone_cls()

    # 无人机起始位置
    translations_list = []
    for i in range(n):
        # translations_list.append([0., -4.0 + i * 2.0,  HEIGHT])
        translations_list.append(FORMATION[i])
    # translations = torch.tensor([[0., 0., height], [1., 0., height], [0., 1., height], [1., 1., height]])
    translations = torch.tensor(translations_list)
    drone.spawn(translations=translations)

    scene_utils.design_scene()

    # marker_array_static_mesh, bbox = construct_forest(cfg)
    # tree_num = len(marker_array_static_mesh.markers)
    # print(f"tree_num = {tree_num}")

    sim.reset()
    drone.initialize()

    # create a position controller
    # note: the controller is state-less (but holds its parameters)
    controller = LeePositionController(g=9.81, uav_params=drone.params).to(sim.device)

    root_state = torch.zeros(n, 13, device=sim.device)
    # target_pos_list = []
    # for i in range(n):
    #     target_pos_list.append([0., i * 1.0,  height])
    # target_pos = torch.tensor([[0., 0., height], [1., 0., height], [0., 1., height], [1., 1., height]], device=sim.device)
    # target_pos = torch.tensor(target_pos_list, device=sim.device)
    target_pos = translations.to(device=sim.device)
    target_yaw = torch.zeros(n, device=sim.device)
    target_vel = torch.zeros_like(target_pos)
    target_acc = torch.zeros_like(target_vel)

    # target_pos =torch.tensor([[FORMATION[i][0],FORMATION[i][1]+35., FORMATION[i][2]] for i in range(n)], device=sim.device)


    # ball_num = 5
    ball_row = 5
    ball_col = 5
    ball_num = ball_row * ball_col
    ball_list = []
    ball_pos = []
    ball_radius = 0.4
    # marker_array_static_mesh = MarkerArray()
    ball_height = 1.5
    ball_offset_list = []
    ball_scale_list = []
    def init_balls(ball_row, ball_col): 
        for i in range(ball_row):
            for j in range(ball_col):
                pos = [-4 + i*2., 5 + j*3.]
                # pos = [0., 5 + i*2]
                ball_prim_path = f"/World/envs/env_0/ball_{i*ball_col + j}"
                ball = DynamicSphere(
                    prim_path = ball_prim_path,
                    name = "ball",
                    radius = ball_radius,
                    color = torch.tensor([0.4, 0.8, 0.])
                )
                kit_utils.set_rigid_body_properties(
                    ball_prim_path, disable_gravity=True
                )
                # kit_utils.set_collision_properties(
                #     ball_prim_path, collision_enabled=True
                # )
                ball.set_world_pose(position = [*pos, ball_height])
                ball.set_mass(3.)
                ball_list.append(ball)
                ball_pos.append(pos)
                offset = np.random.rand(3) * np.pi * 2
                ball_offset_list.append(offset)
                x_scale = np.random.rand() * 1.
                y_scale = np.random.rand() * 1.2
                z_scale = 1.4
                ball_scale_list.append([x_scale, y_scale, z_scale])

        # for i in range(ball_num):
        #     # pos = [0., 0.,]
        #     pos = [0., 5 + i*2]
        #     ball_prim_path = f"/World/envs/env_0/ball_{i}"
        #     ball = DynamicSphere(
        #         prim_path = ball_prim_path,
        #         name = "ball",
        #         radius = ball_radius,
        #         color = torch.tensor([0.4, 0.8, 0.])
        #     )
        #     kit_utils.set_rigid_body_properties(
        #         ball_prim_path, disable_gravity=True
        #     )
        #     # kit_utils.set_collision_properties(
        #     #     ball_prim_path, collision_enabled=True
        #     # )
        #     ball.set_world_pose(position = [*pos, ball_height])
        #     ball.set_mass(1.)
        #     ball_list.append(ball)

        #     ball_pos.append(pos)
    
    init_balls(ball_row, ball_col)


    from omni.isaac.debug_draw import _debug_draw
    marker_colors = torch.rand((n, 3))
    goal_draw = _debug_draw.acquire_debug_draw_interface()
    # goal_color = (0.8, 0.8, 0.8, 1.)
    goal_color = torch.clamp(marker_colors - 0.2, min = 0., max = 1.)
    goal_draw_size=5. 
    class GoalChange:
        def __init__(self, drone_id):
            self.drone_id = drone_id

        def change_goal(self, goal: Goal):
            # print(f"goal = \n{goal}")
            pos = torch.tensor([goal.p.x, goal.p.y, goal.p.z,], device=sim.device)
            # yaw = get_yaw_from_goal(goal)
            yaw = goal.psi
            # print(f"idx = {self.drone_id}, goal.p={goal.p}")
            
        
            target_pos[self.drone_id-1, :] = pos
            target_yaw[self.drone_id-1] = yaw
            target_vel[self.drone_id-1, :] = torch.tensor([goal.v.x, goal.v.y, goal.v.z,])
            target_acc[self.drone_id-1, :] = torch.tensor([goal.a.x, goal.a.y, goal.a.z,])


    # # rospy.Subscriber('/SQ01s/goal', Goal, change_goal)
    rospy.init_node('issac_sim')
    # rospy.Subscriber('/SQ01s/mader/point_G', PointStamped, change_goal)
    goal_change_list : list[GoalChange] = []
    pub_state_list : list[rospy.Publisher] = []
    pub_rviz_pos_list : list[rospy.Publisher] = []
    
    # colors = []

    r = 0.3
    for i in range(1, n+1):
        goal_change_list.append(GoalChange(i))
        rospy.Subscriber('/SQ0' + str(i) + 's/goal', Goal, goal_change_list[i-1].change_goal)
        pub_state_list.append(rospy.Publisher('/SQ0' + str(i) + 's/state', State, queue_size=10, latch=True))
        pub_rviz_pos_list.append(rospy.Publisher('/SQ0' + str(i) + 's/pos', Marker, queue_size=10))

    # obstacles traj
    pubTraj = rospy.Publisher('/trajs', DynTraj, queue_size=ball_num)
    # Rviz for tree
    # pubShapes_static_mesh = rospy.Publisher('/shapes_static_mesh', MarkerArray, queue_size=1, latch=True)
    pubShapes_dynamic_mesh = rospy.Publisher('/shapes_dynamic_mesh', MarkerArray, queue_size=1, latch=True)

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
    


    # pos_lists = [[] for i in range(n)]
    # sizes = [[] for i in range(n)]
    # colors = [[] for i in range(n)]
    size = 10
    draw = _debug_draw.acquire_debug_draw_interface()
    def step():        
        root_state[:, :] = drone.get_state()[..., :13].squeeze(0)
        # print(f"target yaw = {target_yaw}")
        action = controller(root_state, 
                            target_pos=target_pos,
                            target_vel=target_vel,
                            target_yaw=target_yaw,
                            )
        # print(target_pos)
        drone.apply_action(action)
        sim.step(render=True)
        # print(f"mass= {ball_list[0].get_mass()}")

        for i in range(1, n+1):
            statemsg = State()    
            statemsg = fill_statemsg(statemsg, root_state[i-1])
            pub_state_list[i-1].publish(statemsg)
            
            # publish pos in Rviz
            marker = getRvizPosMarker(statemsg, idx=i, 
                                      marker_color=marker_colors[i-1], 
                                      r=r)
            pub_rviz_pos_list[i-1].publish(marker)

            # debug_draw in IsaacSim
            # raise NotImplementedError()
            # pos_lists[i-1].append((statemsg.pos.x, statemsg.pos.y, statemsg.pos.z))
            color_list = marker_colors[i-1].tolist()
            color_list.append(1.)
            # colors[i-1].append(tuple(color_list))
            # sizes[i-1].append(size)
            draw.draw_points([(statemsg.pos.x, statemsg.pos.y, statemsg.pos.z)], [tuple(color_list)], [size])
            goal_color_list = list(goal_color[i - 1])
            goal_color_list.append(1.)
            goal_draw.draw_points([tuple(target_pos[i-1].tolist())], [tuple(goal_color_list)], [goal_draw_size])
        
        marker_array_dynamic_mesh = MarkerArray()
        for j in range(ball_num):
            t_ros = rospy.Time.now()
            t = rospy.get_time()
            dynamic_trajectory_msg=DynTraj()
            bbox_j = [ball_radius*2, ball_radius*2, ball_radius*2]
            # [x_string, y_string, z_string] = wave_in_z(x=ball_pos[j][0], y=ball_pos[j][1], z=ball_height, offset=ball_offset_list[j])
            [x_string, y_string, z_string] = wave(x=ball_pos[j][0], y=ball_pos[j][1], z=ball_height, scale = ball_scale_list[j], offset=ball_offset_list[j])
            # print(z_string)
            x = eval(x_string)
            y = eval(y_string)
            z = eval(z_string)
            dynamic_trajectory_msg.bbox = bbox_j
            dynamic_trajectory_msg.is_agent = False
            dynamic_trajectory_msg.header.stamp = t_ros
            dynamic_trajectory_msg.function = [x_string, y_string, z_string]
            # print(f"func = {dynamic_trajectory_msg.function}")
            
            dynamic_trajectory_msg.pos.x = x
            dynamic_trajectory_msg.pos.y = y
            dynamic_trajectory_msg.pos.z = z
            dynamic_trajectory_msg.id = 4000 + j
            pubTraj.publish(dynamic_trajectory_msg)

            marker = getBallRvizMarker(idx=j, pos=[x, y], height=z, ball_radius=ball_radius)
            marker_array_dynamic_mesh.markers.append(marker)
            ball_list[j].set_world_pose(position = [x, y, z])

        # ball in Rviz
        pubShapes_dynamic_mesh.publish(marker_array_dynamic_mesh)

            



    # cnt = 0
    while not rospy.is_shutdown():
        step()

    simulation_app.close()


if __name__ == "__main__":
    main()