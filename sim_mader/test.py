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
from visualization_msgs.msg import Marker

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="train")
def main(cfg):
    OmegaConf.resolve(cfg)
    simulation_app = init_simulation_app(cfg)
    print(OmegaConf.to_yaml(cfg))

    # from omni.isaac.core.objects import DynamicSphere, DynamicCuboid
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
    n = 4

    drone_model = "Hummingbird"
    drone_cls = MultirotorBase.REGISTRY[drone_model]
    drone = drone_cls()

    translations = torch.tensor([[0., 0., 1.5]])
    drone.spawn(translations=translations)

    scene_utils.design_scene()

    camera_cfg = PinholeCameraCfg(
        sensor_tick=0,
        resolution=(320, 240),
        data_types=["rgb", "distance_to_camera"],
    )

    # cameras used as sensors

    # camera_sensor.spawn([
    #     f"/World/envs/env_0/{drone.name}_{i}/base_link/Camera" 
    #     for i in range(n)
    # ])
    # camera for visualization
    # camera_vis = Camera(dataclasses.replace(camera_cfg, resolution=(960, 720)))

    sim.reset()
    # camera_sensor.initialize(f"/World/envs/env_0/{drone.name}_*/base_link/Camera")
    # camera_vis.initialize("/OmniverseKit_Persp")
    drone.initialize()

    # create a position controller
    # note: the controller is state-less (but holds its parameters)
    controller = LeePositionController(g=9.81, uav_params=drone.params).to(sim.device)

    root_state = torch.zeros(1, 13, device=sim.device)
    target_pos = torch.tensor([0., 0., 1.5], device=sim.device)
    target_yaw = torch.zeros(1, device=sim.device)
    target_vel = torch.zeros_like(target_pos)
    target_acc = torch.zeros_like(target_vel)


    def change_goal(goal: Goal):
        # print(f"goal = \n{goal}")
        pos = torch.tensor([[goal.p.x, goal.p.y, goal.p.z,]], device=sim.device)
        # yaw = get_yaw_from_psi(goal)
        # quat = root_state[0, 3:7].tolist()
        # rpy = euler_from_quaternion(quat)
        # print(goal.psi, rpy)
        target_pos[:] = pos
        target_yaw[:] = goal.psi
        target_vel[:] = torch.tensor([[goal.v.x, goal.v.y, goal.v.z,]])
        target_acc[:] = torch.tensor([[goal.a.x, goal.a.y, goal.a.z,]])


    # rospy.Subscriber('/SQ01s/goal', Goal, change_goal)
    rospy.init_node('issac_sim')
    # rospy.Subscriber('/SQ01s/mader/point_G', PointStamped, change_goal)
    rospy.Subscriber('/SQ01s/goal', Goal, change_goal)
    pub_state = rospy.Publisher('/SQ01s/state', State, queue_size=10, latch=True)
    pub_rviz_pos = rospy.Publisher('/SQ01s/pos', Marker, queue_size=10)
    # pub_tf = rospy.Publisher('', )

    # def handle_state(statemsg: State, name):
    #     br = tf.TransformBroadcaster()
    #     br.sendTransform((statemsg.pos.x, statemsg.pos.y, statemsg.z), 
    #                      )
    # rospy.Subscriber('?', State, handle_state, '??')




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
    
    def step():
        root_state[:] = drone.get_state()[..., :13].squeeze(0)
        action = controller(root_state, 
                            target_pos=target_pos,
                            target_vel=target_vel,
                            target_yaw=target_yaw,
                            )
        drone.apply_action(action)
        sim.step(render=True)

        statemsg = State()
        statemsg = fill_statemsg(statemsg, root_state.squeeze())
        pub_state.publish(statemsg)

        # # publish TF in Rviz
        # br = tf.TransformBroadcaster()
        # br.sendTransform((statemsg.pos.x, statemsg.pos.y, statemsg.pos.z),
        #                  (statemsg.quat.x, statemsg.quat.y, statemsg.quat.z, statemsg.quat.w),
        #                  rospy.Time.now(),
        #                  "SQ01s",
        #                  "vicon")

        # publish pos in TF by sphere
        marker = Marker()
        marker.header.frame_id = "world";
        marker.header.stamp = rospy.Time.now();

        # // Set the namespace and id for this marker.  This serves to create a unique ID
        # // Any marker sent with the same namespace and id will overwrite the old one
        # marker.ns = "basic_shapes";
        marker.id = 0;

        # // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
        marker.type = Marker.SPHERE;

        # // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
        marker.action = Marker.ADD;

        # // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
        marker.pose.position.x = statemsg.pos.x;
        marker.pose.position.y = statemsg.pos.y;
        marker.pose.position.z = statemsg.pos.z;
        marker.pose.orientation.x = 0.0;
        marker.pose.orientation.y = 0.0;
        marker.pose.orientation.z = 0.0;
        marker.pose.orientation.w = 1.0;

        # // Set the scale of the marker -- 1x1x1 here means 1m on a side
        r = 0.25
        marker.scale.x = r;
        marker.scale.y = r;
        marker.scale.z = r;

        # // Set the color -- be sure to set alpha to something non-zero!
        marker.color.r = 0.8;
        marker.color.g = 1.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0;

        marker.lifetime = rospy.Duration();
        pub_rviz_pos.publish(marker)





    while not rospy.is_shutdown():
        # print(f"receive goal = {received_goal}")
        step()
        # rate.sleep()

    simulation_app.close()


if __name__ == "__main__":
    main()