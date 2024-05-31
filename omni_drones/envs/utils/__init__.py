# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Dict, Any, Sequence, Union
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
# import omni.kit.commands
import omni.usd.commands
from pxr import UsdGeom, Usd, UsdPhysics, Gf

import torch
import omni_drones.utils.kit as kit_utils

def create_obstacle(
    prim_path: str,
    prim_type: str,
    translation: Sequence[float],
    attributes: Dict,
):
    prim = prim_utils.create_prim(
        prim_path=prim_path,
        prim_type=prim_type,
        translation=translation,
        attributes=attributes
    )
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    prim.GetAttribute("physics:kinematicEnabled").Set(True)
    kit_utils.set_collision_properties(
        prim_path, contact_offset=0.02, rest_offset=0
    )

    return prim

DEFAULT_JOINT_ATTRIBUTES = {
    "limit:rotX:physics:low": -120,
    "limit:rotX:physics:high": 120, 
    "limit:rotY:physics:low": -120,
    "limit:rotY:physics:high": 120,
    "drive:rotX:physics:damping": 2e-6,
    "drive:rotY:physics:damping": 2e-6
}

def create_bar(
    prim_path: str,
    length: float,
    from_prim: Union[str, Usd.Prim]=None,
    to_prim: Union[str, Usd.Prim]=None,
    joint_from_attributes=None,
    joint_to_attributes=None,
):

    bar = prim_utils.create_prim(prim_path)
    seg_0 = prim_utils.create_prim(
        f"{prim_path}/seg_0", 
        "Capsule",
        translation=(0., 0., -length/2),
        attributes={"radius": 0.01, "height": length}
    )
    seg_1 = prim_utils.create_prim(
        f"{prim_path}/seg_1", 
        "Capsule", 
        translation=(0., 0., -length/2),
        attributes={"radius": 0.01, "height": length}
    )
    for seg in [seg_0, seg_1]:
        UsdPhysics.RigidBodyAPI.Apply(seg)
        UsdPhysics.CollisionAPI.Apply(seg)
        massAPI = UsdPhysics.MassAPI.Apply(seg)
        massAPI.CreateMassAttr().Set(0.001)
        
    stage = prim_utils.get_current_stage()
    joint = script_utils.createJoint(stage, "Prismatic", seg_0, seg_1)
    UsdPhysics.DriveAPI.Apply(joint, "linear")
    
    joint.GetAttribute("physics:axis").Set("Z")
    joint.GetAttribute("physics:upperLimit").Set(1.)
    joint.GetAttribute("physics:lowerLimit").Set(-1.)
    joint.GetAttribute("drive:linear:physics:damping").Set(10.0)
    joint.GetAttribute("drive:linear:physics:stiffness").Set(1000.0)
    
    def setup_joint(joint, attributes):
        drives = set([key.split(":")[1] for key in attributes.keys() if key.startswith("drive")])
        for drive in drives:
            UsdPhysics.DriveAPI.Apply(joint, drive)
        for k, v in attributes.items():
            joint.GetAttribute(k).Set(v)

    if from_prim is not None:
        if isinstance(from_prim, str):
            from_prim = prim_utils.get_prim_at_path(from_prim)
        joint_from = script_utils.createJoint(stage, "D6", seg_0, from_prim)
        if joint_from_attributes is None:
            joint_from_attributes = DEFAULT_JOINT_ATTRIBUTES
        setup_joint(joint_from, joint_from_attributes)
        # omni.usd.commands.MovePrimCommand(
        #     path_from=from_prim.GetPath().pathString + "/D6Joint",
        #     path_to=prim_path + "/joint_from",
        #     keep_world_transform=False,
        #     time_code=Usd.TimeCode.Default(),
        #     destructive=False
        # ).do()
    
    if to_prim is not None:
        if isinstance(to_prim, str):
            to_prim = prim_utils.get_prim_at_path(to_prim)
        joint_to = script_utils.createJoint(stage, "Fixed", seg_0, to_prim)
        if joint_to_attributes is None:
            joint_to_attributes = DEFAULT_JOINT_ATTRIBUTES
        setup_joint(joint_to, joint_to_attributes)
        # omni.usd.commands.MovePrimCommand(
        #     path_from=to_prim.GetPath().pathString + "/D6Joint",
        #     path_to=prim_path + "/joint_to",
        #     keep_world_transform=False,
        #     time_code=Usd.TimeCode.Default(),
        #     destructive=False
        # ).do()
    
    return bar
        

def lemniscate(t, c):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    sin2p1 = torch.square(sin_t) + 1

    x = torch.stack([
        cos_t, sin_t * cos_t, c * sin_t
    ], dim=-1) / sin2p1.unsqueeze(-1)

    return x

def line_acc(t, a, threshold, alpha):
    # Convert angle c from degrees to radians
    c = torch.deg2rad(alpha)
    
    # Calculate the components of acceleration
    a_x = a * torch.cos(c)
    a_y = a * torch.sin(c)
    
    # Calculate maximum velocities in x and y directions
    v_max_x = a_x * threshold
    v_max_y = a_y * threshold
    
    # Calculate position components for x and y directions
    x = torch.where(t <= threshold, 0.5 * a_x * t**2, 
                    0.5 * a_x * threshold**2 + v_max_x * (t - threshold) - 0.5 * a_x * (t - threshold)**2)
    y = torch.where(t <= threshold, 0.5 * a_y * t**2, 
                    0.5 * a_y * threshold**2 + v_max_y * (t - threshold) - 0.5 * a_y * (t - threshold)**2)
    
    # z remains zero since no movement along z-axis
    z = torch.zeros_like(t)

    return torch.stack([x, y, z], dim=-1)

def line_segments(t, v, threshold, c):
    # v = torch.tensor(v)
    # threshold = torch.tensor(threshold)
    # c = torch.tensor(c)
    x = torch.where(t <= threshold, v * t, v * threshold + v * (t - threshold) * torch.cos(c))
    y = torch.where(t <= threshold, torch.zeros_like(t), v * (t - threshold) * torch.sin(c))
    z = torch.zeros_like(t)

    return torch.stack([x, y, z], dim=-1)

def line_segments_acc(t, a, unif_start, unif_end, c):
    v_turn = a * unif_start
    
    # 1th phase
    x1 = 0.5 * a * t**2
    y1 = torch.zeros_like(t)
    
    # 2th phase
    x2 = 0.5 * a * unif_start**2 + v_turn * (t - unif_start) * torch.cos(c)
    y2 = v_turn * (t - unif_start) * torch.sin(c)
    
    # 3th phase
    x3 = 0.5 * a * unif_start**2 + v_turn * (unif_end - unif_start) * torch.cos(c) + \
          (v_turn * (t - unif_end) - 0.5 * a * (t - unif_end)**2) * torch.cos(c)
    y3 = v_turn * (unif_end - unif_start) * torch.sin(c) + \
          (v_turn * (t - unif_end) - 0.5 * a * (t - unif_end)**2) * torch.sin(c)
    
    x = torch.where(t <= unif_start, x1, torch.where(t <= unif_end, x2, x3))
    y = torch.where(t <= unif_start, y1, torch.where(t <= unif_end, y2, y3))
    z = torch.zeros_like(t)
    
    return torch.stack([x, y, z], dim=-1)

def pentagram(t):
    a = 1.0
    b = 0.5
    x = -a * torch.sin(2 * t) - b * torch.sin(3 * t)
    y = a * torch.cos(2 * t) - b * torch.cos(3 * t)
    z = torch.zeros_like(t)

    return torch.stack([x, y, z], dim=-1)

def infeasible_pentagram(t, max_time):    
    v = 1.0
    T = 0.2 * max_time
    max_cycle_number = 5
    cycle_number = torch.clip(torch.floor(t / T).long(), 0, max_cycle_number - 1)
    time_in_cycle = t % T
    
    # init
    x = torch.zeros_like(t)
    y = torch.zeros_like(t)
    
    # get all corners
    x0 = torch.tensor(0.0).to(t.device)
    y0 = torch.tensor(0.0).to(t.device)
    angle = torch.tensor(0.0).to(t.device)
    pos_init = [torch.stack([x0, y0], dim=-1).clone()]
    angle_list = [angle.clone()]
    for _ in range(max_cycle_number - 1):
        x0 += v * T * torch.cos(angle)
        y0 += v * T * torch.sin(angle)
        angle -= torch.tensor(144.0) * (torch.pi / 180.0)
        pos_init.append(torch.stack([x0, y0], dim=-1).clone())
        angle_list.append(angle.clone())
    pos_init = torch.stack(pos_init)
    angle_list = torch.stack(angle_list)

    x = v * time_in_cycle * torch.cos(angle_list[cycle_number]) + pos_init[cycle_number][:, 0]
    y = v * time_in_cycle * torch.sin(angle_list[cycle_number]) + pos_init[cycle_number][:, 1]
    
    z = torch.zeros_like(x)

    return torch.stack([x, y, z], dim=-1)

def zigzag(t, target_times, target_points):
    # target_times: [batch, num_points]
    # target_points: [batch, num_points, 2]
    
    target_times = torch.concat([torch.zeros(1, device=target_times.device), torch.cumsum(target_times, dim=0)])
    num_points = target_times.shape[0]
    
    times_expanded = target_times.unsqueeze(0).expand(t.shape[-1], -1)
    t_expanded = t.unsqueeze(-1)
    prev_idx = num_points - (times_expanded > t_expanded).sum(dim=-1) - 1
    next_idx = num_points - (times_expanded > t_expanded).sum(dim=-1)
    # clip
    prev_idx = torch.clamp(prev_idx, max=num_points - 2) # [batch, future_step]
    next_idx = torch.clamp(next_idx, max=num_points - 1) # [batch, future_step]

    prev_x = torch.gather(target_points[:,0], 0, prev_idx) # [batch, future_step]
    next_x = torch.gather(target_points[:,0], 0, next_idx)
    prev_y = torch.gather(target_points[:,1], 0, prev_idx)
    next_y = torch.gather(target_points[:,1], 0, next_idx)
    prev_times = torch.gather(target_times, 0, prev_idx)
    next_times = torch.gather(target_times, 0, next_idx)
    k_x = (next_x - prev_x) / (next_times - prev_times)
    k_y = (next_y - prev_y) / (next_times - prev_times)
    x = prev_x + k_x * (t - prev_times) # [batch, future_step]
    y = prev_y + k_y * (t - prev_times)
    z = torch.zeros_like(x)
    
    return torch.stack([x, y, z], dim=-1)

def scale_time(t, a: float=1.0):
    return t / (1 + 1/(a*torch.abs(t)))


class TimeEncoding:
    def __init__(self, max_t):
        ...
    
    def encode(self, t):
        ...
