# import torch
# import torch.testing
# def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor):
#     shape = q.shape
#     q_w = q[:, 0]
#     q_vec = q[:, 1:]
#     a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
#     b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
#     c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
#     return a - b + c

# def quat_rotate(q: torch.Tensor, v: torch.Tensor):
#     # Forward rotation
#     shape = q.shape
#     q_w = q[:, 0]
#     q_vec = q[:, 1:]
#     a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
#     b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
#     c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
#     rotated_v = a + b + c  # Forward rotation result

#     return rotated_v

# # 设置随机种子以便复现
# torch.manual_seed(42)

# # 生成随机的四元数和向量
# q = torch.rand(5, 4)
# v = torch.rand(5, 3)

# # 使用quat_rotate_inverse逆旋转
# rotated_v = quat_rotate_inverse(q, v)

# # 使用quat_rotate还原向量
# restored_v = quat_rotate(q, rotated_v)
# import pdb; pdb.set_trace()

# # 检查还原后的向量是否接近原始向量
# torch.testing.assert_allclose(restored_v, v, rtol=1e-5, atol=1e-8)

# print("测试通过!")

import torch
import numpy as np

def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=-1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    matrix = torch.stack(
        [
            1 - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            1 - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            1 - (txx + tyy),
        ],
        dim=-1,
    )
    matrix = matrix.unflatten(matrix.dim() - 1, (3, 3))
    return matrix

def rotate_point(v, rotation_matrix):
    v = v.view(-1, 3)  # Reshape the point to a column vector
    rotated_point = torch.mm(v, rotation_matrix)
    return rotated_point.view(-1)  # Reshape back to a 1D tensor

def quaternion_multiply(q1, q2):
    # Quaternion multiplication
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.tensor([w, x, y, z])

def quaternion_rotate_vector(quaternion, vector):
    # Quaternion rotation of a vector
    # qvq^-1
    quaternion_vector = torch.concat((torch.tensor([0]), vector))
    conjugate_quaternion = torch.tensor([quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]])
    
    rotated_vector = quaternion_multiply(quaternion, quaternion_multiply(quaternion_vector, conjugate_quaternion))[1:]
    
    return rotated_vector

# Example values for testing
torch.manual_seed(1)
q = torch.tensor([[np.cos(np.pi/6), 0, np.sin(np.pi/6), 0]]).float()  # New quaternion (w, x, y, z)
v = torch.tensor([[2.0, 1.0, 3.0]])  # New vector (x, y, z)

# Function calls
body_rate = quat_rotate_inverse(q, v)
restored_v = quat_rotate(q, body_rate)

quaternion = torch.tensor([np.cos(np.pi/6), 0, np.sin(np.pi/6), 0]).float()  # Example quaternion
rotation_matrix = quaternion_to_rotation_matrix(quaternion)
point_to_rotate = torch.tensor([2.0, 1.0, 3.0])  # Example point in 3D space

rotated_point = rotate_point(point_to_rotate, rotation_matrix)

# Display the results
print("Original Quaternion (q):", q)
print("Original Vector (v):", v)
print("Inverse Rotated Vector (body_rate):", body_rate)
print("Final Rotated Vector (restored_v):", restored_v)
# print('matrix', matrix)