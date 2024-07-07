import numpy as np

# 无人机参数
m = 0.03  # 质量（kg）
g = 9.81  # 重力加速度（m/s^2
# TODO
d = 0.25  # 电机到质心的距离（m）
I = np.diag([1.4e-5, 1.4e-5, 2.17e-5])  # 惯性张量矩阵（kg*m^2）

# 测量值（世界坐标系下）
a_world = np.array([0.1, 0.2, -9.7])  # 线加速度（m/s^2）
alpha_world = np.array([0.01, 0.02, 0.03])  # 角加速度（rad/s^2）
omega_world = np.array([0.1, 0.2, 0.3])  # 角速度（rad/s）

# 假设通过IMU测得的从世界坐标系到机体坐标系的旋转矩阵
# 注意：这是一个示例，实际旋转矩阵应根据姿态数据计算得到
R = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# 将测量值转换到机体坐标系下
a_body = np.dot(R, a_world)
alpha_body = np.dot(R, alpha_world)
omega_body = np.dot(R, omega_world)

# 计算总推力（机体坐标系下）
F = m * a_body # 包含重力

# 计算总扭矩（机体坐标系下）
M = np.dot(I, alpha_body) + np.cross(omega_body, np.dot(I, omega_body))

# 构建系数矩阵和常数向量
A = np.array([
    [-1, -1, -1, -1],
    [d/np.sqrt(2), -d/np.sqrt(2), -d/np.sqrt(2), d/np.sqrt(2)],
    [d/np.sqrt(2), d/np.sqrt(2), -d/np.sqrt(2), -d/np.sqrt(2)],
])

b = np.array([
    F[2],
    M[0],
    M[1],
])

# 求解电机推力
T = np.linalg.lstsq(A, b, rcond=None)[0]

print("电机推力 (N):", T)
