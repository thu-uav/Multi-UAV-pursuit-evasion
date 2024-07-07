import numpy as np
import cvxpy as cp
import time
import torch

def solve_qp_batch(actions, prev_actions, delta):
    # 获取batch大小和动作维度
    batch_size, action_size = actions.shape

    # 定义变量
    a = cp.Variable((batch_size, action_size))

    # 定义目标函数
    objective = cp.Minimize(cp.sum_squares(a - actions))

    # 定义约束条件
    constraints = [cp.norm(a - prev_actions, 2, axis=1) <= delta]

    # 构建问题
    prob = cp.Problem(objective, constraints)

    # 求解问题
    prob.solve()

    # 获取解
    corrected_actions = a.value

    return corrected_actions

# 示例数据
# prev_actions = np.array([[0.5, 0.5], [0.4, 0.4]])
prev_actions = np.ones((4096, 2)) * 0.5
# actions = np.array([[0.6, 0.7], [0.7, 0.8]])
actions = np.ones((4096, 2))
actions[:, 0] = 0.6
actions[:, 1] = 0.7
delta = 0.1

# 使用QP求解器找到修正后的动作
start = time.time()
corrected_actions = solve_qp_batch(actions, prev_actions, delta)
end = time.time()
print('time', end - start)
# print("Corrected Actions:\n", corrected_actions)


# import numpy as np
# import torch

# def solve_qp_batch(actions, prev_actions, delta):
#     # 将数据转换为PyTorch张量
#     actions_tensor = torch.tensor(actions, dtype=torch.float32, device='cuda:0')
#     prev_actions_tensor = torch.tensor(prev_actions, dtype=torch.float32, device='cuda:0')
#     delta_tensor = torch.tensor(delta, dtype=torch.float32, device='cuda:0')

#     # 计算二范数差值
#     norms = torch.norm(actions_tensor - prev_actions_tensor, dim=1)

#     # 找到超过阈值的索引
#     exceeded_indices = norms > delta_tensor

#     # 对超过阈值的动作进行修正
#     corrected_actions = actions_tensor.clone()
#     corrected_actions[exceeded_indices] = prev_actions_tensor[exceeded_indices] + delta_tensor

#     # 将张量转换为NumPy数组并返回
#     return corrected_actions.cpu().numpy()

# # 示例数据
# # prev_actions = np.array([[0.5, 0.5], [0.4, 0.4]])
# prev_actions = np.ones((2048, 2)) * 0.5
# # actions = np.array([[0.6, 0.7], [0.7, 0.8]])
# actions = np.ones((2048, 2))
# actions[:, 0] = 0.6
# actions[:, 1] = 0.7
# delta = 0.1
# actions = torch.tensor(actions, device='cuda:0')
# prev_actions = torch.tensor(prev_actions, device='cuda:0')

# # 使用QP求解器找到修正后的动作
# import time
# start = time.time()
# corrected_actions = solve_qp_batch(actions, prev_actions, delta)
# end = time.time()
# print('time', end - start)
# # print("Corrected Actions:\n", corrected_actions)
