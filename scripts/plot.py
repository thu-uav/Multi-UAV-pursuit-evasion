import torch
import matplotlib.pyplot as plt

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

# 生成轨迹
max_time = 10
t = torch.linspace(0, max_time, 1000)
trajectory = infeasible_pentagram(t, max_time)

# 提取坐标
x_traj = trajectory[:, 0]
y_traj = trajectory[:, 1]

# 绘制轨迹
plt.figure(figsize=(8, 8))
plt.plot(x_traj.cpu(), y_traj.cpu(), label='Trajectory')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory of Infeasible Pentagram')
plt.grid(True)
plt.legend()
plt.savefig('star')
