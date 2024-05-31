import torch
import matplotlib.pyplot as plt
torch.seed()
num_points = 20
interval_min = 0.5
interval_max = 1.5
size_min = -1.0
size_max = 1.0
n = 100 # batch size
random_data = torch.rand(n, num_points - 1) # 0~1
intervals = interval_min + (interval_max - interval_min) * random_data
times = torch.concat([torch.zeros((intervals.shape[0], 1)), torch.cumsum(intervals, dim=1)], dim=1)
x_interval = size_min + (size_max - size_min) * torch.rand(n, num_points)
y_interval = size_min + (size_max - size_min) *  torch.rand(n, num_points)
t = torch.rand(n, 1) * 10.0
steps = 4
step_size = 0.05
t = t + step_size * torch.arange(0, steps)
times_expanded = times.unsqueeze(1).expand(-1, t.shape[-1], -1)
t_expanded = t.unsqueeze(-1)
breakpoint()
prev_idx = num_points - (times_expanded > t_expanded).sum(dim=-1) - 1
next_idx = num_points - (times_expanded > t_expanded).sum(dim=-1)
# clip
prev_idx = torch.clamp(prev_idx, max=num_points - 2) # [batch, future_step]
next_idx = torch.clamp(next_idx, max=num_points - 1) # [batch, future_step]

prev_x = torch.gather(x_interval, 1, prev_idx) # [batch, future_step]
next_x = torch.gather(x_interval, 1, next_idx)
prev_y = torch.gather(y_interval, 1, prev_idx)
next_y = torch.gather(y_interval, 1, next_idx)
prev_times = torch.gather(times, 1, prev_idx)
next_times = torch.gather(times, 1, next_idx)
k_x = (next_x - prev_x) / (next_times - prev_times)
k_y = (next_y - prev_y) / (next_times - prev_times)
x = prev_x + k_x * (t - prev_times) # [batch, future_step]
y = prev_y + k_y * (t - prev_times)

fig, axs = plt.subplots(2, 1, figsize=(8, 8))  # 1行2列
axs[0].plot(t[0], x[0], label='X Coordinate')
axs[0].plot(t[0], y[0], label='Y Coordinate')
axs[1].plot(times[0], x_interval[0], label='gt X Coordinate')
axs[1].plot(times[0], y_interval[0], label='gt Y Coordinate')
# axs[0].xlabel('Time (seconds)')
# axs[0].ylabel('Coordinate')
axs[0].legend()
axs[0].grid(True)
axs[0].axis('equal')

# axs[1].xlabel('Time (seconds)')
# axs[1].ylabel('gt Coordinate')
axs[1].legend()
axs[1].grid(True)
axs[1].axis('equal')

plt.title('Trajectory Plot')
plt.savefig('plot')