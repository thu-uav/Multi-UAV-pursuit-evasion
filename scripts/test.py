import torch
import matplotlib.pyplot as plt

def trajectory(t):
    t_threshold = torch.tensor(5.0)
    v = torch.tensor(3.0)
    c = torch.tensor(2 * torch.pi / 3)
    x = torch.where(t <= t_threshold, v * t, v * t_threshold + v * (t - t_threshold) * torch.cos(c))
    y = torch.where(t <= t_threshold, torch.zeros_like(t), v * (t - t_threshold) * torch.sin(c))
    return x, y

# Generate points on the curve
t = torch.linspace(0, 10, 100)
x, y = trajectory(t)

# Plot the curve
plt.figure(figsize=(6, 6))
plt.plot(x.numpy(), y.numpy(), label='Trajectory')

# Plot the axes
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)

# Set labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory')

# Add a legend
plt.legend()

# Show plot
plt.savefig('tmp.png')