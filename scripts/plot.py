import torch
import matplotlib.pyplot as plt

def line_segments_acc(t, a, threshold):
    v_max = a * threshold
    x = torch.where(t <= threshold, 0.5 * a * t**2, 0.5 * a * threshold**2 + v_max * (t - threshold) - 0.5 * a * (t - threshold)**2)
    y = torch.zeros_like(t)
    z = torch.zeros_like(t)

    return x, y, z

# 定义参数
a = 3.0
t_max = 2
t = torch.linspace(0, t_max, 200)
threshold = torch.tensor(0.5) * t_max

# 调用函数
x, y, z = line_segments_acc(t, a, threshold)

# 绘制图形
plt.figure(figsize=(10, 5))

# 绘制 x 曲线
plt.subplot(1, 2, 1)
plt.plot(t.numpy(), x.numpy(), label='x(t)')
plt.xlabel('t')
plt.ylabel('x')
plt.title('x vs t')
plt.legend()

# # 绘制 y 曲线
# plt.subplot(1, 3, 2)
# plt.plot(t.numpy(), y.numpy(), label='y(t)', color='orange')
# plt.xlabel('t')
# plt.ylabel('y')
# plt.title('y vs t')
# plt.legend()

# 绘制 x-y 曲线
plt.subplot(1, 2, 2)
plt.plot(x.numpy(), y.numpy(), label='y(x)', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('y vs x')
plt.legend()

plt.tight_layout()
plt.savefig('plot.png')
