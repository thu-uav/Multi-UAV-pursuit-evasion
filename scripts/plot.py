import torch
import matplotlib.pyplot as plt

def line_segments_acc(t, a, unif_start, unif_end, c):
    # TODO: use c
    a = 1.0 # 增大a
    b = 0.5 # 减小b
    x = -a * torch.sin(2 * t) - b * torch.sin(3 * t)
    y = a * torch.cos(2 * t) - b * torch.cos(3 * t)
    z = torch.zeros_like(t)
    
    return x, y, z

# 定义参数
a = 3.0
unif_start = 1.0
unif_end = 2.0
c = torch.tensor(0.95) * torch.pi
t = torch.linspace(0, 10, 1000)  # 生成从0到3的300个点

# 调用函数
x, y, z = line_segments_acc(t, a, unif_start, unif_end, c)

# 绘制图形
plt.figure(figsize=(7, 7))

# # 绘制 x 曲线
# plt.subplot(1, 3, 1)
# plt.plot(t.numpy(), x.numpy(), label='x(t)')
# plt.xlabel('t')
# plt.ylabel('x')
# plt.title('x vs t')
# plt.legend()

# # 绘制 y 曲线
# plt.subplot(1, 3, 2)
# plt.plot(t.numpy(), y.numpy(), label='y(t)', color='orange')
# plt.xlabel('t')
# plt.ylabel('y')
# plt.title('y vs t')
# plt.legend()

# 绘制 x-y 曲线
# plt.subplot(1, 3, 3)
plt.plot(x.numpy(), y.numpy(), label='y(x)', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('y vs x')
plt.legend()

plt.tight_layout()
plt.savefig('plot.png')
