import torch

def taylor_coefficients(func, t_values, order):
    coefficients = []

    for t in t_values:
        t.requires_grad_(True)
        y = func(t)
        derivatives = [torch.autograd.grad(y, t, create_graph=True, allow_unused=True)[0]]

        for i in range(2, order + 1):
            derivatives.append(torch.autograd.grad(derivatives[-1], t, create_graph=True, allow_unused=True)[0])

        coefficients.append([derivative.item() for derivative in derivatives])

    return coefficients

# 定义给定的函数
def x_func(t):
    return -1.5 * torch.sin(2 * t) - 0.5 * torch.sin(3 * t)

def y_func(t):
    return 1.5 * torch.cos(2 * t) - 0.5 * torch.cos(3 * t)

# 设定 t 值
t_values = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# 计算 x 和 y 的零次方项到七次方项的系数
order = 7
x_coefficients = taylor_coefficients(x_func, t_values, order)
y_coefficients = taylor_coefficients(y_func, t_values, order)

# 打印结果
for i in range(len(t_values)):
    print(f"t = {t_values[i].item()}: x coefficients = {x_coefficients[i]}, y coefficients = {y_coefficients[i]}")
