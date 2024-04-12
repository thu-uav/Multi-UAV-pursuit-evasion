import matplotlib.pyplot as plt
import numpy as np
fontsize = 12
def plot_multi_bar_graph(x_values, y_values_list, y_std_values_list, x_label, y_label, labels):
    plt.style.use("ggplot")
    plt.figure(figsize=(5.4, 4.5))  # 设置画布大小
    x = np.arange(len(x_values))
    width = 0.2
    for i, (y_values, y_std_values, label) in enumerate(zip(y_values_list, y_std_values_list, labels)):
        plt.bar(x + i*width, y_values, width=width, label=label)
        for j, (x_val, y_val, y_std) in enumerate(zip(x_values, y_values, y_std_values)):
            plt.vlines(x_val + i*width, y_val - y_std, y_val + y_std, colors='black')

    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.xticks(x + width * (len(y_values_list) - 1) / 2, x_values)
    plt.legend()
    plt.grid(True)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('bar.png')

# 示例数据
x_values = [0, 1, 2, 3]
y_values_list = [
    [0.8, 0.7, 0.6, 0.5],  # Algorithm 1
    [0.7, 0.6, 0.5, 0.4],  # Algorithm 2
    [0.6, 0.5, 0.4, 0.3],  # Algorithm 3
    [0.5, 0.4, 0.3, 0.2]   # Algorithm 4
]
y_std_values_list = [
    [0.05, 0.03, 0.06, 0.08],  # Algorithm 1
    [0.04, 0.02, 0.05, 0.07],  # Algorithm 2
    [0.03, 0.02, 0.04, 0.06],  # Algorithm 3
    [0.02, 0.01, 0.03, 0.05]   # Algorithm 4
]
labels = ["Algorithm 1", "Algorithm 2", "Algorithm 3", "Algorithm 4"]

# 设置图形属性并画图
plot_multi_bar_graph(x_values, y_values_list, y_std_values_list, "Number of obstacles", "Capture Rate", labels)
