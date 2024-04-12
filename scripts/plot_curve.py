import matplotlib.pyplot as plt
fontsize = 12
def plot_multi_mean_std_graph(x_values_list, y_mean_values_list, y_std_values_list, x_label, y_label, labels, name):
    plt.style.use("ggplot")
    plt.figure(figsize=(5.4, 4.5))  # 设置画布大小
    for x_values, y_mean_values, y_std_values, label in zip(x_values_list, y_mean_values_list, y_std_values_list, labels):
        plt.errorbar(x_values, y_mean_values, yerr=y_std_values, fmt='-o', label=label)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.legend()
    plt.grid(True)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('{}.png'.format(name))

# catch radius - capture rate
x_values_list = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
y_mean_values_list = [[2, 3, 5, 7, 11], [1, 4, 9, 16, 25]]
y_std_values_list = [[0.5, 0.3, 0.6, 0.8, 1.0], [0.2, 0.4, 0.7, 0.9, 1.2]]
labels = ["HCMP", "MAPPO"]

plot_multi_mean_std_graph(x_values_list, y_mean_values_list, y_std_values_list, "Catch Radius", "Capture Rate", labels, 'CatchRadius_capture')