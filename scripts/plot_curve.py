import matplotlib.pyplot as plt
fontsize = 22
def plot_multi_mean_std_graph(x_values_list, y_mean_values_list, y_std_values_list, x_label, y_label, labels, name, inverse=False):
    plt.style.use("ggplot")
    plt.figure(figsize=(5.4, 4.5))  # 设置画布大小
    for x_values, y_mean_values, y_std_values,y2_mean_values, y2_std_values, label in zip(x_values_list, y_mean_values_list, y_std_values_list, y2_mean_values_list, y2_std_values_list, labels):
        plt.errorbar(x_values, y_mean_values, yerr=y_std_values, fmt='-o', label=label)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.legend(loc='upper left', fontsize=15)
    plt.grid(True)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if inverse:
        plt.gca().invert_xaxis()  # 反转 x 轴
    plt.tight_layout()
    plt.savefig('{}.pdf'.format(name))

# catch radius - capture rate
x_values_list = [[0.5, 0.4, 0.3, 0.2, 0.12], [0.5, 0.4, 0.3, 0.2, 0.12]]
y_mean_values_list = [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.0, 0.0, 0.0]]
y_std_values_list = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]

# catch radius - capture timestep
y2_mean_values_list = [[200.0, 200.0, 200.0, 200.0, 200.0], [200.0, 400.0, 800.0, 800.0, 800.0]]
y2_std_values_list = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
labels = ["HCMP", "MAPPO"]

plot_multi_mean_std_graph(x_values_list, y_mean_values_list, y_std_values_list, "Catch Radius", "Capture Rate", labels, 'CatchRadius_rate', inverse=True)
plot_multi_mean_std_graph(x_values_list, y2_mean_values_list, y2_std_values_list, "Catch Radius", "Capture Timestep", labels, 'CatchRadius_time', inverse=True)

# speed - capture rate
x_values_list = [[0.8, 1.2, 1.6, 2.0, 2.4], [0.8, 1.2, 1.6, 2.0, 2.4]]
y_mean_values_list = [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.5, 0.0, 0.0, 0.0]]
y_std_values_list = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]

# speed - capture timestep
y2_mean_values_list = [[200.0, 200.0, 200.0, 200.0, 200.0], [200.0, 400.0, 800.0, 800.0, 800.0]]
y2_std_values_list = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]
labels = ["HCMP", "MAPPO"]

plot_multi_mean_std_graph(x_values_list, y_mean_values_list, y_std_values_list, "Evader Speed", "Capture Rate", labels, 'Speed_rate')
plot_multi_mean_std_graph(x_values_list, y2_mean_values_list, y2_std_values_list, "Evader Speed", "Capture Timestep", labels, 'Speed_time')