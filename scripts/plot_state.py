import matplotlib.pyplot as plt
import numpy as np

# action error
start_idx = 20
data = np.load('/home/jiayu/OmniDrones/scripts/action_error.npy')
data = data[:,:,0].mean(axis=1)[start_idx:]
T = data.shape[0]  # 时间步长的数量
subplots = [4, 1]  # 1行4列的子图布局

# 创建一个图形框
fig, axs = plt.subplots(subplots[0], subplots[1])

# 如果只需要一个轴对象，可以直接使用axs
if subplots[0] * subplots[1] == 1:
    axs = [axs]

# 对每个通道绘制数据
for i in range(4):
    # 绘制每个子图
    axs[i].plot(data[:, i], label=f'Channel {i}')
    
    # 计算并标注最大值和平均值
    max_value = np.max(data[:, i])
    avg_value = np.mean(data[:, i])
    axs[i].text(0.5, 0.5, f'Max: {max_value:.5f}\nAvg: {avg_value:.5f}',
                horizontalalignment='center',
                verticalalignment='center',
                transform=axs[i].transAxes,
                fontdict={'color': 'red', 'weight': 'bold'})
    
    # 设置y轴标签
    axs[i].set_ylabel(f'Channel {i}')
    
# 设置x轴标签
plt.xlabel('Time')

# 显示图例
for i in range(4):
    axs[i].legend()

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.savefig('action_error')

##################################################################
# velocity
start_idx = 0
data = np.load('/home/jiayu/OmniDrones/scripts/vel.npy')
data = data[:,:,0].mean(axis=1)[start_idx:]
T = data.shape[0]  # 时间步长的数量
num_channel = data.shape[-1]
subplots = [num_channel, 1]  # 1行4列的子图布局

# 创建一个图形框
fig, axs = plt.subplots(subplots[0], subplots[1])

# 如果只需要一个轴对象，可以直接使用axs
if subplots[0] * subplots[1] == 1:
    axs = [axs]

# 对每个通道绘制数据
for i in range(num_channel):
    # 绘制每个子图
    axs[i].plot(data[:, i], label=f'Channel {i}')
    
    # 计算并标注最大值和平均值
    max_value = np.max(data[:, i])
    avg_value = np.mean(data[:, i])
    axs[i].text(0.5, 0.5, f'Max: {max_value:.5f}\nAvg: {avg_value:.5f}',
                horizontalalignment='center',
                verticalalignment='center',
                transform=axs[i].transAxes,
                fontdict={'color': 'red', 'weight': 'bold'})
    
    # 设置y轴标签
    axs[i].set_ylabel(f'Channel {i}')
    
# 设置x轴标签
plt.xlabel('Time')

# 显示图例
for i in range(num_channel):
    axs[i].legend()

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.savefig('velocity')

##################################################################
# acc
start_idx = 0
data = np.load('/home/jiayu/OmniDrones/scripts/acc.npy')
data = data[...,0].mean(axis=-1)[start_idx:]
T = data.shape[0]  # 时间步长的数量
num_channel = data.shape[-1]
subplots = [num_channel, 1]

# 创建一个图形框
fig, axs = plt.subplots(subplots[0], subplots[1])

# 如果只需要一个轴对象，可以直接使用axs
if subplots[0] * subplots[1] == 1:
    axs = [axs]

# 对每个通道绘制数据
for i in range(num_channel):
    # 绘制每个子图
    axs[i].plot(data[:, i], label=f'Channel {i}')
    
    # 计算并标注最大值和平均值
    max_value = np.max(data[:, i])
    avg_value = np.mean(data[:, i])
    axs[i].text(0.5, 0.5, f'Max: {max_value:.5f}\nAvg: {avg_value:.5f}',
                horizontalalignment='center',
                verticalalignment='center',
                transform=axs[i].transAxes,
                fontdict={'color': 'red', 'weight': 'bold'})
    
    # 设置y轴标签
    axs[i].set_ylabel(f'Channel {i}')
    
# 设置x轴标签
plt.xlabel('Time')

# 显示图例
for i in range(num_channel):
    axs[i].legend()

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.savefig('acc')