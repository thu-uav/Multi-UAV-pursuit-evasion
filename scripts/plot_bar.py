import matplotlib.pyplot as plt

# 柱状图数据
x_values = ['', 'DR+DFS', '', '', '', '', 'Ours', '',]
y_values = [1.0, 0.9, 0.9, 0.85, 1.0, 1.0, 1.0, 1.0]

# 创建画布和子图
fig, ax = plt.subplots()

# 设置柱子的位置
bar_width = 0.5
index = [0.5, 1.5, 2.5, 3.5, 6.5, 7.5, 8.5, 9.5]

# 绘制柱状图
bars = ax.bar(index[:4], y_values[:4], bar_width, color='purple')
bars = ax.bar(index[4:], y_values[4:], bar_width, color='red')

# 在柱子上方标注数值
for i in range(len(index)):
    plt.text(index[i], y_values[i], str(y_values[i]), ha='center', va='bottom', color='black')

# 添加标题和标签
# ax.set_title('Custom Spacing Bar Chart')
# ax.set_xlabel('Bars')
ax.set_ylabel('Capture Rate')

# 设置x轴刻度和标签
ax.set_xticks(index)
ax.set_xticklabels(x_values)

# 显示图形
plt.savefig('bar.png')