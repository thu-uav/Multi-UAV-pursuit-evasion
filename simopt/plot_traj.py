import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/jiayu/OmniDrones/simopt/real_data/rl_hover_1.csv')

# 创建2x3的子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
breakpoint()

# 绘制x坐标
axes[0, 0].plot(df['pos.x'], label='X Coordinate')
axes[0, 0].set_title('X Coordinate over Time')
axes[0, 0].legend()

# 绘制y坐标
axes[0, 1].plot(df['pos.y'], label='Y Coordinate')
axes[0, 1].set_title('Y Coordinate over Time')
axes[0, 1].legend()

# 绘制z坐标
axes[0, 2].plot(df['pos.z'], label='Z Coordinate')
axes[0, 2].set_title('Z Coordinate over Time')
axes[0, 2].legend()

# 绘制vx速度
axes[1, 0].plot(df['vel.x'], label='Velocity X')
axes[1, 0].set_title('Velocity X over Time')
axes[1, 0].legend()

# 绘制vy速度
axes[1, 1].plot(df['vel.y'], label='Velocity Y')
axes[1, 1].set_title('Velocity Y over Time')
axes[1, 1].legend()

# 绘制vz速度
axes[1, 2].plot(df['vel.z'], label='Velocity Z')
axes[1, 2].set_title('Velocity Z over Time')
axes[1, 2].legend()

# 调整子图间距
plt.tight_layout()

# 显示图表
plt.savefig('trajectory')