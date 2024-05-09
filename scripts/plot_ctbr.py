import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
def plot_ctbr(duration, ratex, ratey, ratez, thrust_ratio):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    ax1.plot(duration, ratex, 'r.', label='rate x')
    ax2.plot(duration, ratey, 'b.', label='rate y')
    ax3.plot(duration, ratez, 'g.', label='rate z')
    ax4.plot(duration, thrust_ratio, 'g.', label='thrust z')
    # plt.legend()
    plt.xlabel('Duration')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    plt.savefig('ctbr.png')

max_length = 500
t = np.linspace(0, max_length * 0.01, max_length)
data = np.load('/home/jiayu/OmniDrones/scripts/ctbr.npy')
rate_x = data[:max_length, 0, 0]
rate_y = data[:max_length, 0, 1]
rate_z = data[:max_length, 0, 2]
thrust = data[:max_length, 0, 3]
plot_ctbr(t, rate_x, rate_y, rate_z, thrust)