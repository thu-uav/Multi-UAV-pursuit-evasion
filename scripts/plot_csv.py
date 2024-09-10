import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fontsize = 18
# timestep = -1
# steps = pd.read_csv('/home/chenjy/OmniDrones/scripts/data/ours_seed0.csv')['Step'].to_numpy()[:timestep] * 131090.0
timestep1 = 230
timestep2 = 500
timestep3 = 500
steps = {
    'Ours': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/ours_seed0.csv')['Step'].to_numpy()[:timestep1] * 131090.0,
    'MAPPO + EP': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/MAPPO_EP_seed0.csv')['Step'].to_numpy()[:timestep2] * 131090.0,
    'MAPPO + Envgen': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/MAPPO_Envgen_seed0.csv')['Step'].to_numpy()[:timestep3] * 131090.0,
}
data = {
    'Ours': {
        'seed0': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/ours_seed0.csv')['success'].to_numpy()[:timestep1],
        'seed1': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/ours_seed1.csv')['success'].to_numpy()[:timestep1],
        'seed2': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/ours_seed2.csv')['success'].to_numpy()[:timestep1],
    },
    'MAPPO + EP': {
        'seed0': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/MAPPO_EP_seed0.csv')['success'].to_numpy()[:timestep2],
        'seed1': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/MAPPO_EP_seed1.csv')['success'].to_numpy()[:timestep2],
        'seed2': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/MAPPO_EP_seed2.csv')['success'].to_numpy()[:timestep2],
    },
    'MAPPO + Envgen': {
        'seed0': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/MAPPO_Envgen_seed0.csv')['success'].to_numpy()[:timestep3],
        'seed1': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/MAPPO_Envgen_seed1.csv')['success'].to_numpy()[:timestep3],
        'seed2': pd.read_csv('/home/chenjy/OmniDrones/scripts/data/MAPPO_Envgen_seed2.csv')['success'].to_numpy()[:timestep3],
    }
}

def calculate_mean_std(data, steps):
    # steps = data['timestep']
    means = {}
    stds = {}
    for algo, seeds in data.items():
        seed_values = np.array(list(seeds.values()))
        means[algo] = np.mean(seed_values, axis=0)
        stds[algo] = np.std(seed_values, axis=0)
    return steps, means, stds

steps, means, stds = calculate_mean_std(data, steps)

def plot_mean_std(steps, means, stds):
    plt.style.use("ggplot")
    plt.figure(figsize=(5.4, 4.5))
    for algo in means.keys():
        mean_curve = means[algo]
        std_curve = stds[algo]
        plt.plot(steps[algo], mean_curve, label=f'{algo}')
        plt.fill_between(steps[algo], mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
    
    plt.xlabel('Timesteps', fontsize=fontsize)
    plt.ylabel('Capture rate', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ablation.pdf')

plot_mean_std(steps, means, stds)