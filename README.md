# Multi-UAV Pursuit-Evasion with Online Planning in Unknown Environments by Deep Reinforcement Learning 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Jiayu Chen, Chao Yu, Guosheng Li, Wenhao Tang, Xinyi Yang, Botian Xu, Huazhong Yang, Yu Wang

Website: https://sites.google.com/view/pursuit-evasion-rl

The implementation in this repositorory is used in the paper "Multi-UAV Pursuit-Evasion with Online Planning in Unknown Environments by Deep Reinforcement Learning". This repository is heavily based on https://github.com/btx0424/OmniDrones.git.


## Install

#### 1. Download Isaac Sim (local version)

Download the [Omniverse Isaac Sim (local version)](https://developer.nvidia.com/isaac-sim) and install the desired Isaac Sim release **(version 2022.2.0)** following the [official document](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html). *Note that Omniverse Isaac Sim supports multi-user access, eliminating the need for repeated downloads and installations across different user accounts.*

Set the following environment variables to your ``~/.bashrc`` or ``~/.zshrc`` files :

```
# Isaac Sim root directory
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2022.2.0"
```

*(Currently we use isaac_sim-2022.2.0. Whether other versions can work or not is not guaranteed.)*

After adding the environment variable, apply the changes by running:
```
source ~/.bashrc
```

#### 2. Conda

Although Isaac Sim comes with a built-in Python environment, we recommend using a seperate conda environment which is more flexible. We provide scripts to automate environment setup when activating/deactivating a conda environment at ``OmniDrones/conda_setup``.

```
conda create -n sim python=3.7
conda activate sim

# at OmniDrones/
cp -r conda_setup/etc $CONDA_PREFIX
# re-activate the environment
conda activate sim
# install OmniDrones
pip install -e .

# verification
python -c "from omni.isaac.kit import SimulationApp"
# which torch is being used
python -c "import torch; print(torch.__path__)"
```

#### 3. Third Party Packages
OmniDrones requires specific versions of the `tensordict` and `torchrl` packages. For the ``deploy`` branch, it supports `tensordict` version 0.1.2+5e6205c and `torchrl` version 0.1.1+e39e701. 

We manage these two packages using Git submodules to ensure that the correct versions are used. To initialize and update the submodules, follow these steps:

Get the submodules:
```
# at OmniDrones/
git submodule update --init --recursive
```
Pip install these two packages respectively:
```
# at OmniDrones/
cd third_party/tensordict
pip install -e .
```
```
# at OmniDrones/
cd third_party/torchrl
pip install -e .
```
#### 4. Verification
```
# at OmniDrones/
cd scripts
python train.py headless=true wandb.mode=disabled total_frames=50000 task=Hover
```

#### 5. Working with VSCode

To enable features like linting and auto-completion with VSCode Python Extension, we need to let the extension recognize the extra paths we added during the setup process.

Create a file ``.vscode/settings.json`` at your workspace if it is not already there.

After activating the conda environment, run

```
printenv > .vscode/.python.env
``````

and edit ``.vscode/settings.json`` as:

```
{
    // ...
    "python.envFile": "${workspaceFolder}/.vscode/.python.env",
}
```

## Usage

For usage and more details of *Omnidrones*, please refer to the [documentation](https://omnidrones.readthedocs.io/en/latest/).

```
# at OmniDrones/
cd scripts
python train_generator.py # Train the pursuit-evasion task with Automatic Environment Generator
python train.py # Train the pursuit-evasion task without Automatic Environment Generator
python train_deploy.py # fine-tune the policy by smoothness reward
```

## Citation

Please cite [this paper](xxx) if you use our method in your work:

```
xxx
```
