![Visualization of OmniDrones](docs/source/_static/visualization.jpg)

---

# OmniDrones

[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-2022.2.0-orange.svg)](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://docs.python.org/3/whatsnew/3.7.html)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://omnidrones.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


*OmniDrones* is an open-source platform designed for reinforcement learning research on multi-rotor drone systems. Built on [Nvidia Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html), *OmniDrones* features highly efficient and flxeible simulation that can be adopted for various research purposes. We also provide a suite of benchmark tasks and algorithm baselines to provide preliminary results for subsequent works.


## Installation (Local version)

### 1. Isaac Sim

Download the [Omniverse Isaac Sim (local version/in the cloud version)](https://developer.nvidia.com/isaac-sim) and install the desired Isaac Sim release **(version 2022.2.0)** following the [official document](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html). *Note that Omniverse Isaac Sim supports multi-user access, eliminating the need for repeated downloads and installations across different user accounts.*

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

### 2. Conda

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

### 3. Third Party Packages
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
### 4. Verification
```
# at OmniDrones/
cd scripts
python train.py headless=true wandb.mode=disabled total_frames=50000 task=Hover
```

### 5. Working with VSCode

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

 
## Using Docker (Container version)
First, make sure your computer has installed ``Docker``, ``NVIDIA Driver`` and ``NVIDIA Container Toolkit``. Then, you should successfully run: 

```
# Verification
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

Download the image from Docker Hub:
```
docker pull jimmyzhangruize/isaac-sim:2022.2.0_deploy
```
You need to clone the OmniDrones repository to your local computer because we use OmniDrones mounted in the container.

Then, run the image:
```
docker run -dit --gpus all -e "ACCEPT_EULA=Y" --net=host \
    -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache/Kit:rw \
    -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
    -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
    -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
    -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
    -v ~/docker/isaac-sim/documents:/root/Documents:rw \
    -v ***:/root/OmniDrones:rw \
    -v /data:/data \
    -e "WANDB_API_KEY=***" \
    --runtime=nvidia \
    jimmyzhangruize/isaac-sim:2022.2.0_deploy /bin/bash
```

Note that:
1. You need to replace *** with the directory where OmniDrones is located in your local computer in -v ***:/root/OmniDrones:rw.
2. You need to replace *** with your own WANDB_API_KEY in -e "WANDB_API_KEY=***" . If you do not need to use Wandb, you can omit this line.
3. In the container, OmniDrones is located at ``/root/OmniDrones`` and Isaac-sim is located at ``/isaac-sim``.


Install OmniDrones in the container:
```
conda activate sim
cd /root/OmniDrones
cp -r conda_setup/etc $CONDA_PREFIX
conda activate sim # re-activate the environment
pip install -e . # install OmniDrones
```

Verify you can successfully run OmniDrones in the container (use ``deploy`` branch):

```
cd /root/OmniDrones/scripts
python train.py headless=true wandb.mode=disabled total_frames=50000 task=Hover
```

## Usage

For usage and more details, please refer to the [documentation](https://omnidrones.readthedocs.io/en/latest/).

Note that for this ``deploy`` branch, it currently supports following environments:

| Environment       | Single-agent or Multi-agent task |
|-------------------|----------------------------------|
| Hover             | Single                           |
| Track             | Single                           |
| InvPendulumHover  | Single                           |
| InvPendulumTrack  | Single                           |
| PayloadTrack      | Single                           |


## Citation

Please cite [this paper](https://arxiv.org/abs/2309.12825) if you use *OmniDrones* in your work:

```
@misc{xu2023omnidrones,
    title={OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control}, 
    author={Botian Xu and Feng Gao and Chao Yu and Ruize Zhang and Yi Wu and Yu Wang},
    year={2023},
    eprint={2309.12825},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

## Ackowledgement

Some of the abstractions and implementation was heavily inspired by [Isaac Orbit](https://github.com/NVIDIA-Omniverse/Orbit).
