# Massive-MARL-Benchmark

## Installation
Download isaacgym and learn about installation instructions from this [website](https://developer.nvidia.com/isaac-gym). Preview Release 3/4 version of IsaacGym is supported. 

### Pre-requisites
The code is tested on Ubuntu 18.04. The Nvidia driver used for code testing is 470.74. Create a new virtual environment using Anaconda.

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 4 install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:
```
pip install -e .
```

## Introduction
Isaac Gym provides a high-performance learning platform that enables training of various agents directly on the GPU. Compared with traditional RL training using CPU simulators and GPU neural networks, Isaac Gym greatly reduces the training time of complex tasks on a single GPU, increasing its training speed by 1-2 orders of magnitude.

This repository contains complex multi-agent control tasks. Multiple ants and ingenuities are embedded in NVIDIA Isaac Gym, providing high performance guarantee for training RL algorithms. Our environment focuses on applying model-free RL/MARL algorithms to control multi-agent operations, which are considered a challenging task for traditional control methods.

## Characteristic

Ants and ingenuities provide a set of multi-agent manipulation tasks and RL algorithms. Better coordination and manipulation of multi-agents is a challenge for researchers. The multi-agent tasks developments have the following characteristics:

* **Isaac Efficiency:** Multiple agents are built into IsaacGym, which supports running multiple agents in thousands of environments simultaneously.
* **Mluti-agents Cooperation:** These multi-agents work together to accomplish the same task. This collaboration is asynchronous. In the asynchronous case, agents can perform actions independently of each other, but still need to communicate and coordinate with each other to ensure that they ultimately reach a common goal.
* **Task Diversity:** We introduce various manipulation tasks for multi-agents (e.g., circles, pushing, paralleling...); the diversity of tasks allows for better testing of RL algorithms.
* **Quick Demos:**

## Tasks
Source code for tasks can be found in `envs/tasks`. So far, we have released the following tasks including multiple ants in circling, pushing, and ingenuities in parallel.

## Training
### Training Examples
#### RL/MARL Examples
For example, if you want to train a policy for the Agents task by the PPO algorithm, run this line in `agents` folder:
```
python train.py --task=TenAnt --algo=ppo
```

## Testing
### Testing Examples
#### RL/MARL Examples
The trained model will be saved to `logs/${Task Name}/${Algorithm Name}`folder.

To load a trained model and only perform inference (no training), pass `--test` as an argument, and pass `--model_dir` to specify the trained models which you want to load. For agents reinforcement learning, you need to pass `--model_dir` to specify exactly what .pt model you want to load. An example of PPO algorithm is as follows:
```
python train.py --task=TenAnt --algo=ppo --model_dir=logs/ten_ant/ppo/ppo_seed0/model_5000.pt --test
```

