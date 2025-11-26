# Agents for Drone Control  
### Reinforcement Learning Agents for Autonomous Drone Navigation

This project explores and compares several deep reinforcement learning (DRL) algorithms for controlling a drone in a custom navigation environment. The main goal is to evaluate which RL method performs best in terms of stability, reward, and safety inside the **Column Traverse Aviary** environment.

---

## Algorithms Implemented

- **PPO (Proximal Policy Optimization)**
- **SAC (Soft Actor-Critic)**
- **DDPG (Deep Deterministic Policy Gradient)**
- **TD3 (Twin Delayed Deep Deterministic Policy Gradient)**

Each agent is trained using Stable-Baselines3 and evaluated under identical conditions.

---

## Environment

Environment used for training:

```
ColumnTraverseAviary  (from column_traverse_env_final.py)
```

Features:
- Continuous control  
- Drone obstacle navigation  
- Collision detection  
- Sparse + shaped rewards  
- Gymnasium-compatible API  

---

## Installation

```bash
git clone https://github.com/sutegi/agents_for_drone_control.git
cd agents_for_drone_control
pip install -r requirements.txt
```

---

## Training Agents

Train PPO:

```bash
python train_ppo.py
```

Train SAC:

```bash
python train_sac.py
```

Train DDPG:

```bash
python train_ddpg.py
```

Train TD3:

```bash
python train_td3.py
```

Metrics:
- Average reward  
- Survival duration  
- Success rate  
- Crashes  
- Trajectory smoothness  

---

## Requirements

```
stable-baselines3
gymnasium
numpy
torch
pybullet
```
