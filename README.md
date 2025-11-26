# Quadruped RL Environment (MuJoCo + Gymnasium)

### Overview
This repository contains a reinforcement learning environment and controller framework designed for training a quadruped robot to walk robustly over varied terrains using **MuJoCo** and **Gymnasium**.

The project is inspired by the dynamics and domain–randomized locomotion training pipeline from:  
**"Dynamics and Domain Randomized Gait Modulation with Bezier Curves for Sim-to-Real Legged Locomotion"
**  
(Peng et al., 2020) — https://arxiv.org/abs/2010.12070

While the original work used PyBullet environments, this project recreates the core concepts in **MuJoCo**, including:
- A Bezier-curve-based gait generator  
- Derived inverse kinematics  
- Terrain randomization  
- Dynamics randomization  
- A controller-modulating reinforcement learning policy  

---

# 🚀 Getting Started

### Environment Requirements
- Python **3.10** (recommended via `venv` or Conda)
- MuJoCo + `mujoco` Python bindings
- Gymnasium
- Stable-Baselines3
- All dependencies listed in `requirements.txt`

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Install Dependencies
```bash
mjpython <script_name>.py
```
