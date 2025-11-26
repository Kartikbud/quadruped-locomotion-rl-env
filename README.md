Quadruped RL Environment (MuJoCo + Gymnasium)
Overview

This repository contains a reinforcement learning environment and controller framework for training a quadruped robot to walk robustly over varied terrains using MuJoCo and Gymnasium.
The project is inspired by the domain–randomized training pipeline from “Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning” (Peng et al., 2020) arXiv:2010.12070
.

Unlike the original paper—whose environments were PyBullet-based—this project implements a similar idea in MuJoCo, including trajectory generation, inverse kinematics, randomized dynamics, and a controller-modulating RL policy.

🚀 Getting Started
Environment Requirements

Python 3.10 (recommended via venv or Anaconda)

MuJoCo + mujoco-python bindings

Gymnasium

Stable-Baselines3

Dependencies listed in requirements.txt

Install Dependencies
pip install -r requirements.txt

Running the Scripts

Scripts are executed using MuJoCo’s Python launcher:

mjpython <script_name>.py

📁 Script Descriptions
File	Description
trot.py	Runs the baseline trot controller on flat terrain.
env_test.py	Runs the baseline controller inside the randomized environment (no policy).
train.py	Trains a PPO policy and saves models to /final_models.
inference.py	Loads trained policies and demonstrates controller modulation.
🎥 Demo Videos (Placeholders — replace with embeds)

Baseline Controller — Flat Terrain
[Insert video/GIF here]

Baseline Controller — Randomized Terrain (No Policy)
[Insert video/GIF here]

Trained Policy Modulating Controller — Randomized Terrain
[Insert video/GIF here]

🧩 Project Motivation & Design
Why This Project?

The referenced paper shows that repeatedly randomizing environment dynamics strengthens the robustness and sim2real transfer of locomotion policies.
This project re-creates that concept in MuJoCo, with a controller-modulating RL approach instead of raw action control.

🦿 Controller & Trajectory Generation
Bezier Curve Trajectory

Two quadratic Bezier curves define each leg’s motion:

Swing phase

Support phase

Parameters:

Foot clearance

Foot penetration

L_span (step length)

These curves produce foot trajectories which are converted to joint angles via derived inverse kinematics.

Trot Gait

Diagonal legs move as pairs

Second pair offset by ½ gait cycle → improves dynamic stability

Robot Model

Based on this open-source quadruped:
https://github.com/adham-elarabawy/open-quadruped

Converted URDF → MuJoCo XML

Policy Role

Instead of producing torques/positions, the policy modulates the existing controller by outputting:

Adjusted Bezier parameters

Small residual (x, y, z) offsets per foot

🌍 The Custom Gymnasium Environment
Terrain

Flat plane with a randomized height map

Encourages adaptation to uneven ground and perturbations

Observations Provided to the Policy

The policy receives a compact observation vector containing:

IMU Data

Linear accelerations

Roll, pitch, yaw (orientation angles)

Gait Information

Phase of each leg within its Bezier trajectory cycle
(normalized 0 → 1 for smooth periodic behavior)

These observations provide enough information for the policy to stabilize gait timing and body orientation without requiring full joint-state feedback.

Domain & Dynamics Randomization

Triggered periodically during training:

Ground and foot friction

Body, leg, and foot mass properties

Height map x-, y-, and z-axis scaling

Randomization uses Gaussian noise (σ = 0.2)

This produces robust, generalizable locomotion.

🎯 Reward Structure

The reward encourages stable, forward walking while discouraging unstable movements.

Positive Rewards

Forward velocity (progress in x-direction)

Penalties

Large pitch or roll angles

High angular velocities around roll, pitch, and yaw axes

Excessive base rotation or jitter
These elements promote smooth, controlled locomotion rather than chaotic movement.

✔️ Training Results

PPO (Stable-Baselines3) successfully learns:

Smoother, more stable gait patterns

Improved performance on varied terrain

Stronger robustness to perturbations and dynamics shifts

📝 Notes & Future Work

Even though training focuses on forward motion, lateral and rotational skills often improve as a natural byproduct.

Potential future additions:

Multiple gaits (pace, bound, gallop)

Complex terrains (rocks, steps, slopes)

More IK/controller refinement

Curriculum learning
