import gymnasium as gym
import stable_baselines3 as SB3
import numpy as np

"""
- Observation Space: roll, pitch, yaw, [linear accel: x, y, z], [leg_phases: FL, FR, BL, BR]
- Action Space: 
"""

class QuadEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, arg1=0):
        super().__init__()


