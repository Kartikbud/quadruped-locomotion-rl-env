import gymnasium as gym
import stable_baselines3 as SB3
import numpy as np
import mujoco

#during training only forward motion is used with fixed L_span of 3.5cm and yaw is controlled with proportional controller that maintains a forward heading

"""
- Observation Space: roll, pitch, yaw, [linear accel: x, y, z], [leg_phases: FL, FR, BL, BR] 
    - dim: 10
- Action Space: [{dx, dy, dz for each leg}, clearance height param, ground penetration param] 
    - dim: 14
"""

class QuadEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, arg1=0):
        super().__init__()


