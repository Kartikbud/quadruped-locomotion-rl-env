import gymnasium as gym
from gymnasium import spaces
import stable_baselines3 as SB3
import numpy as np

import mujoco
import mujoco.viewer
import numpy as np
import math

from inverse_kinematics import get_joint_angles
from alternate_trajectory import generate_position_trajectory_point

#during training only forward motion is used with fixed L_span of 3.5cm and yaw is controlled with proportional controller that maintains a forward heading

"""
- Observation Space: roll, pitch, yaw, [linear accel: x, y, z], [leg_phases: FL, FR, BL, BR] 
    - dim: 10
- Action Space: [{dx, dy, dz for each leg}, clearance height param, ground penetration param] 
    - dim: 14
"""

class QuadEnv(gym.Env):
    metadata = {'render.modes': ["human", "rgb_array"],
                'render_fps': 50}
    
    def __init__(self):
        super().__init__()
        #defining the observation and action spaces according to the above
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)

        #initializing the mujoco model and settings
        self.robot_model = mujoco.MjModel.from_xml_path("models/spot.xml")
        self.robot_data = mujoco.MjData(self.robot_model)
        self.site_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_SITE, "base_link_site")

        self.robot_model.opt.gravity[:] = [0, 0, -9.81] #setting the gravity
        for i in range(self.robot_model.ngeom): #setting the opacity of the collision box geoms to 0.3
            if self.robot_model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
                self.robot_model.geom_rgba[i, 3] = 0.3
        for i in range(self.robot_model.nv):
            self.robot_model.dof_damping[i] = 1.0 #adding damping to the joints to stabilize the movements

        #fixed gait generation parameters
        self.f_stand = [-0.38912402, 6.3763, 14.20384174] #this is the neutral standing position of the foot with respect to its hip for each leg
        self.L_span = 3.5 #half of the stride length
        self.gait_period = 0.4 #how many seconds the swing + support phases is
        self.angular_vel = 0.0 #keeping the angular translation and yaw rotation at 0 for training
        self.rho = 0.0
        self.clearance_limits = [0, 4] #the range of the clearance parameter
        self.penetration_limits = [0, 2] #the range of the penetration parameter
        self.robot_length = 22.93 #length and width of the robot from hip to hip based on official docs
        self.robot_width = 7.6655
        self.phase_offsets = {"FL": 0.0, "FR": 0.5, "BL": 0.5, "BR": 0.0} #defining the phase offsets for each leg

        self.gait_elapsed = 0.0 #keeping track of the gait phase during each trotting sequence

        self.frequency = 50
        self.control_dt = 1/self.frequency

        self.x_pos = self.robot_data.qpos[0] #keeping track of the forward position of the robot
        
    def step(self, action):
        
        mat = self.robot_data.site_xmat[self.site_id].reshape(3, 3) #getting the rotation matrix orientation of the robot from the IMU
        yaw = math.atan2(mat[1, 0], mat[0, 0]) #extracting the yaw angle from the quaternion
        roll  = math.atan2(mat[2,1], mat[2,2])
        pitch = math.atan2(-mat[2,0], math.sqrt(mat[2,1]**2 + mat[2,2]**2))

        yaw_error = self.angular_vel - yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error)) #wrapping the error between [-pi, pi]
        yaw_gain = 3.5
        yaw_correction = yaw_gain * yaw_error
        corrected_angular_vel = yaw_correction
        
        global_phase = (self.gait_elapsed % self.gait_period) / self.gait_period
        joint_targets = []

        clearance = normalize(action[12], [self.action_space.low[12], self.action_space.high[12]], self.clearance_limits)
        penetration = normalize(action[13], [self.action_space.low[13], self.action_space.high[13]], self.penetration_limits)

        corrections = {"FL": action[0:3], "FR": action[3:6], "BL": action[6:9], "BR": action[9:12]}

        for leg in ["FL", "FR", "BL", "BR"]:
            leg_phase = (global_phase + self.phase_offsets[leg]) % 1.0

            if leg_phase < 0.5:
                swing = True
                u = leg_phase * 2.0
            else:
                swing = False
                u = (leg_phase - 0.5) * 2.0

            pt = generate_position_trajectory_point(self.L_span, self.rho, corrected_angular_vel, self.f_stand, u, swing, self.gait_period, self.robot_length, self.robot_width, leg, clearance, penetration)

            corrected_pt = [pt[0] + corrections[leg][0], pt[1] + corrections[leg][1], pt[2] + corrections[leg][2]]

            q = get_joint_angles(corrected_pt)
            joint_targets.extend(q)

        self.robot_data.ctrl[:] = joint_targets
        self.gait_elapsed += self.control_dt

        mujoco.mj_step(self.robot_model, self.robot_data)

        for i in range(int((500/self.frequency)) - 1):
            mujoco.mj_step(self.robot_model, self.robot_data)

        observations = self.get_obs()
        reward = self.get_reward()
        
        terminated = (
            abs(roll) > np.deg2rad(60) or
            abs(pitch) > np.deg2rad(60) or
            self.robot_data.qpos[2] < 0.05  # base hit ground (tune threshold)
        )

        truncated = self.step_count >= self.max_steps
        info = {}

        return observations, reward, terminated, truncated, info

    def get_reward(self):
        rotation_matrix = self.robot_data.site_xmat[self.site_id].reshape(3, 3) #getting the rotation matrix orientation of the robot from the IMU

        #isolating the euler angles of the robot
        roll  = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
        pitch = math.atan2(-rotation_matrix[2,0], math.sqrt(rotation_matrix[2,1]**2 + rotation_matrix[2,2]**2))

        new_x = self.robot_data.qpos[0]
        dx = new_x - self.x_pos

        self.x_pos = new_x

        reward = dx - (10 * (abs(pitch) + abs(roll))) - (0.03 * np.sum(np.abs(self.robot_data.qvel[3:6])))

        return reward
    
    def get_obs(self):
        rotation_matrix = self.robot_data.site_xmat[self.site_id].reshape(3, 3) #getting the rotation matrix orientation of the robot from the IMU

        #isolating the euler angles of the robot
        roll  = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
        pitch = math.atan2(-rotation_matrix[2,0], math.sqrt(rotation_matrix[2,1]**2 + rotation_matrix[2,2]**2))
        yaw   = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])

        #
        lin_acc = self.robot_data.qacc[:3].copy()  # [ax, ay, az]

        global_phase = (self.gait_elapsed % self.gait_period) / self.gait_period
        phases = np.array([
            (global_phase + self.phase_offsets[leg]) % 1.0
            for leg in ["FL", "FR", "BL", "BR"]
            ], dtype=np.float32)
        
        obs = np.concatenate([[roll, pitch, yaw], lin_acc, phases])

        return obs.astype(np.float32)





#-----------------------HELPER FUNCTIONS--------------------------

def normalize(x, old, new):
    return (new[0] + ((x - old[0]) * (new[1] - new[0]) / (old[1] - old[0])))
