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
    metadata = {'render_modes': ["human", "rgb_array"],
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
        self.robot_model.body("base_link").pos[:] = [0.0, 0.0, 0.15]
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
        self.clearance_limits = [0, 7] #the range of the clearance parameter
        self.penetration_limits = [0, 4] #the range of the penetration parameter
        self.robot_length = 22.93 #length and width of the robot from hip to hip based on official docs
        self.robot_width = 7.6655
        self.phase_offsets = {"FL": 0.0, "FR": 0.5, "BL": 0.5, "BR": 0.0} #defining the phase offsets for each leg

        self.gait_elapsed = 0.0 #keeping track of the gait phase during each trotting sequence

        self.frequency = 50
        self.control_dt = 1/self.frequency

        self.x_pos = self.robot_data.qpos[0] #keeping track of the forward position of the robot

        self.max_steps = 2000
        self.step_count = 0

        self.viewer = None

        #making a copy of the default values of frictions and masses so that during the reset phase where dynamics and the domain is randomized it is done with respect to the original values
        self.default_body_mass = self.robot_model.body_mass.copy()
        self.default_friction = self.robot_model.geom_friction.copy()
        if self.robot_model.nhfield > 0:
            self.default_hfield = self.robot_model.hfield_data.copy()
        else:
            self.default_hfield = None
        
        self._hfield_dirty = False

        self.ref_height = self.robot_data.qpos[2].copy()  # baseline standing height

        
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
        self.step_count += 1

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
        
        forward_reward_gain = 20
        dx = forward_reward_gain * (new_x - self.x_pos)

        self.x_pos = new_x

        reward = dx - (10 * (abs(pitch) + abs(roll))) - (0.03 * np.sum(np.abs(self.robot_data.qvel[3:6])))

        current_height = self.robot_data.qpos[2]
        height_deviation = abs(current_height - self.ref_height)

        # Strong penalty if it jumps more than 0.5 m above reference
        # if current_height > (self.ref_height + 0.5):
        #     reward -= 50.0 * (current_height - (self.ref_height + 0.5))
        # else:
        #     reward -= 1.0 * height_deviation

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
    
    def reset(self, seed=None, options=None):

        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

        super().reset(seed=seed)

        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Restore defaults before applying randomization so values don't drift across resets
        self.robot_model.body_mass[:] = self.default_body_mass
        self.robot_model.geom_friction[:] = self.default_friction

        for i in range(self.robot_model.nbody): #randomizing the mass of each body part to introduce randomization of the robot dynamics
            scale = self.np_random.uniform(0.8, 1.2)
            self.robot_model.body_mass[i] *= scale
        
        for i in range(self.robot_model.ngeom):
            geom_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, i)

            if "foot" in geom_name:
                # Test with higher traction
                self.robot_model.geom_friction[i, 0] = self.np_random.uniform(2.0, 2.5)
                self.robot_model.geom_friction[i, 1] = self.default_friction[i, 1]  # keep torsional same
                self.robot_model.geom_friction[i, 2] = self.default_friction[i, 2]  # keep rolling same

            elif "ground" in geom_name:
                # Raise ground friction to test traction limits
                self.robot_model.geom_friction[i, 0] = self.np_random.uniform(1.5, 1.9)

            else:
                # Keep body friction modest as before
                self.robot_model.geom_friction[i, 0] = self.np_random.uniform(0.5, 0.7)


        if self.robot_model.nhfield > 0: #randomizing the height field of the ground to add some terrain/domain randomization for higher robustness
            nrows = int(self.robot_model.hfield_nrow[0])
            ncols = int(self.robot_model.hfield_ncol[0])
            size = nrows * ncols

            base = self.default_hfield[:size].copy().reshape(nrows, ncols)
            sz = float(self.robot_model.hfield_size[0, 2])
            if sz <= 0.0:
                sz = 1.0  # fallback to avoid divide-by-zero

            # Convert baseline heights to meters before adding directional noise
            terrain_m = base * sz

            axis_range = 0.04  # 4 cm variation per-axis
            noise = self.np_random.uniform(-axis_range, axis_range, size=(nrows, ncols))

            terrain_m += noise
            terrain_m = np.clip(terrain_m, -axis_range, axis_range)

            # Shift back into [0, 1] for MuJoCo after scaling by sz
            normalized = np.clip((terrain_m / sz) + 0.5, 0.0, 1.0)
            self.robot_model.hfield_data[:size] = normalized.ravel()

            # Mark for viewer re-upload
            self._hfield_dirty = True
            #mujoco.mj_uploadHField(self.robot_model, 0)


        mujoco.mj_resetData(self.robot_model, self.robot_data) #resetting the world and robot model
        self.robot_data.qpos[2] = 0.15
        self.robot_data.qvel[:] = 0
        self.gait_elapsed = 0.0 #resetting gait elapsed counter
        self.x_pos = self.robot_data.qpos[0] #resetting the position after the robot goes back to starting position
        self.step_count = 0  # reset step counter
        obs = self.get_obs() #returning the new initial set of observations
        return obs, {}
    
    def render(self, mode="human"):
        if mode == "human":
            if self.viewer is None:
                # Open exactly once; never relaunch here
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.robot_model, self.robot_data)
                except RuntimeError as e:
                    # If UI thread hasn’t fully closed yet, just skip this frame
                    if "another MuJoCo viewer is already open" in str(e):
                        return
                    raise

            try:
                self.viewer.sync()
            except Exception as e:
                print(f"Viewer sync error: {e}")
                try:
                    self.viewer.close()
                except Exception:
                    pass
                self.viewer = None

    
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None




#-----------------------HELPER FUNCTIONS--------------------------

def normalize(x, old, new):
    return (new[0] + ((x - old[0]) * (new[1] - new[0]) / (old[1] - old[0])))
