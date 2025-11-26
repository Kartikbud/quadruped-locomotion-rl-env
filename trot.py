import mujoco
import mujoco.viewer
import numpy as np
import math

from inverse_kinematics import get_joint_angles
from bezier_trajectory_generator import generate_position_trajectory_point


model = mujoco.MjModel.from_xml_path("models/spot.xml") #loading the model
data = mujoco.MjData(model)

# Physics options
model.opt.gravity[:] = [0, 0, -9.81] #setting the gravity
for i in range(model.ngeom): #setting the opacity of the collision box geoms to 0.3
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
        model.geom_rgba[i, 3] = 0.3
for i in range(model.nv):
    model.dof_damping[i] = 1.0 #adding damping to the joints to stabilize the movements

#gait params
f_stand = [-0.38912402, 6.3763, 14.20384174] # neutral standing stance (cm)

f_stand_cm = {
    "FL": np.array([-0.3891, +6.3763, 14.2038]),
    "FR": np.array([-0.3891, -6.3763, 14.2038]),
    "BL": np.array([-0.3891, +6.3763, 14.2038]),
    "BR": np.array([-0.3891, -6.3763, 14.2038]),
}
           
control_freq = 50.0  # Hz (control update frequency)
control_dt = 1.0 / control_freq

sim_dt = model.opt.timestep # e.g. 0.002 (500 Hz sim)
steps_per_control = max(1, int(round(control_dt / sim_dt))) #since the mujoco sim frequency is much higher than my desired frequency rate I only run the logic after this many steps in simulation

L_span = 3.5  # step length (cm)
gait_period = 0.4  # seconds per full gait cycle (swing + support)
angular_vel = 0.0 # yaw rate
rho = 0.0 # translational angle
clearance = 4 # gait parameters
penetration = 2

length = 22.93
width = 7.6655

# Trot offsets (normalized 0–1, diagonal pairs)
offsets = {"FL": 0.0, "FR": 0.5, "BL": 0.5, "BR": 0.0}

# ✨ Logging setup
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "base_link_site")
logfile = open("alt_path.txt", "w")

desired_yaw = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("viewer launched")

    step_count = 0
    gait_elapsed = 0.0  # keeping track of the realtime that has elapsed when starting the controller

    while viewer.is_running():
        if step_count % steps_per_control == 0:
            #proportional control for the yaw  
            mat = data.site_xmat[site_id].reshape(3, 3) #getting the rotation matrix orientation of the robot from the IMU
            yaw = math.atan2(mat[1, 0], mat[0, 0]) #extracting the yaw angle from the quaternion

            yaw_error = desired_yaw - yaw
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error)) #wrapping the error between [-pi, pi]
            yaw_gain = 3.5
            yaw_correction = yaw_gain * yaw_error
            angular_vel = yaw_correction

            #angular_vel = 0.0
            
            global_phase = (gait_elapsed % gait_period) / gait_period
            joint_targets = []

            for leg in ["FL", "FR", "BL", "BR"]:
                leg_phase = (global_phase + offsets[leg]) % 1.0

                if leg_phase < 0.5:
                    swing = True
                    u = leg_phase * 2.0
                else:
                    swing = False
                    u = (leg_phase - 0.5) * 2.0

                pt = generate_position_trajectory_point(L_span, rho, angular_vel, f_stand, u, swing, gait_period, length, width, leg, clearance, penetration)

                q = get_joint_angles(pt)
                joint_targets.extend(q)

            data.ctrl[:] = joint_targets
            gait_elapsed += control_dt

        # Always step physics
        mujoco.mj_step(model, data)

        #Loging robot base position for trajectory visualization
        base_pos = data.site_xpos[site_id].copy()
        logfile.write(f"{base_pos[0]} {base_pos[1]} {base_pos[2]}\n")

        viewer.sync()
        step_count += 1

# Close logfile after viewer exits
logfile.close()
