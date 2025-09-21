import mujoco
import mujoco.viewer
import numpy as np
import math

from inverse_kinematics import get_joint_angles
from alternate_trajectory import generate_position_trajectory_point  # your (u, swing) API


model = mujoco.MjModel.from_xml_path("models/spot.xml")
data = mujoco.MjData(model)

# Physics options
model.opt.gravity[:] = [0, 0, -9.81]
for i in range(model.ngeom):
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
        model.geom_rgba[i, 3] = 0.3
for i in range(model.nv):
    model.dof_damping[i] = 1.0

#gait params
f_stand = [-0.3891, 6.3763, 14.2038] # neutral standing stance (cm)

"""
stances = {"FL": [-0.3891, 6.3763, 14.2038], 
           "FR": [-0.3891, -6.3763, 14.2038], 
           "BL": [-0.3891, 6.3763, 14.2038], 
           "BR": [-0.3891, -6.3763, 14.2038]}
"""
           
control_freq = 50.0  # Hz (control update frequency)
control_dt = 1.0 / control_freq

sim_dt = model.opt.timestep # e.g. 0.002 (500 Hz sim)
steps_per_control = max(1, int(round(control_dt / sim_dt))) #

L_span = 4.0  # step length (cm)
rho = math.pi/2
gait_period = 0.5  # seconds per full gait cycle (swing + support)

# Trot offsets (normalized 0–1, diagonal pairs)
# while one pair is in support, the other is in swing
offsets = {"FL": 0.0, "FR": 0.5, "BL": 0.5, "BR": 0.0}

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("viewer launched")

    step_count = 0
    gait_elapsed = 0.0  # keeping track of the realtime that has elapsed when starting the controller

    while viewer.is_running():
        if step_count % steps_per_control == 0: #running the controller at our own frequency rather than 500Hz (mujoco default)
            #normalizing the global phase to be between [0,1]
            global_phase = (gait_elapsed % gait_period) / gait_period

            joint_targets = [] #initializing the empty array for the targets for each joint

            #defining the phase offsets for each leg
            for leg in ["FL", "FR", "BL", "BR"]:
                leg_phase = (global_phase + offsets[leg]) % 1.0 #offseting the local phase for each leg according to the pairings above

                if leg_phase < 0.5:
                    swing = True
                    u = leg_phase * 2.0             # [0,0.5] → [0,1]
                else:
                    swing = False
                    u = (leg_phase - 0.5) * 2.0     # [0.5,1.0] → [0,1]

                #generating the points based on the scaled times from above
                pt = generate_position_trajectory_point(L_span, rho, 0.0, f_stand, u, swing)

                #using the inverse kinematics to get the angles
                q = get_joint_angles(pt)
                joint_targets.extend(q)

            # Apply control
            data.ctrl[:] = joint_targets

            # Advance gait clock
            gait_elapsed += control_dt

        # Always step physics
        mujoco.mj_step(model, data)
        viewer.sync()
        step_count += 1
