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
steps_per_control = max(1, int(round(control_dt / sim_dt))) #

L_span = 4.0  # step length (cm)
gait_period = 0.5  # seconds per full gait cycle (swing + support)
angular_vel = -math.pi/6
#angular_vel = 0

length = 22.93
width = 7.6655

# Trot offsets (normalized 0–1, diagonal pairs)
offsets = {"FL": 0.0, "FR": 0.5, "BL": 0.5, "BR": 0.0}

# ✨ Logging setup
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "base_link_site")
logfile = open("alt_path.txt", "w")

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("viewer launched")

    step_count = 0
    gait_elapsed = 0.0  # keeping track of the realtime that has elapsed when starting the controller

    while viewer.is_running():
        if step_count % steps_per_control == 0:
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

                pt = generate_position_trajectory_point(L_span, 0.0, angular_vel, f_stand, u, swing, gait_period, length, width, leg)

                q = get_joint_angles(pt)
                joint_targets.extend(q)

            data.ctrl[:] = joint_targets
            gait_elapsed += control_dt

        # Always step physics
        mujoco.mj_step(model, data)

        # ✨ Log robot base position
        base_pos = data.site_xpos[site_id].copy()
        logfile.write(f"{base_pos[0]} {base_pos[1]} {base_pos[2]}\n")

        viewer.sync()
        step_count += 1

# ✨ Close logfile after viewer exits
logfile.close()
