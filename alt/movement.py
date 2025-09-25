import mujoco
import mujoco.viewer
import numpy as np
import math

from alt.new_ik import get_joint_angles
from alt.new_trajectory import generate_position_trajectory_point  # your (u, swing) API


model = mujoco.MjModel.from_xml_path("models/alt_spot.xml")
data = mujoco.MjData(model)

# Physics options
model.opt.gravity[:] = [0, 0, -9.81]
for i in range(model.ngeom):
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
        model.geom_rgba[i, 3] = 0.3
for i in range(model.nv):
    model.dof_damping[i] = 1.0

#gait params
#hip-to-foot for neutral stance
htf_vecs = { 
    "FL": np.array([-0.3890840259, 6.3763, -13.06694293]),
    "FR": np.array([-0.3890840259, -6.3763, -13.06694293]),
    "BL": np.array([-0.3890840259, 6.3763, -13.06694293]),
    "BR": np.array([-0.3890840259, -6.3763, -13.06694293]),
}

#center-to-hip
#body params:
length = 22.93
width = 7.6655
cth_vecs = {
    "FL": np.array([length/2, width/2, 0]),
    "FR": np.array([length/2, -width/2, 0]),
    "BL": np.array([-length/2, width/2, 0]),
    "BR": np.array([-length/2, -width/2, 0])
}

#--gpt
# cth_vecs = {
#     "FL": np.array([  9.15,  3.94, 0.0 ]),  # front-left
#     "FR": np.array([  9.15, -3.94, 0.0 ]),  # front-right
#     "BL": np.array([-13.65,  3.94, 0.0 ]),  # back-left
#     "BR": np.array([-13.65, -3.94, 0.0 ]),  # back-right
# }


#center-to-foot (f_stand) neutral stance for each foot relative to the center of the frame
ctf_vecs = {
    "FL": htf_vecs["FL"] + cth_vecs["FL"],
    "FR": htf_vecs["FR"] + cth_vecs["FR"],
    "BL": htf_vecs["BL"] + cth_vecs["BL"],
    "BR": htf_vecs["BR"] + cth_vecs["BR"]
}

#---prior vealues before restrcuturing
# f_stand_cm = {
#     "FL": np.array([-0.3891, +6.3763, 14.2038]),
#     "FR": np.array([-0.3891, -6.3763, 14.2038]),
#     "BL": np.array([-0.3891, +6.3763, 14.2038]),
#     "BR": np.array([-0.3891, -6.3763, 14.2038]),
# }
           
control_freq = 50.0  # Hz (control update frequency)
control_dt = 1.0 / control_freq

sim_dt = model.opt.timestep # e.g. 0.002 (500 Hz sim)
steps_per_control = max(1, int(round(control_dt / sim_dt))) #

L_span = 4.0  # step length (cm)
rho = math.pi/2
rho = 0.0
gait_period = 0.5  # seconds per full gait cycle (swing + support)

# Trot offsets (normalized 0–1, diagonal pairs)
# while one pair is in support, the other is in swing
offsets = {"FL": 0.0, "FR": 0.5, "BL": 0.5, "BR": 0.0}

# ✨ open a log file and get site id
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "base_link_site")
logfile = open("robot_path.txt", "w")

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
                pt = generate_position_trajectory_point(L_span, rho, 0.0, ctf_vecs[leg], u, swing, control_dt) #this is ctf

                y_factor = 1
                if leg in ["FR", "BR"]: 
                    y_factor = -1

                #converting the ctf to htf
                pt = pt - cth_vecs[leg]                

                #using the inverse kinematics to get the angles
                q = get_joint_angles([pt[0], y_factor*pt[1], -pt[2]])
                if leg in ["FR", "BR"]:
                    q[0] = -q[0]  # flip abduction sign
                joint_targets.extend(q)

            # Apply control
            data.ctrl[:] = joint_targets

            # Advance gait clock
            gait_elapsed += control_dt

        # Always step physics
        mujoco.mj_step(model, data)

        # ✨ record robot base position (x,y,z)
        base_pos = data.site_xpos[site_id].copy()
        logfile.write(f"{base_pos[0]:.4f} {base_pos[1]:.4f} {base_pos[2]:.4f}\n")

        viewer.sync()
        step_count += 1

# ✨ close log file after simulation ends
logfile.close()
