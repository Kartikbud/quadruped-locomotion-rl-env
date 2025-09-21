import mujoco
import mujoco.viewer
import numpy as np

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


f_stand = [-0.3891, 6.3763, 14.2038]   # neutral stance (cm)

control_freq = 20.0                      # Hz (control update frequency)
control_dt   = 1.0 / control_freq

sim_dt = model.opt.timestep              # e.g. 0.002 (500 Hz sim)
steps_per_control = max(1, int(round(control_dt / sim_dt))) #

L_span      = 4.0                        # step length (cm)
gait_period = 0.5                        # seconds per full gait cycle (swing + support)

# Trot offsets (normalized 0–1, diagonal pairs)
offsets = {"FL": 0.0, "FR": 0.5, "BL": 0.5, "BR": 0.0}

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("viewer launched")

    step_count   = 0
    gait_elapsed = 0.0  # advances continuously

    while viewer.is_running():
        if step_count % steps_per_control == 0:
            # 1) Global normalized phase in [0,1]
            global_phase = (gait_elapsed % gait_period) / gait_period

            joint_targets = []

            # 2) Each leg follows swing/support with offsets
            for leg in ["FL", "FR", "BL", "BR"]:
                leg_phase = (global_phase + offsets[leg]) % 1.0

                if leg_phase < 0.5:
                    swing = True
                    u = leg_phase * 2.0             # [0,0.5] → [0,1]
                else:
                    swing = False
                    u = (leg_phase - 0.5) * 2.0     # [0.5,1.0] → [0,1]

                # 3) Trajectory point from generator
                pt = generate_position_trajectory_point(L_span, 0.0, 0.0, f_stand, u, swing)

                # 4) Inverse kinematics to joint angles
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
