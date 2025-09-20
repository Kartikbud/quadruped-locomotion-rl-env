import mujoco
import mujoco.viewer
import numpy as np

from inverse_kinematics import get_joint_angles
from trajectory_generator import generate_position_trajectory_point

# -----------------------------
# Load model
# -----------------------------
model = mujoco.MjModel.from_xml_path("models/spot.xml")
data = mujoco.MjData(model)

# Enable gravity
model.opt.gravity[:] = [0, 0, -9.81]

# Optional: transparency for non-mesh geoms
for i in range(model.ngeom):
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
        model.geom_rgba[i, 3] = 0.3

# Optional: add damping
for i in range(model.nv):
    model.dof_damping[i] = 1.0

# -----------------------------
# Gait parameters
# -----------------------------
f_stand = [-0.3891, 6.3763, 14.2038]   # standing pose (cm)

control_freq = 20.0                         # Hz, gait update frequency
control_dt = 1.0 / control_freq

sim_dt = model.opt.timestep                 # simulation timestep (e.g. 0.002s = 500 Hz)
steps_per_control = int(control_dt / sim_dt)

L_span = 6                            # step length (cm)
rho = 0.0                                   # translation angle
angular_vel = 0.0                           # yaw rate
stand_time = 2.0                            # seconds standing
gait_period = 3.0                           # seconds walking

cycle_period = stand_time + gait_period     # full cycle length (s)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("viewer launched")

    step_count = 0
    cycle_clock = 0.0  # resets every cycle

    while viewer.is_running():
        # Update control at control frequency (20 Hz)
        if step_count % steps_per_control == 0:

            phase_t = cycle_clock % cycle_period

            if phase_t < stand_time:
                # --- Standing phase ---
                FL_angles = get_joint_angles(f_stand)
                joint_targets = FL_angles * 4
                print(f"[{cycle_clock:.2f}s] Standing...")
            else:
                # --- Gait phase ---
                gait_t = phase_t - stand_time  # local clock inside gait phase

                # Wrap gait_t to gait_period
                gait_t = gait_t % gait_period

                print(f"[{cycle_clock:.2f}s] Walking... gait_t={gait_t:.2f}")

                # Generate desired foot positions
                FL_pt = generate_position_trajectory_point(L_span, rho, angular_vel, f_stand, gait_t, 0.0)
                FR_pt = generate_position_trajectory_point(L_span, rho, angular_vel, f_stand, gait_t, -1.0)
                BL_pt = generate_position_trajectory_point(L_span, rho, angular_vel, f_stand, gait_t, -1.0)
                BR_pt = generate_position_trajectory_point(L_span, rho, angular_vel, f_stand, gait_t, 0.0)

                # Convert poses from m→cm for IK
                FL_angles = get_joint_angles(FL_pt)
                FR_angles = get_joint_angles(FR_pt)
                BL_angles = get_joint_angles(BL_pt)
                BR_angles = get_joint_angles(BR_pt)

                joint_targets = FL_angles + FR_angles + BL_angles + BR_angles

            # Apply joint targets
            data.ctrl[:] = joint_targets

            # Advance cycle clock
            cycle_clock += control_dt

        # Always step simulation (500 Hz)
        mujoco.mj_step(model, data)
        viewer.sync()
        step_count += 1
