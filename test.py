import mujoco
import mujoco.viewer
import numpy as np

# -----------------------------
# Load model
# -----------------------------
model = mujoco.MjModel.from_xml_path("models/spot.xml")
data = mujoco.MjData(model)

# Target actuators
actuator_names = [
    "motor_front_left_lower_leg_ctrl",
    "motor_front_right_lower_leg_ctrl",
    "motor_back_left_lower_leg_ctrl",
    "motor_back_right_lower_leg_ctrl",
]
actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in actuator_names]
print("Actuators:", actuator_names)
print("IDs:", actuator_ids)

for i in range(model.ngeom):
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
        model.geom_rgba[i, 3] = 0.3  # semi-transparent

# -----------------------------
# Interpolation setup
# -----------------------------
start_pos = 0.0
end_pos = -0.37
n_steps = 20                   # number of control updates per ramp
control_freq = 20.0            # Hz
control_dt = 1.0 / control_freq

# MuJoCo timestep (usually 0.002 → 500 Hz)
sim_dt = model.opt.timestep
steps_per_control = int(round(control_dt / sim_dt))  # ~25 sim steps per control update

# Precompute one ramp down and one ramp up
ramp_down = np.linspace(start_pos, end_pos, n_steps, endpoint=True)
ramp_up = np.linspace(end_pos, start_pos, n_steps, endpoint=True)
positions = np.concatenate([ramp_down, ramp_up])  # 40 steps total

# -----------------------------
# Viewer loop
# -----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("viewer launched")

    step_count = 0
    control_step = 0

    while viewer.is_running():
        # Update control every steps_per_control sim steps (20 Hz)
        if step_count % steps_per_control == 0:
            cmd = positions[control_step]

            for aid in actuator_ids:
                data.ctrl[aid] = cmd

            print(f"control step {control_step}, target={cmd:.3f}")

            control_step = (control_step + 1) % len(positions)  # loop forever

        # Always step simulation (500 Hz)
        mujoco.mj_step(model, data)
        viewer.sync()
        step_count += 1
