import mujoco
import mujoco.viewer
import numpy as np

# -------------------- IK inputs (base/hip-to-foot vectors) --------------------
htf_vecs = { 
    "FL": np.array([-0.3890840259, 6.3763, 13.06694293]),
    "FR": np.array([-0.3890840259, 6.3763, 13.06694293]),
    "BL": np.array([-0.3890840259, 6.3763, 13.06694293]),
    "BR": np.array([-0.3890840259, 6.3763, 13.06694293]),
}

# -------------------- Load model & IK --------------------
model = mujoco.MjModel.from_xml_path("models/alt_spot.xml")
data = mujoco.MjData(model)

from alt.new_ik import get_joint_angles  # expected to return [hip, upper, lower] in radians

# -------------------- Visual & damping niceties --------------------
for i in range(model.ngeom):
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
        model.geom_rgba[i, 3] = 0.3

for i in range(model.nv):
    model.dof_damping[i] = 1.0

# Optional: disable gravity if desired
# model.opt.gravity[:] = [0, 0, 0]

# -------------------- Leg & joint mapping (from your XML) --------------------
LEG_TO_JOINTS = {
    "FL": ["motor_front_left_hip",  "motor_front_left_upper_leg",  "motor_front_left_lower_leg"],
    "FR": ["motor_front_right_hip", "motor_front_right_upper_leg", "motor_front_right_lower_leg"],
    "BL": ["motor_back_left_hip",   "motor_back_left_upper_leg",   "motor_back_left_lower_leg"],
    "BR": ["motor_back_right_hip",  "motor_back_right_upper_leg",  "motor_back_right_lower_leg"],
}

# (Position) actuator names that target those joints
JOINT_TO_ACTUATOR = {
    "motor_front_left_hip":        "motor_front_left_hip_ctrl",
    "motor_front_left_upper_leg":  "motor_front_left_upper_leg_ctrl",
    "motor_front_left_lower_leg":  "motor_front_left_lower_leg_ctrl",
    "motor_front_right_hip":       "motor_front_right_hip_ctrl",
    "motor_front_right_upper_leg": "motor_front_right_upper_leg_ctrl",
    "motor_front_right_lower_leg": "motor_front_right_lower_leg_ctrl",
    "motor_back_left_hip":         "motor_back_left_hip_ctrl",
    "motor_back_left_upper_leg":   "motor_back_left_upper_leg_ctrl",
    "motor_back_left_lower_leg":   "motor_back_left_lower_leg_ctrl",
    "motor_back_right_hip":        "motor_back_right_hip_ctrl",
    "motor_back_right_upper_leg":  "motor_back_right_upper_leg_ctrl",
    "motor_back_right_lower_leg":  "motor_back_right_lower_leg_ctrl",
}

# -------------------- Helpers --------------------
def name2id_joint(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

def name2id_actuator(name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

def joint_current_qpos(jname: str) -> float:
    jid = name2id_joint(jname)
    adr = model.jnt_qposadr[jid]
    return float(data.qpos[adr])

def joint_limits(jname: str):
    """Return (min,max) if limited; else (-inf, inf)."""
    jid = name2id_joint(jname)
    if model.jnt_limited[jid] == 1:
        lo, hi = model.jnt_range[jid]
        return float(lo), float(hi)
    return -np.inf, np.inf

def clip_to_limits(jname: str, val: float) -> float:
    lo, hi = joint_limits(jname)
    return float(np.clip(val, lo, hi))

def ik_for_leg(leg_key: str, vec: np.ndarray):
    """Call IK robustly; try (vec, leg_name) then fallback to (vec)."""
    try:
        return np.asarray(get_joint_angles(vec, leg_key), dtype=float)
    except TypeError:
        return np.asarray(get_joint_angles(vec), dtype=float)

def smoothstep01(x: float) -> float:
    """C^1 and C^2 continuous ease: 3x^2 - 2x^3"""
    x = np.clip(x, 0.0, 1.0)
    return x*x*(3 - 2*x)

# -------------------- Compute IK targets & setup interpolation --------------------
# Gather target joint angles per joint name
target_angles_by_joint = {}

for leg, joints in LEG_TO_JOINTS.items():
    tgt = ik_for_leg(leg, htf_vecs[leg])  # [hip, upper, lower] (radians)
    if tgt.shape != (3,):
        raise ValueError(f"IK for leg {leg} must return 3 angles, got shape {tgt.shape}")

    for jname, angle in zip(joints, tgt):
        target_angles_by_joint[jname] = clip_to_limits(jname, float(angle))

# Read current (start) angles
start_angles_by_joint = {j: joint_current_qpos(j) for j in target_angles_by_joint.keys()}

# Pre-resolve actuator ids for speed
actuator_ids = {j: name2id_actuator(JOINT_TO_ACTUATOR[j]) for j in target_angles_by_joint.keys()}

# Interp settings
TRANSITION_TIME = 1.5  # seconds to smoothly blend into IK pose
hold_targets_after = True  # stay at IK after transition

# -------------------- Launch viewer --------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    # nice bright background
    #viewer._renderer.viewer.set_background_color(np.array([0.95, 0.95, 0.95, 1.0]))
    # start time of the transition
    start_time = data.time

    while True:
        # progress in [0,1]
        t = data.time - start_time
        if TRANSITION_TIME > 0.0:
            alpha = smoothstep01(t / TRANSITION_TIME)
        else:
            alpha = 1.0

        # blend and command each joint via its position actuator
        for jname, tgt in target_angles_by_joint.items():
            start = start_angles_by_joint[jname]
            desired = (1.0 - alpha) * start + alpha * tgt
            desired = clip_to_limits(jname, desired)
            aid = actuator_ids[jname]
            data.ctrl[aid] = desired

        # advance sim & draw
        mujoco.mj_step(model, data)
        viewer.sync()

        # If we’ve finished the transition, optionally pin start angles to targets
        # so if you later swap htf_vecs on the fly, it restarts smoothly from where it is.
        if hold_targets_after and t >= TRANSITION_TIME:
            start_angles_by_joint = {j: joint_current_qpos(j) for j in target_angles_by_joint.keys()}
            start_time = data.time  # reset so if target updates later you still get a smooth blend
