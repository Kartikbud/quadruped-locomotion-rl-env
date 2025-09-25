import mujoco
import mujoco.viewer
import numpy as np

# Load your model
model = mujoco.MjModel.from_xml_path("models/alt_spot.xml")
data = mujoco.MjData(model)

# Optional: make non-mesh geoms slightly transparent
for i in range(model.ngeom):
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
        model.geom_rgba[i, 3] = 0.3  # semi-transparent

# Optional: adjust joint damping for stability
for i in range(model.nv):
    model.dof_damping[i] = 1.0  # adds some resistance to prevent wild movements

# Disable gravity
#model.opt.gravity[:] = [0, 0, 0]


# Launch interactive viewer (this one lets you use control tab)
with mujoco.viewer.launch(model, data) as viewer:
    # Optional: set background color
    viewer._renderer.viewer.set_background_color(np.array([0.95, 0.95, 0.95, 1.0]))

    while True:
        mujoco.mj_step(model, data)  # step physics
        viewer.sync()                # sync frame with viewer
