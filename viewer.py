import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("models/spot.xml")
data = mujoco.MjData(model)

# Disable gravity
model.opt.gravity[:] = 0.0

for i in range(model.ngeom):
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
        model.geom_rgba[i, 3] = 0  # Set alpha to 0

viewer = mujoco.viewer.launch(model, data)

viewer._renderer.viewer.set_background_color(np.array([0.95, 0.95, 0.95, 1.0]))

while True:
    mujoco.mj_step(model, data)
    viewer.sync()
