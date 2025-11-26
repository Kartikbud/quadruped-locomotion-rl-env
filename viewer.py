import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("models/alt_spot.xml")
data = mujoco.MjData(model)

for i in range(model.ngeom):
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
        model.geom_rgba[i, 3] = 0.0

for i in range(model.nv):
    model.dof_damping[i] = 1.0

with mujoco.viewer.launch(model, data) as viewer:
    viewer._renderer.viewer.set_background_color(np.array([0.95, 0.95, 0.95, 1.0]))

    while True:
        mujoco.mj_step(model, data)
        viewer.sync()
