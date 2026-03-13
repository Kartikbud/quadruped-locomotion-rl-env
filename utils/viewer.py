from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def main():
    repo_root = Path(__file__).resolve().parents[1]
    model_xml = repo_root / "robot" / "alt_spot.xml"
    model = mujoco.MjModel.from_xml_path(str(model_xml))
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


if __name__ == "__main__":
    main()
