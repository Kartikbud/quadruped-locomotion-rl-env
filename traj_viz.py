import numpy as np
import matplotlib.pyplot as plt

# Load both trajectory datasets
data_robot = np.loadtxt("previous.txt")   # Nx3 [x y z]
data_alt   = np.loadtxt("alt_path.txt")     # Nx3 [x y z]

x1, y1, z1 = data_robot[:, 0], data_robot[:, 1], data_robot[:, 2]
x2, y2, z2 = data_alt[:, 0], data_alt[:, 1], data_alt[:, 2]

# --- 2D Top-down plots side by side ---
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].plot(x1, y1, marker=".", markersize=2, linewidth=1, color="blue")
axes[0].set_xlabel("X position (m)")
axes[0].set_ylabel("Y position (m)")
axes[0].set_title("previous Path (Top-Down)")
axes[0].axis("equal")
axes[0].grid(True)

axes[1].plot(x2, y2, marker=".", markersize=2, linewidth=1, color="green")
axes[1].set_xlabel("X position (m)")
axes[1].set_ylabel("Y position (m)")
axes[1].set_title("Alt Path (Top-Down)")
axes[1].axis("equal")
axes[1].grid(True)

plt.tight_layout()
plt.show()
