import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.

from AMCParser.amc_parser import *
from Camera import *

# Create a new figure and a 3D axes object.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def add_sphere(ax, center, radius, resolution=20, color='b', alpha=0.5):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    
    # Parametric equations for a sphere
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='none')

joints = parse_asf("../Datasets/CMU/subjects/01/01.asf")
motions = parse_amc('../Datasets/CMU/subjects/01/01_01.amc')
poses = []

for i in range(len(motions)):
    joints['root'].set_motion(motions[i])
    poses.append(joints['root'].to_cartesian())

cam_poses = generate_cam_seqs(poses, 10)[6]
points = [pose.opt_center for pose in cam_poses]

# Extract x, y, and z coordinates from the list of points.
cam_xs = [p[0] for p in points]
cam_ys = [p[1] for p in points]
cam_zs = [p[2] for p in points]

joints['root'].set_motion(motions[50])
joints['root'].draw(fig, ax)

ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

# Plot the points using a 3D scatter plot.
ax.scatter(cam_xs, cam_ys, cam_zs, c='r', marker='o')

# Label the axes and add a title.
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Scatter Plot of Points')

# Display the plot.
plt.show()
