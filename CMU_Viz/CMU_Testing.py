import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.

from AMCParser.amc_parser import *
from Camera import *

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

cam_poses = generate_cam_seqs(poses, 10, 1.5, 1.8, 0.25, 3.5, 1)
points = [pose.opt_center for pose in cam_poses]

# Extract x, y, and z coordinates from the list of points.
xs = [p[0] for p in points]
ys = [p[1] for p in points]
zs = [p[2] for p in points]

# Create a new figure and a 3D axes object.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_zlim(-10, 10)

bound_center, bound_r = get_bounding_circle(poses, 1.5)
add_sphere(ax, [bound_center[0], bound_center[1], 1.0], bound_r)

# Plot the points using a 3D scatter plot.
ax.scatter(xs, ys, zs, c='r', marker='o')

# Label the axes and add a title.
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Scatter Plot of Points')

# Display the plot.
plt.show()
