import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.

from AMCParser.amc_parser import *
from Camera import *
from Procedural_Cameras import *
from Plotting import *

joints = parse_asf("../Datasets/CMU/subjects/01/01.asf")
motions = parse_amc('../Datasets/CMU/subjects/01/01_01.amc')

joints['root'].set_motion(motions[0])

poses = []

for i in range(len(motions)):
    joints['root'].set_motion(motions[i])
    poses.append(joints['root'].get_pose().to_numpy())

cam_poses = generate_cam_seqs(poses, 10)[6]

# projected_poses = [cam_poses[i].project_points(poses[i]) for i in range(0, len(poses), 8)]

# for i, pose in enumerate(projected_poses):
#     plot_projected_pose(pose, "Plots/" + ('%05d' % i) + "projected.png")

plot_cam_trajectory(poses, cam_poses)
plot_human_and_cam_pose(poses[1000], cam_poses[1000])

plt.show()