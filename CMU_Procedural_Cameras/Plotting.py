import numpy as np

import numpy.typing as npt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.

from typing import List

from Camera import *
from Procedural_Cameras import *
from AMCParser.amc_parser import *

def plot_projected_pose(projected_pose: npt.NDArray[np.float64], filename=None):
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlim(0, 1280)
    ax.set_ylim(0, 720)

    pose_2D = CMU_Pose(projected_pose)
    pose_2D.plot_2D(fig, ax)
    
    if filename:
        fig.savefig(filename)
        plt.close()

def plot_cam_trajectory(poses: npt.NDArray[np.float64], cams: List[Camera]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    cam_centers = [cam.opt_center for cam in cams]
    cam_look_ats = [cam.cam_look_at for cam in cams]

    cam_xs = [p[0] for p in cam_centers]
    cam_ys = [p[1] for p in cam_centers]
    cam_zs = [p[2] for p in cam_centers]

    cam_look_at_xs = [p[0] for p in cam_look_ats]
    cam_look_at_ys = [p[1] for p in cam_look_ats]
    cam_look_at_zs = [p[2] for p in cam_look_ats]

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    ax.scatter(cam_xs, cam_ys, cam_zs, c='r', marker='o', label='Cam Location')
    ax.scatter(cam_look_at_xs, cam_look_at_ys, cam_look_at_zs, c='g', marker='o', label='Cam Look At')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Camera Location and Camera Look At over time')

def plot_human_and_cam_pose(pose: npt.NDArray[np.float64], cam: Camera):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)

    pose_3D = CMU_Pose(pose)
    pose_3D.plot_3D(fig, ax)

    cam.plot(fig, ax)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Human and Camera Pose')
