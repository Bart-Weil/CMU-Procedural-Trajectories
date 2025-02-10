import numpy as np

import numpy.typing as npt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.

from typing import List

from Camera import *
from Procedural_Cameras import *
from AMCParser.amc_parser import *

from tqdm import tqdm

def plot_cam_frames(poses: List[npt.NDArray[np.float64]], cams: List[Camera], filename_prefix="", framerate=120):
    step = 120 // framerate

    # Plot Floor, offset by (x, y) of root joint of first pose in sequence
    floor_x_size = 20
    floor_y_size = 20
    grid_size = 1

    origin = np.array([[poses[0][0, 0] - floor_x_size // 2, poses[0][0, 1] - floor_y_size // 2, 0]])

    cells_x = floor_x_size // grid_size
    cells_y = floor_y_size // grid_size

    x_coords = np.arange(cells_x) * grid_size + origin[0, 0]
    y_coords = np.arange(cells_y) * grid_size + origin[0, 1]
    xx, yy = np.meshgrid(x_coords, y_coords)

    floor_points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())))

    # Plot frames
    n_frames = len(poses) // step
    
    for i in tqdm(range(n_frames)):
        frame = step*i
        projected_pose = cams[frame].project_points(poses[frame])
        projected_floor_points = cams[frame].project_points(floor_points).reshape((cells_x, cells_y, 2))[:, :2] # Remove occlusion flag

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.set_xlim(0, cams[frame].screen_w)
        ax.set_ylim(0, cams[frame].screen_h)

        plot_projected_scene(projected_pose, projected_floor_points, cams[frame].screen_w, cams[frame].screen_h, fig, ax)

        filename = filename_prefix + (f"%0{len(str(n_frames))}d" % i) + ".png"
        plt.savefig(filename)
        plt.close()

# Helper to check if any part of a projected quad is inside screen space
def is_quad_visible(quad, screen_w: int, screen_h: int):
    x_coords, y_coords = zip(*quad)
    return (
        min(x_coords) >= 0 and max(x_coords) <= screen_w and
        min(y_coords) >= 0 and max(y_coords) <= screen_h
    )

def plot_projected_scene(projected_pose: npt.NDArray[np.float64],
                        projected_floor_grid: npt.NDArray[np.float64],
                        screen_w: int, screen_h: int,
                        fig, ax):
    pose_2D = CMU_Pose(projected_pose)
    pose_2D.plot_2D(fig, ax)
    
    # Plot row-wise points
    for x in range(projected_floor_grid.shape[1] - 1):
        for y in range(projected_floor_grid.shape[0] - 1):
            quad = [
                projected_floor_grid[x, y],
                projected_floor_grid[x + 1, y],
                projected_floor_grid[x + 1, y + 1],
                projected_floor_grid[x, y + 1],
            ]
            
            if is_quad_visible(quad, screen_w, screen_h):
                ax.fill(*zip(*quad), color='gray', edgecolor='white')

def plot_projected_pose(projected_pose: npt.NDArray[np.float64], fig, ax):
    pose_2D = CMU_Pose(projected_pose)
    pose_2D.plot_2D(fig, ax)

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
