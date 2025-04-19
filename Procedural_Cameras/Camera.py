import functools
import transforms3d
import math

import numpy as np
import numpy.typing as npt

from typing import List, Tuple

class Camera:
    def __init__(self, 
                 cam_extrinsic: npt.NDArray[np.float64],
                 cam_intrinsic: npt.NDArray[np.float64]):

        self.cam_ext = cam_extrinsic
        self.cam_intrinsic = cam_intrinsic

        # Extract rotation matrix and translation vector
        R = cam_extrinsic[:, :3]
        t = cam_extrinsic[:, 3]

        self.cam_x = R[0, :]
        self.cam_y = R[1, :]
        self.cam_z = R[2, :]

        # Compute optical center from extrinsic matrix
        self.opt_center = -R.T @ t

        # Reconstruct "look_at" point and up vector for compatibility (optional)
        self.cam_look_at = self.opt_center + self.cam_z
        self.cam_up = self.cam_y

        # Euler angles (negated to match original logic)
        euler = transforms3d.euler.mat2euler(R)
        self.euler = (-euler[0], -euler[1], -euler[2])

        self.screen_w = 2 * cam_intrinsic[0, 2]
        self.screen_h = 2 * cam_intrinsic[1, 2]

    # Project row-wise array of points with camera
    def project_points(self, points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        projected_hom = points_hom @ (self.cam_intrinsic @ self.cam_ext).T
        
        invalid_depth_mask = projected_hom[:, 2] <= 0

        projected_points = projected_hom[:, :2] / projected_hom[:, 2, np.newaxis]

        out_of_frame_mask = (
            (projected_points[:, 0] < 0) | (projected_points[:, 0] >= self.screen_w) |
            (projected_points[:, 1] < 0) | (projected_points[:, 1] >= self.screen_h)
        )

        occluded_flag = np.zeros((projected_points.shape[0], 1))
        occluded_flag[out_of_frame_mask] = 1
        occluded_flag[invalid_depth_mask] = 2

        return np.hstack((projected_points, occluded_flag))

    def plot(self, fig, ax):
        scale = 1

        near_dist = scale
        near_center = self.opt_center + self.cam_z * near_dist

        fov_size = scale * 0.5

        corners = np.array([
            near_center + fov_size * (self.cam_x + self.cam_y),
            near_center + fov_size * (self.cam_x - self.cam_y),
            near_center + fov_size * (-self.cam_x - self.cam_y),
            near_center + fov_size * (-self.cam_x + self.cam_y)
        ])

        edges = [
            [self.opt_center, corners[0]],
            [self.opt_center, corners[1]],
            [self.opt_center, corners[2]],
            [self.opt_center, corners[3]],
            [corners[0], corners[1]],
            [corners[1], corners[2]],
            [corners[2], corners[3]],
            [corners[3], corners[0]]
        ]

        for edge in edges:
            ax.plot3D(*zip(*edge), color="black")

        ax.scatter(self.cam_look_at[0], self.cam_look_at[1], self.cam_look_at[2], c='g', marker='o')
