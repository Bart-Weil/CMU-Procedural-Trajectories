import functools

import math

import numpy as np
import numpy.typing as npt

import scipy.stats as stats

from typing import List, Tuple

class Camera:
    def __init__(self, opt_center: npt.NDArray[np.float64],
                 cam_look_at: npt.NDArray[np.float64],
                 cam_up: npt.NDArray[np.float64],
                 cam_intrinsic: npt.NDArray[np.float64]):
        
        self.opt_center = opt_center
        self.cam_look_at = cam_look_at
        self.cam_up = cam_up
        self.cam_intrinsic = cam_intrinsic

        self.cam_z = cam_look_at - opt_center
        self.cam_z = self.cam_z / np.linalg.norm(self.cam_z)
        
        self.cam_x = np.cross(self.cam_z, cam_up)
        self.cam_x = self.cam_x / np.linalg.norm(self.cam_x)
        
        self.cam_y = np.cross(self.cam_x, self.cam_z)

        self.screen_w = 2*cam_intrinsic[0, 2]
        self.screen_h = 2*cam_intrinsic[1, 2]

        R = np.vstack([self.cam_x, self.cam_y, self.cam_z]) 
        t = -R @ opt_center
        
        self.cam_ext = np.hstack([R, t.reshape(3, 1)])

    def lookat_to_quaternion(self):
        look_at = self.cam_look_at - self.opt_center
        up = self.cam_up
        # Normalize the forward vector (look-at)
        F = normalize(np.array(look_at))
        
        # Compute the right vector
        R = normalize(np.cross(up, F))
        
        # Compute the orthonormal up vector
        U_prime = np.cross(F, R)
        
        # Build the rotation matrix (3x3)
        rot_matrix = np.array([R, U_prime, F]).T  # Columns are R, U_prime, F

        # Compute trace of the matrix
        tr = np.trace(rot_matrix)
        
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2  # S = 4 * qw
            qw = 0.25 * S
            qx = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
            qy = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
            qz = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
        elif (rot_matrix[0, 0] > rot_matrix[1, 1]) and (rot_matrix[0, 0] > rot_matrix[2, 2]):
            S = np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2  # S = 4 * qx
            qw = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
            qx = 0.25 * S
            qy = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
            qz = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
        elif rot_matrix[1, 1] > rot_matrix[2, 2]:
            S = np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2  # S = 4 * qy
            qw = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
            qx = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
            qy = 0.25 * S
            qz = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
        else:
            S = np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2  # S = 4 * qz
            qw = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
            qx = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
            qy = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
            qz = 0.25 * S

        # Return the quaternion as (x, y, z, w)
        return np.array([qx, qy, qz, qw])
        
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

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm