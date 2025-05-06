import numpy as np
import numpy.typing as npt

from scipy.spatial.transform import Rotation as R

import math
from typing import List, Tuple

from scipy import stats

from Procedural_Cameras.Camera import *
from procedural_params import *
from Procedural_Cameras.Constants import *

CAM_INTRINSIC = np.array([
    [F_x, 0.0, o_x], 
    [0.0, F_y, o_y],
    [0.0, 0.0, 1.0]
])

# Rodrigues rotation matrix
# R = I + sin(w)K + (1 - cos(w))(K^2)
def rotation_about_normal(n, omega):
    cos_w = np.cos(-omega)
    sin_w = np.sin(-omega)

    K = np.array([
        [0, -n[2], n[1]],
        [n[2], 0, -n[0]],
        [-n[1], n[0], 0]
    ])
    
    R = np.eye(3) + sin_w * K + (1 - cos_w) * (K @ K)
    return R


# Generate camera trajectories
def generate_cam_seqs(poses: List[npt.NDArray[np.float64]],
                      n_seqs: int,
                      rng) -> List[List[Camera]]:
    starts = rng.integers(0, len(poses) - SEQ_LEN, size=n_seqs)
    ends = starts + SEQ_LEN

    cam_seqs = []

    for i in range(n_seqs):
        start_frame = starts[i]
        end_frame = ends[i]
        pose_mid = poses[(start_frame + end_frame)//2]
        cam_seq = gen_cam_seq(pose_mid, CAM_INTRINSIC, rng)
        cam_seqs.append([Camera(cam_seq[i], CAM_INTRINSIC) for i in range(SEQ_LEN)])

    return cam_seqs, starts, ends

def get_pose_bounding_joints(cam_mat, pose):
    pos_hom = np.hstack((pose, np.ones((pose.shape[0], 1))))
    pose_projected = pos_hom @ cam_mat.T
    pose_projected /= pose_projected[:, 2][:, np.newaxis]
    pose_min_x_idx = np.argmin(pose_projected[:, 0])
    pose_max_x_idx = np.argmax(pose_projected[:, 0])
    pose_min_y_idx = np.argmin(pose_projected[:, 1])
    pose_max_y_idx = np.argmax(pose_projected[:, 1])

    return np.array([
        pose[pose_min_x_idx],
        pose[pose_max_x_idx],
        pose[pose_min_y_idx],
        pose[pose_max_y_idx]
    ])


def point_to_line_distance(point, line_start, line_end):
    line_vec = line_end - line_start
    point_vec = point - line_start
    line_unit_vec = line_vec / np.linalg.norm(line_vec)
    proj_length = np.dot(point_vec, line_unit_vec)
    proj_point = line_start + proj_length * line_unit_vec
    distance = np.linalg.norm(point - proj_point)
    return distance


def get_min_r_to_capture(point_to_capture, cam_look_at, cam_extrinsic, cam_intrinsic):
    """
    get distance required to move along view direction until point_to_capture is just visible
    """
    x = -cam_extrinsic[:, :3].T @ cam_extrinsic[:, 3]
    min_d = point_to_line_distance(point_to_capture, x, x+cam_look_at)
    # Point angle relative to pixel space x-axis
    point_proj = cam_intrinsic @ cam_extrinsic @ np.hstack((point_to_capture, 1))
    point_proj /= point_proj[2]
    pixel_angle = np.arctan2(point_proj[1], point_proj[0])
    # Assume point is on screen boundary
    dist_to_screen_boundary = min(o_x/abs(np.cos(pixel_angle)), o_y/abs(np.sin(pixel_angle)))
    
    return 2 * max(F_x, F_y) * min_d/dist_to_screen_boundary


def axis_angle_to_matrix(axis_angle):
    """Convert an axis-angle vector to a rotation matrix."""
    angle = np.linalg.norm(axis_angle)
    if angle == 0:
        return np.eye(3)
    axis = axis_angle / angle
    return R.from_rotvec(axis * angle).as_matrix()


def get_cam_poses(x_start, R_start, v, a, omega, alpha, motion_interval, CAM_FPS):
    t = np.linspace(0, motion_interval, int(motion_interval*CAM_FPS))

    xs = np.array([x_start + v*ti + 0.5*a*ti**2 for ti in t])

    Rs = [R_start @ axis_angle_to_matrix(omega*ti + 0.5*alpha*ti**2) for ti in t]

    return [np.hstack((R.T, -R.T @ x.reshape(3, 1))) for R, x in zip(Rs, xs)]

# ------------------------------------------------------------------
# helper: axis–angle  ➜  3 × 3 rotation matrix  (Rodrigues formula)
# ------------------------------------------------------------------
def axis_angle_to_matrix(theta, tol=1e-12):
    """
    theta : (3,) array-like, axis-angle vector  (rad⋅axis)
    return: (3,3) ndarray  rotation matrix
    """
    theta = np.asarray(theta, dtype=float)
    angle = np.linalg.norm(theta)

    if angle < tol:                           # ~ zero rotation
        return np.eye(3)

    axis = theta / angle
    kx, ky, kz = axis
    K = np.array([[   0, -kz,  ky],
                  [  kz,   0, -kx],
                  [ -ky,  kx,   0]])
    return (np.eye(3) +
            np.sin(angle) * K +
            (1.0 - np.cos(angle)) * K @ K)


# ------------------------------------------------------------------
# helper: times when |u t + ½ a t²| is extremal (we wrote this last time)
# ------------------------------------------------------------------
def extremum_times(u, a, *, tol=1e-12):
    """
    Return the positive real roots of ½ a² t² + 3/2(u·a) t + u² = 0.
    They are the instants when the displacement |u t + ½ a t²| is extremal.
    """
    u = np.asarray(u, dtype=float)
    a = np.asarray(a, dtype=float)

    uu = np.dot(u, u)
    ua = np.dot(u, a)
    aa = np.dot(a, a)

    if aa < tol:                 # no acceleration → monotone
        return np.empty(0)

    coeff = [0.5 * aa, 1.5 * ua, uu]   # quadratic in t
    roots = np.roots(coeff)

    roots = roots[np.isreal(roots)].real      # keep real
    return np.sort(roots[roots > tol])        # keep positive

def get_cam_extrema(x_start, R_start,
                    v, a,
                    omega, alpha,
                    motion_interval, *, tol=1e-12):
    """
    Compute camera extrinsics at:
      0.  start (t = 0)
      1.  end   (t = Δt)
      2.  position & orientation extrema inside (0, Δt]

    Returns  a list of three 3×4 camera matrices  [R | -Rᵀ x]
    """
    # --- make all vectors numpy arrays ---
    x_start = np.asarray(x_start, dtype=float)
    v       = np.asarray(v,       dtype=float)
    a       = np.asarray(a,       dtype=float)
    omega   = np.asarray(omega,   dtype=float)
    alpha   = np.asarray(alpha,   dtype=float)
    R_start = np.asarray(R_start, dtype=float)

    # ------------------------------------------------------------------
    # 1) ***POSITION***  — find t_max_pos ∈ (0, Δt] that maximises |Δx|
    # ------------------------------------------------------------------
    t_pos_cand = extremum_times(v, a)
    t_pos_cand = t_pos_cand[t_pos_cand <= motion_interval + tol]

    if t_pos_cand.size == 0:           # monotone → furthest at the end
        t_max_pos = motion_interval
    else:
        # also compare with the distance at t = Δt
        t_all = np.append(t_pos_cand, motion_interval)
        disp = v * t_all[:, None] + 0.5 * a * t_all[:, None]**2   # Δx(t)
        idx = np.argmax(np.linalg.norm(disp, axis=1))
        t_max_pos = t_all[idx]

    x_end = x_start + motion_interval*v + 0.5 * a * motion_interval**2

    max_pos_R = R_start @ axis_angle_to_matrix(omega * t_max_pos + 0.5 * alpha * t_max_pos**2)
    max_pos_mat = np.hstack([max_pos_R.T, -max_pos_R.T @ x_start.reshape((3, 1))])

    # ------------------------------------------------------------------
    # 2) ***ORIENTATION***  — same idea on θ(t) = ω t + ½ α t²
    # ------------------------------------------------------------------
    t_ang_cand = extremum_times(omega, alpha)
    t_ang_cand = t_ang_cand[t_ang_cand <= motion_interval + tol]

    if t_ang_cand.size == 0:
        t_max_ang = motion_interval
    else:
        t_all = np.append(t_ang_cand, motion_interval)
        theta_all = omega * t_all[:, None] + 0.5 * alpha * t_all[:, None]**2
        idx = np.argmax(np.linalg.norm(theta_all, axis=1))
        t_max_ang = t_all[idx]

    theta_end = omega * motion_interval + 0.5 * alpha * motion_interval**2
    theta_max = omega * t_max_ang + 0.5 * alpha * t_max_ang**2

    R_end = R_start @ axis_angle_to_matrix(theta_end)
    R_max = R_start @ axis_angle_to_matrix(theta_max)

    max_ang_pos = x_start + v * t_max_ang + 0.5 * a * t_max_ang**2
    max_ang_mat = np.hstack([R_max.T, -R_max.T @ max_ang_pos.reshape((3, 1))])

    # ------------------------------------------------------------------
    # build the 3 × 4 extrinsic blocks   [ R  |  x ]
    # ------------------------------------------------------------------
    return [np.hstack([R_start.T, -R_start.T @ x_start.reshape((3, 1))]),
            max_pos_mat, max_ang_mat,
            np.hstack([R_end.T, -R_end.T @ x_end.reshape((3, 1))])]


def gen_cam_seq(pose, cam_intrinsics, rng):

    r = rng.uniform(r_min, r_max)
    h = rng.uniform(h_min, h_max)
    theta = rng.uniform(0, 2*np.pi)

    x_mid = np.array([r*np.cos(theta) + pose[0, 0], r*np.sin(theta) + pose[0, 1], h])

    # Sample motion parameters
    v_mid = rng.normal(0, 1, size=3)
    v_mid = rng.uniform(v_min, v_max) * v_mid/np.linalg.norm(v_mid)

    a = rng.normal(0, 1, size=3)
    a = rng.uniform(a_min, a_max) * a/np.linalg.norm(a)

    v_start = v_mid - a*motion_interval/2

    omega_mid = rng.normal(0, 1, size=3)
    omega_mid = rng.uniform(omega_min, omega_max) * omega_mid/np.linalg.norm(omega_mid)

    alpha = rng.normal(0, 1, size=3)
    alpha = rng.uniform(alpha_min, alpha_max) * alpha/np.linalg.norm(alpha)

    omega_start = omega_mid - alpha*motion_interval/2

    x_start = x_mid - v_mid*(motion_interval/2) + 1/2 * a * (motion_interval/2)**2

    # Construct R_mid to look at pose[0]
    forward = pose[0] - x_mid
    forward /= np.linalg.norm(forward)

    # Choose arbitrary up vector that isn't colinear with forward
    world_up = np.array([0, 0, 1])
    if np.abs(np.dot(forward, world_up)) > 0.99:
        world_up = np.array([1, 0, 0])  # fallback if forward is almost vertical
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    R_mid = np.vstack([right, up, forward]).T

    R_start = R_mid @ axis_angle_to_matrix(-omega_mid*(motion_interval/2) + 1/2 * alpha * (motion_interval/2)**2)

    cam_extrema = get_cam_extrema(
        x_start, R_start, v_start, a, omega_start, alpha, motion_interval
    )

    r_capture = r

    for cam_extremum in cam_extrema:
        bounding_joints = get_pose_bounding_joints(cam_extremum, pose)
        for joint in bounding_joints:
            safe_r = get_min_r_to_capture(joint, forward, cam_extremum, cam_intrinsics)
            if safe_r > r_capture:
                r_capture = safe_r

    # Move the camera radially outwards
    x_start += (r_capture * -forward) - (x_mid - pose[0])
    cam_mats = get_cam_poses(
        x_start, R_start, v_start, a, omega_start, alpha, motion_interval, cam_fps
    )

    return cam_mats
