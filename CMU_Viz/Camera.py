import functools

import math
import numpy as np
import numpy.typing as npt

from typing import List, Tuple

class Camera:
    def __init__(this, opt_center: npt.NDArray[np.float64],
                 cam_look_at: npt.NDArray[np.float64],
                 cam_up: npt.NDArray[np.float64],
                 cam_intrinsic: npt.NDArray[np.float64]):
        
        this.opt_center = opt_center
        this.cam_look_at = cam_look_at
        this.cam_up = cam_up
        this.cam_intrinsic = cam_intrinsic

        cam_z = cam_look_at - opt_center
        cam_z = cam_z / np.linalg.norm(cam_z)
        
        cam_x = np.cross(cam_x, up)
        cam_x = cam_x / np.linalg.norm(cam_x)
        
        cam_y = np.cross(cam_x, cam_z)
        
        R = np.vstack([cam_x, cam_y, -cam_z]) 
        t = -R @ opt_center
        
        cam_ext = np.hstack([R, t.reshape(3, 1)])
        
        this.cam_mat = cam_intrinsic @ cam_ext

    # Project row-wise array of points with camera
    def project_points(this, points: npt.NDArray[np.float64]) -> List[npt.NDArray[np.float64]]:
        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])
        projected_hom = points_hom @ this.cam_mat.T
        projected_points = proj_hom[:, :2] / proj_hom[:, 2, np.newaxis]

        return projected_points

# Capture human poses given camera poses
def capture_poses(poses: List[npt.NDArray[np.float64]], cams: List[Camera]) -> List[npt.NDArray[np.float64]]:
    assert(len(poses) == len(cam_matrices))
    projected_poses = []
    for i in range(len(poses)):
        projected_poses.append(cams[i].project_points(poses[i]))
    return projected_poses

# Procedurally Generate Camera Trajectories
def generate_cam_seqs(poses: List[npt.NDArray[np.float64]],
                      cam_intrinsic: npt.NDArray[np.float64],
                      max_dist: float,
                      min_h: float,
                      max_h: float,
                      rotational_factor: float,
                      lateral_factor: float,
                      n_seqs: int,
                      safe_dist=1.5) -> List[Camera]:
    # Additional parameters
    roll_factor = 0.1
    zoom_factor = 0.05
    n_starting_pos = 4
    min_freq = 2
    max_freq = 6
    smoothing_window = 20

    # Camera Intrinsics
    F_x = 1000
    F_y = 1000
    o_x = 640
    o_y = 360

    seed = 42

    n_frames = len(poses)

    bound_center, bound_r = get_bounding_circle(poses, safe_dist)

    start, end = get_path_endpoints(bound_center, bound_r, dist)

    # Generate camera positions
    path_points = np.linspace(start, end, n_frames)

    lateral_disp = generate_sin_sq_noise(min_freq, max_freq, n_frames, seed)
    vertical_disp = (max_h - min_h)/2 * generate_sin_sq_noise(min_freq, max_freq, n_frames, seed+1) + (max_h - min_h)/2

    path_lateral = (start-end)/2 - bound_center
    path_lateral = path_lateral * lateral_factor/np.linalg.norm(path_lateral)

    path_points_xy = path_points + (lateral_disp[:, np.newaxis] * path_lateral)
    path_points = np.hstack((path_points_xy, vertical_disp))

    # Generate camera orientations
    root_positions = np.array([pose[0, :] for pose in poses])
    tracking_positions = moving_average_rows(root_positions, smoothing_window)

    roll_noise = roll_factor * math.pi * (generate_sin_sq_noise(min_freq, max_freq, n_frames, seed+2) - 0.5)
    tracking_noise_x = generate_sin_sq_noise(min_freq, max_freq, n_frames, seed+3)
    tracking_noise_y = generate_sin_sq_noise(min_freq, max_freq, n_frames, seed+4)
    tracking_noise_z = generate_sin_sq_noise(min_freq, max_freq, n_frames, seed+5)

    tracking_noise = rotational_factor * np.hstack((tracking_noise_x, tracking_noise_y, tracking_noise_z))
    noisy_tracking_positions = tracking_positions + tracking_noise

    cam_opt_centers = list(path_points)
    cam_look_ats = list(noisy_tracking_positions)
    cam_ups = [get_camera_up(cam_look_ats[i], cam_opt_centers[i], roll_noise[i]) for i in range(n_frames)]

    # Generate camera intrinsics
    zoom_noise = zoom_factor*(generate_sin_sq_noise(min_freq, max_freq, n_frames, seed+6)-0.5) + 1
    cam_intrinsics = [get_cam_intrinsic(F_x, F_y, o_x, o_y, zoom_noise[i]) for i in range(n_frames)]

    return [Camera(cam_opt_centers[i], cam_look_ats[i], cam_ups[i], cam_intrinsics[i]) for i in range(n_frames)]

# Moving Average for smoothing camera tracking
def moving_average_rows(arr, window_size, mode='edge'):
    kernel = np.ones(window_size) / window_size
    pad_width = window_size // 2
    
    arr_padded = np.pad(arr, ((0, 0), (pad_width, pad_width)), mode=mode)

    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=1, arr=arr_padded)

# Generate N linspaced samples of composed sin^2 signals of various frequencies
def generate_sin_sq_noise(min_freq: int, max_freq: int, N: int, seed = 42) -> npt.NDArray[np.float64]:
    np.random.seed(seed)
    
    freqs = np.linspace(min_freq, max_freq, N)
    amplitudes = np.random.uniform(0, 1, N)
    t = np.linspace(0, 1, N)
    
    signal = np.zeros_like(t)
    for i in range(N):
        signal += amplitudes[i] * 0.5 * np.sin(2*math.pi*freqs[i]*t - math.pi/2) + 0.5
    signal /= np.max(np.abs(signal))
    
    return signal

def get_path_endpoints(bound_center: npt.NDArray[np.float64], bound_r: float, dist: float) -> Tuple[npt.NDArray[np.float64]]:
    start = np.array([bound_center - bound_r - dist, 0])

    linear_angle = math.asin(bound_r/(bound_r+dist))
    linear_midpoint_dist = (bound_r + dist) * math.cos(linear_angle)
    linear_midpoint = start + linear_midpoint_dist*np.array([math.cos(linear_angle), math.sin(linear_angle)])

    end = 2*linear_midpoint - start
    return start, end

def get_bounding_circle(poses: List[npt.NDArray[np.float64]], safe_dist: float) -> Tuple[npt.NDArray[np.float64], float]:
    roots = np.array([pose[0, :] for pose in poses])
    x_min = roots[:, 0].min()
    x_max = roots[:, 0].max()
    y_min = roots[:, 1].min()
    y_max = roots[:, 1].max()

    bound_center = np.array([x_min + x_max, y_min + y_max])/2
    bound_r = safe_dist + max(x_max - x_min, y_max - y_min)/2
    return bound_center, bound_r
    
def get_camera_up(cam_look_at: npt.NDArray[np.float64], cam_pos: npt.NDArray[np.float64], roll: float) -> npt.NDArray[np.float64]:
    # Camera look_at never vertical for our purposes
    cam_forward = cam_look_at - cam_pos
    world_up = np.array([0, 0, 1])

    cam_right = np.cross(world_up, cam_forward)
    cam_right = cam_right/np.linalg.norm(cam_right)

    return np.cross(cam_forward, cam_right) * np.cos(roll) + cam_right * np.sin(roll)

def get_cam_intrinsic(F_x: float, F_y: float, o_x: float, o_y: float, zoom=1.0, s=0):
    return np.array([
        [zoom*F_x, s,        o_x],
        [0,        zoom*F_y, o_y],
        [0,        0,          1]])