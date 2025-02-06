import functools

import math

import numpy as np
import numpy.typing as npt

import scipy.stats as stats

from typing import List, Tuple

from Camera import *

# Generate camera trajectories
def generate_cam_seqs(poses: List[npt.NDArray[np.float64]],
                      n_seqs: int,
                      seed: int = 42) -> List[List[Camera]]:
    start_increment = 30 # Start point rotation increment (deg)
    start_angle = 0

    rng = np.random.default_rng(seed)

    cam_seqs = []

    for i in range(n_seqs):
        dist = rng.uniform(low=0, high=10)
        h_range = rng.uniform(low=0.25, high=10, size=2)
        min_h = float(h_range.min())
        max_h = float(h_range.max())
        rot_factor = rng.uniform(low=0.1, high=0.4)
        lat_factor = rng.uniform(low=2, high=6)
        safe_dist = rng.uniform(low=0.25, high=8)

        cam_seqs.append(generate_cam_seq(poses, start_angle, dist, min_h, max_h, rot_factor, lat_factor, safe_dist, rng))

        start_angle += start_increment

    return cam_seqs

# Procedurally Generate Camera Trajectories from a given starting point
def generate_cam_seq(poses: List[npt.NDArray[np.float64]],
                     start_angle: float,
                     dist: float,
                     min_h: float,
                     max_h: float,
                     rotational_factor: float,
                     lateral_factor: float,
                     safe_dist: float,
                     rng) -> List[Camera]:
    # Additional parameters
    roll_factor = 0.05 *  math.pi/180
    zoom_factor = 0.05
    smoothing_window = 81
    assert(smoothing_window % 2 == 1)

    # Camera Intrinsics
    F_x = 1000
    F_y = 1000
    o_x = 640
    o_y = 360

    n_frames = len(poses)

    bound_center, bound_r = get_bounding_sphere(poses, safe_dist)

    start, end = get_path_endpoints(start_angle, bound_center, bound_r, dist)

    # Generate camera positions
    disp = generate_sin_noise(1/n_frames, 0.1, 3, n_frames, rng)
    path_points = start + (end - start)*disp[:, np.newaxis]

    lateral_disp = generate_sin_noise(1/n_frames, 1, 5, n_frames, rng)
    vertical_disp = (max_h - min_h)*(generate_sin_noise(20/n_frames, 5, 8, n_frames, rng)-0.5) + (max_h + min_h)/2

    path_lateral = (start+end)/2 - bound_center
    path_lateral = path_lateral * lateral_factor/np.linalg.norm(path_lateral)

    path_points_xy = path_points + (lateral_disp[:, np.newaxis] * path_lateral)
    path_points = np.hstack((path_points_xy, vertical_disp[:, np.newaxis]))

    # Generate camera orientations
    root_positions = np.array([pose[0, :] for pose in poses])
    tracking_positions = moving_average_rows(root_positions, smoothing_window)

    roll_noise = roll_factor * 2 * math.pi * (generate_sin_noise(30/n_frames, 5, 5, n_frames, rng) - 0.5)
    tracking_noise_x = generate_sin_noise(15/n_frames, 5, 5, n_frames, rng)
    tracking_noise_y = generate_sin_noise(15/n_frames, 5, 5, n_frames, rng)
    tracking_noise_z = generate_sin_noise(15/n_frames, 5, 5, n_frames, rng)

    tracking_noise = rotational_factor * np.vstack((tracking_noise_x, tracking_noise_y, tracking_noise_z)).T
    noisy_tracking_positions = tracking_positions + tracking_noise

    cam_opt_centers = list(path_points)
    cam_look_ats = list(noisy_tracking_positions)
    cam_ups = [get_camera_up(cam_look_ats[i], cam_opt_centers[i], roll_noise[i]) for i in range(n_frames)]

    # Generate camera intrinsics
    zoom_noise = zoom_factor*(generate_sin_noise(1/n_frames, 1, 5, n_frames, rng)-0.5) + 1
    cam_intrinsics = [get_cam_intrinsic(F_x, F_y, o_x, o_y, zoom_noise[i]) for i in range(n_frames)]

    return [Camera(cam_opt_centers[i], cam_look_ats[i], cam_ups[i], cam_intrinsics[i]) for i in range(n_frames)]

# Moving Average for smoothing camera tracking
def moving_average_rows(arr, window_size, mode='edge'):
    kernel = np.ones(window_size) / window_size
    pad_width = window_size // 2
    arr_padded = np.pad(arr, ((pad_width, pad_width), (0, 0)), mode=mode)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=0, arr=arr_padded)

# Generate N linspaced samples of composed sin^2 signals of various frequencies
def generate_sin_noise(mu: float, sigma: float, n_signals: int, N: int, rng) -> npt.NDArray[np.float64]:        
    t = np.linspace(0, 1, N, endpoint=False)
    
    frequencies = rng.normal(loc=mu, scale=sigma, size=n_signals)
    pdf_values = stats.norm.pdf(frequencies, mu, sigma)
    amplitudes = pdf_values / np.max(pdf_values)
    
    amplitudes = 0.5 + 0.5 * amplitudes

    phases = rng.uniform(low=0, high=2 * np.pi, size=n_signals)
    
    signal = np.zeros(N)
    for f, a, p in zip(frequencies, amplitudes, phases):
        signal += a * np.sin(2 * np.pi * f * t + p)

    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    
    return signal

def get_path_endpoints(start_angle: float,
                       bound_center: npt.NDArray[np.float64],
                       bound_r: float,
                       dist: float) -> Tuple[npt.NDArray[np.float64]]:
    start = bound_center - np.array([bound_r + dist, 0])
    path_angle = math.atan(bound_r/dist)
    end = 2*dist*np.array([math.cos(path_angle), math.sin(path_angle)])
    
    theta = np.radians(start_angle)
    
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    start_rot = R @ (start - bound_center) + bound_center
    end_rot = R @ (end - bound_center) + bound_center
    return start_rot, end_rot

def get_bounding_sphere(poses: List[npt.NDArray[np.float64]], safe_dist: float) -> Tuple[npt.NDArray[np.float64], float]:
    roots = np.array([pose[0, :] for pose in poses])
    x_min = roots[:, 0].min()
    x_max = roots[:, 0].max()
    y_min = roots[:, 1].min()
    y_max = roots[:, 1].max()

    bound_center = np.array([x_min + x_max, y_min + y_max])/2
    bound_r = safe_dist + max(x_max - x_min, y_max - y_min)/2
    return bound_center, bound_r
    
def get_camera_up(cam_look_at: npt.NDArray[np.float64], cam_pos: npt.NDArray[np.float64], roll: float) -> npt.NDArray[np.float64]:
    cam_forward = cam_look_at - cam_pos
    cam_forward /= np.linalg.norm(cam_forward)

    world_up = np.array([0, 0, 1], dtype=np.float64)

    cam_right = np.cross(world_up, cam_forward)
    
    if np.linalg.norm(cam_right) < 1e-6:
        cam_right = np.array([1, 0, 0])

    cam_right /= np.linalg.norm(cam_right)

    # Ensure up is positive
    cam_up = np.cross(cam_right, cam_forward)
    if cam_up[2] < 0:
        cam_up = -cam_up

    return cam_up * np.cos(roll) + cam_right * np.sin(roll)

def get_cam_intrinsic(F_x: float, F_y: float, o_x: float, o_y: float, zoom=1.0, s=0):
    return np.array([
        [zoom*F_x, s,        o_x],
        [0,        zoom*F_y, o_y],
        [0,        0,          1]])
