import numpy as np
import numpy.typing as npt

import math
from typing import List, Tuple

from scipy import stats

from Procedural_Cameras.Camera import *
from procedural_params import *
from Procedural_Cameras.Constants import *

from dataclasses import dataclass

@dataclass
class CamSeqConfig:
    start_angle: float
    path_length: float
    base_rad_dist: float
    base_vert: float
    prog_factor: float
    vert_factor: float
    look_at_factor: float
    rad_factor: float
    roll_factor: float
    zoom_factor: float

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

# Generage benchmark camera trajectories
def generate_benchmark_cam_seqs(poses: List[npt.NDArray[np.float64]],
                                n_seqs: int,
                                rng) -> List[List[Camera]]:
    
    cam_seqs = []
    
    motion_frames = np.floor(motion_interval*CMU_FPS).astype(int)

    for seq in range(n_seqs):
        start_frame = rng.integers(low=0, high=(len(poses) - motion_frames))
        end_frame = start_frame + motion_frames

        theta = rng.uniform(low=0, high=2*np.pi)
        bound_center, bound_r = get_bounding_circle(poses[start_frame:end_frame], 3.0)
        tangent_point = bound_center + np.array([bound_r*math.cos(theta), bound_r*math.sin(theta)])
        midpoint = np.hstack([tangent_point, np.array([1.5])]) # Approx avg camera height from H3.6M dataset
        v = rng.normal(loc=0, scale=1, size=(3,))
        v = max(0, rng.normal(loc=v_mu, scale=v_std))/np.linalg.norm(v)
        start = midpoint - v*motion_interval
        end = midpoint + v*motion_interval

        cam_pos = np.linspace(start, end, num=motion_frames)

        omega = max(0, rng.normal(loc=omega_mu, scale=omega_std))
        
        look_at = poses[(start_frame + end_frame)//2][0, :] # Look at root joint

        thetas = np.linspace(-omega*motion_interval/2, omega*motion_interval/2, num=motion_frames)
        Rs = np.array([rotation_about_normal(np.array([0, 0, 1]), theta) for theta in thetas])
        
        cam_look_ats = np.array([((look_at - cam_pos[i]) @ Rs[i]) + cam_pos[i]
                                 for i in range(motion_frames)])
        cam_ups = [get_camera_up(cam_look_ats[i], cam_pos[i], 0) for i in range(motion_frames)]

        cam_seqs.append([Camera(cam_pos[i], cam_look_ats[i], cam_ups[i], 
                                get_cam_intrinsic(F_x, F_y, o_x, o_y)) for i in range(motion_frames)])

    return cam_seqs


# Generate camera trajectories
def generate_cam_seqs(poses: List[npt.NDArray[np.float64]],
                      n_seqs: int,
                      rng) -> List[List[Camera]]:
    start_increment = 360 // n_seqs # Start point rotation increment (deg)

    start_angle = rng.uniform(low=0, high=360)

    cam_seqs = []
    n_frames = len(poses)

    for i in range(n_seqs):
        path_length = max(rng.normal(loc=path_length_mu, scale=path_length_std), 0)
        path_length = path_length * n_frames/avg_seq_length # Account for varying sequence length
        
        # Base radial distance from subject
        base_rad_dist = max(rng.normal(loc=base_rad_mu, scale=base_rad_std), safe_dist)
        
        # Base vertical distance (height of camera)
        base_vert = max(rng.normal(loc=base_vertical_mu, scale=base_vertical_std), safe_vert)

        vert_factor = max(rng.normal(loc=vert_factor_mu, scale=vert_factor_std), 0) if vert_noise else 0

        prog_factor = max(rng.normal(loc=prog_factor_mu, scale=prog_factor_std), 0) if prog_noise else 0

        look_at_factor = max(rng.normal(loc=look_at_factor_mu, scale=look_at_factor_std), 0) if look_at_noise else 0

        rad_factor = max(rng.normal(loc=rad_factor_mu, scale=rad_factor_std), 0) if rad_noise else 0

        zoom_factor = max(rng.normal(loc=zoom_factor_mu, scale=zoom_factor_std), 0) if zoom_noise else 0

        roll_factor = max(rng.normal(loc=roll_factor_mu, scale=roll_factor_std), 0) if roll_noise else 0

        seq_config = CamSeqConfig(
            start_angle=start_angle,
            path_length=path_length,
            base_rad_dist=base_rad_dist,
            base_vert=base_vert,
            prog_factor=prog_factor,
            vert_factor=vert_factor,
            look_at_factor=look_at_factor,
            rad_factor=rad_factor,
            roll_factor=roll_factor,
            zoom_factor=zoom_factor
        )

        cam_seqs.append(generate_cam_seq(poses, seq_config, rng))

        start_angle += start_increment

    return cam_seqs

# Procedurally Generate Camera Trajectories from a given starting point
def generate_cam_seq(poses: List[npt.NDArray[np.float64]], seq_config: CamSeqConfig, rng) -> List[Camera]:

    assert(smoothing_window % 2 == 1)

    n_frames = len(poses)

    bound_center, bound_r = get_bounding_circle(poses, seq_config.base_rad_dist)

    start, end = get_path_endpoints(seq_config.start_angle, bound_center, bound_r, seq_config.path_length)

    if rng.uniform() < 0.5:
        start, end = end, start

    # Generate camera positions
    progress_noise = generate_sin_noise(dist_mu,
                                        dist_std, dist_signals, n_frames, rng)
    path_progress = seq_config.prog_factor * progress_noise + np.linspace(0, 1, n_frames)
    path_points = start + (end - start)*path_progress[:, np.newaxis]

    radial_dist = generate_sin_noise(radial_mu,
                                     radial_std, radial_signals, n_frames, rng)
    radial_dist *= seq_config.rad_factor

    path_radial = (start+end)/2 - bound_center
    path_radial = path_radial/np.linalg.norm(path_radial)

    path_points_xy = path_points + (radial_dist[:, np.newaxis] * path_radial)
    
    vertical_noise = generate_sin_noise(vertical_mu,
                                        vertical_std, vertical_signals, n_frames, rng)
    vertical_noise = seq_config.vert_factor * (vertical_noise - 1/2)
    vertical_dist = seq_config.base_vert + vertical_noise

    path_points = np.hstack((path_points_xy, vertical_dist[:, np.newaxis]))

    # Generate camera orientations
    root_positions = np.array([pose[0, :] for pose in poses])
    tracking_positions = moving_average_rows(root_positions, smoothing_window)

    roll_noise = generate_sin_noise(roll_mu, roll_std, roll_signals, n_frames, rng)
    roll = seq_config.roll_factor*2*np.pi*(roll_noise - 1/2)

    tracking_noise = seq_config.look_at_factor * np.vstack([
    generate_sin_noise(tracking_mu / avg_seq_length,
                       tracking_std / avg_seq_length,
                       tracking_signals, n_frames, rng)
    for _ in range(3)
    ]).T
    
    noisy_tracking_positions = tracking_positions + tracking_noise

    cam_opt_centers = list(path_points)
    cam_look_ats = list(noisy_tracking_positions)
    cam_ups = [get_camera_up(cam_look_ats[i], cam_opt_centers[i], roll[i]) for i in range(n_frames)]

    # Generate camera intrinsics
    zoom_noise = generate_sin_noise(zoom_mu, zoom_std, zoom_signals, n_frames, rng)
    zoom = seq_config.zoom_factor*(zoom_noise-1/2) + 1
    cam_intrinsics = [get_cam_intrinsic(F_x, F_y, o_x, o_y, zoom[i]) for i in range(n_frames)]

    return [Camera(cam_opt_centers[i], cam_look_ats[i], cam_ups[i], cam_intrinsics[i]) for i in range(n_frames)]

# Moving Average for smoothing camera tracking
def moving_average_rows(arr, window_size, mode='edge'):
    kernel = np.ones(window_size) / window_size
    pad_width = window_size // 2
    arr_padded = np.pad(arr, ((pad_width, pad_width), (0, 0)), mode=mode)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode='valid'), axis=0, arr=arr_padded)

# Generate N linspaced samples of composed and shifted sin^2 signals of various frequencies, oscirading in [0, 1]
def generate_sin_noise(mu: float, sigma: float, n_signals: int, N: int, rng) -> npt.NDArray[np.float64]:        
    t = np.linspace(0, 1, N, endpoint=False)
    
    frequencies = rng.normal(loc=mu/avg_seq_length, scale=sigma/avg_seq_length, size=n_signals)
    pdf_values = stats.norm.pdf(frequencies, mu, sigma)
    amplitudes = pdf_values / np.max(pdf_values)
    
    amplitudes = 1/2 + amplitudes/2

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
    dist_center = np.sqrt((dist/2)**2 + bound_r**2)
    start = bound_center - np.array([dist_center, 0])
    path_angle = math.atan(bound_r/(dist/2 + 0.0001))
    end = dist*np.array([math.cos(path_angle), math.sin(path_angle)]) + start
    
    theta = np.radians(start_angle)
    
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    start_rot = R @ (start - bound_center) + bound_center
    end_rot = R @ (end - bound_center) + bound_center
    return start_rot, end_rot

def get_bounding_circle(poses: List[npt.NDArray[np.float64]], min_dist: float) -> Tuple[npt.NDArray[np.float64], float]:
    roots = np.array([pose[0, :] for pose in poses])

    x_coords = roots[:, 0]
    y_coords = roots[:, 1]

    bound_center = np.array([x_coords.mean(), y_coords.mean()])
    bound_r = min_dist + max(np.ptp(x_coords), np.ptp(y_coords)) / 2

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
