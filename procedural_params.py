# Output FPS
cam_fps = 120

# -- Camera Intrinsics
# Focal x
F_x = 1000
# Focal y
F_y = 1000
# Screen x
o_x = 640
# Screen y
o_y = 360

"""
Procedural generation parameters
Each of the below fields serve as a lower and upper bound for the
uniform distribution of their parameter.
"""
# Time interval for motion
motion_interval = 3.0

# Approx radial distance from the camera to the root joint
r_min, r_max = 2.0, 3.0

# Approx camera height
h_min, h_max = 1.3, 1.7

# Cam velocity
v_min, v_max = 0.0, 0.15

# Cam acceleration
a_min, a_max = 0.0, 0.25

# Cam rotational velocity
omega_min, omega_max = 0.0, 0.04

# Cam rotational acceleration
alpha_min, alpha_max = 0.0, 0.1

"""
Parameters for generation of benchmark dataset
Benchmark dataset is far more constrained than procedural dataset.
For each scene we generate paths which contain pure rotation in a random
direction and pure movement in some direction, maintained for a set time.
The angular and linear velocities are normally distributed and constrained
to be positive.
"""

# Interval for motion
benchmark_motion_interval = 0.8

# Linear velocity
v_mu = 0.2
v_std = 0.4

# Angular velocity
omega_mu = 0.08
omega_std = 0.05

"""Error terms for the camera pose"""
# Use cumilative error for camera translation vs per-frame error
simulate_cam_error = False

cumilative_error = False

cam_position_error_std = 0.1
cam_rotation_error_std = 0.1

"""Error terms for the human pose"""
# Keypoint estimation error (px)
simulate_keypoint_error = False

human_keypoint_error_std = 1.0
