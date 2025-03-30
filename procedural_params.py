# Parameters to vary with sinusoidal noise
vert_noise = True    # Vertical distance
prog_noise = True    # Camera progress (velocity)
look_at_noise = True # Camera look vector
rad_noise = True     # Camera radial distance from subject
zoom_noise = False   # Camera zoom
roll_noise = True    # Camera roll

# -- Coarse Grained Procedural Parameters
# Span of camera path
path_length_mu = 4
path_length_std = 3

# Distance factor (how much we deviate from linear progress through camera path)
prog_factor_mu = 0.2
prog_factor_std = 0.1

# Vertical distance factor
vert_factor_mu = 0.35
vert_factor_std = 0.6

# Rotation factor
look_at_factor_mu = 0.15
look_at_factor_std = 0.05

# Radial movement factor
rad_factor_mu = 3.5
rad_factor_std = 3

# Zoom factor
zoom_factor_mu = 0.01
zoom_factor_std = 0.05

# Roll factor
roll_factor_mu = 0.01
roll_factor_std = 0.01

# Minimum safe distance from subject
safe_dist = 1.2

# Base radial distance from subject
base_rad_mu = 1.2
base_rad_std = 5

# Minimum safe height
safe_vert = 0.3

# Base vertical distance (height of camera)
base_vertical_mu = 1.4
base_vertical_std = 0.2

# -- Camera Intrinsics
# Focal x
F_x = 1000
# Focal y
F_y = 1000
# Screen x
o_x = 640
# Screen y
o_y = 360

# Smooths tracking of root joint
smoothing_window = 101

# -- Procedural Noise Frequency
# Control mean and stdev of signal frequency, and number of signals
# to compose. All means are normalised to number of frames in sequence

# Progress along linear path:
dist_mu = 1
dist_std = 20
dist_signals = 3

# Movement away from linear path:
radial_mu = 1
radial_std = 10
radial_signals = 5

# Vertical Movement
vertical_mu = 20
vertical_std = 50
vertical_signals = 20

# Movement of camera look vector
tracking_mu = 15
tracking_std = 50
tracking_signals = 5

# Camera Roll
roll_mu = 30
roll_std = 30
roll_signals = 10

# Camera Zoom
zoom_mu = 4
zoom_std = 2
zoom_signals = 5

# Average CMU sequence length - held as constant to save compution but easily found
avg_seq_length = 3289.875
