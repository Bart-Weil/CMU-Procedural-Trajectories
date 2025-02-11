# -- Coarse Grained Procedural Parameters

# Probability of generating a faster camera motion over a longer distance
p_fast_shot = 0.15

# Span of camera path
path_length_mu = 3
path_length_std = 2

# Fast shot path length increase
fast_dist_mu = 4
fast_dist_std = 2

# Probability of generating an overhead camera motion
p_overhead_shot = 0.2

# Overhead path height increase
overhead_h_mu = 0.8
overhead_h_std = 4

# Minimum height along path
min_h_mu = 1.3
min_h_sigma = 0.3

# Minimum safe height
safe_h = 0.75

# Height range
h_range_mu = 0.75
h_range_std = 0.6

# Rotation factor
rot_factor_mu = 0.6
rot_factor_std = 0.2

# Lateral movement factor
lat_factor_mu = 3.5
lat_factor_std = 3

# Minimum safe distance from subject
safe_dist = 1.2

# Minimum distance from subject
min_dist_mu = 5
min_dist_std = 5

# -- Camera Intrinsics

# Focal x
F_x = 1000
# Focal y
F_y = 1000
# Screen x
o_x = 640
# Screen y
o_y = 360

# -- Procedural Noise Amplitude
# Camera roll and zoom
roll_factor = 0.005
zoom_factor = 0.05
# Extent to which we deviate from linear progress through camera path
disp_factor = 0.7
# Smooths tracking of root joint
smoothing_window = 101

# -- Procedural Noise Frequency:
# Control mean and stdev of signal frequency, and number of signals
# to compose. All means are normalised to number of frames in sequence

# Progress along linear path:
disp_mu = 1
disp_std = 2
disp_signals = 3

# Movement away from linear path:
lateral_mu = 1
lateral_std = 1
lateral_signals = 5

# Vertical Movement
vertical_mu = 20
vertical_std = 5
vertical_signals = 20

# Movement of camera look vector
tracking_mu = 15
tracking_std = 5
tracking_signals = 5

# Camera Roll
roll_mu = 30
roll_std = 5
roll_signals = 10

# Camera Zoom
zoom_mu = 4
zoom_std = 2
zoom_signals = 5
