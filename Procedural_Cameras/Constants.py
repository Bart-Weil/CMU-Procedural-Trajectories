from procedural_params import *

import numpy as np

# FPS of the CMU dataset
CMU_FPS = 120

SEQ_LEN = int(np.ceil(cam_fps * motion_interval))
