import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.

from AMCParser.amc_parser import *
from Camera import *

joints = parse_asf("../Datasets/CMU/subjects/01/01.asf")
motions = parse_amc('../Datasets/CMU/subjects/01/01_01.amc')
joints['root'].set_motion(motions[0])

joints['root'].draw()
