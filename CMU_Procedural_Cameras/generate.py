import Procedural_Cameras.Generate_Dataset as gen
from Procedural_Cameras.Pose_Implementations import Pose_Impl, H36M_Pose
from Procedural_Cameras.Plotting import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    gen.generate_dataset(H36M_Pose.H36M_Pose())
    # render_scene("../Datasets/CMU_Camera/subjects/127/127_2_5.pkl", H36M_Pose.H36M_Pose(), framerate=15, filename_prefix="Plots/test_projection")
    scene = load_scene("../Datasets/CMU_Camera/subjects/01/01_1_1.pkl")
    plot_human_and_cam_pose(scene["pose_3d_cam_coords"][0], H36M_Pose.H36M_Pose(), scene['cam_obj_sequence'][0])
    plt.show()