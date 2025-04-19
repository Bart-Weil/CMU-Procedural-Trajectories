import Procedural_Cameras.Generate_Dataset as gen
from Procedural_Cameras.Pose_Implementations import PoseImpl, H36mPose, CocoPose
from Procedural_Cameras.Plotting import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Dataset generation (supplying skeleton for 2d and 3d poses)
    gen.generate_dataset(pose_impl_2d=CocoPose.CocoPose(), pose_impl_3d=H36mPose.H36mPose(), benchmark=True)
    
    # # Scene loading
    scene = load_scene("../Datasets/CMU_Camera/subjects/01/01_1_1.pkl")
    
    # # Plotting a frame from scene
    plot_human_and_cam_pose(scene["pose_3d_cam"][0], CocoPose.CocoPose(), scene['cam_obj_sequence'][0])
    
    # # Plot scene with projected floor
    plot_cam_frames(scene['pose_3d'], scene['cam_obj_sequence'], H36mPose.H36mPose(), 'Plots/output', 15)

    # Render camera frames
    render_scene("../Datasets/CMU_Camera/subjects/01/01_1_benchmark_01.pkl", CocoPose.CocoPose(), framerate=60, filename_prefix="Plots/benchmark")
    
    plt.show()