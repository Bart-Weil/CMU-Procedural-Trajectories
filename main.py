import Procedural_Cameras.Generate_Dataset as gen
from Procedural_Cameras.Pose_Implementations import PoseImpl, H36mPose, CocoPose, SMPLPose
from Procedural_Cameras.Plotting import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    CMU_dir = '/vol/bitbucket/bw1222/data/CMU'
    write_dir = '/vol/bitbucket/bw1222/data/CMU_Camera_3DPW'
    # Dataset generation (supplying skeleton for 2d and 3d poses)
    gen.generate_dataset(CocoPose.CocoPose(), SMPLPose.SMPLPose(), CMU_dir, write_dir)
    
    # Scene loading
    scene = load_scene("/vol/bitbucket/bw1222/data/CMU_Camera_3DPW/subjects/24/24_1_2.pkl")

    plot_cam_trajectory(scene["pose_3d"], scene["cam_obj_sequence"])
    scene_len = scene["pose_3d"].shape[0]
    np.set_printoptions(suppress=True)
    print("eg pose: ", scene["pose_3d"][scene_len//2])
    # Plotting a frame from scene
    plot_human_and_cam_pose(scene["pose_3d"][0], SMPLPose.SMPLPose(), scene['cam_obj_sequence'][0])
    
    # Plot scene with projected floor
    # plot_cam_frames(scene['pose_3d'], scene['cam_obj_sequence'], H36mPose.H36mPose(), 'Plots/output', 60)

    # Render camera frames
    render_scene("/vol/bitbucket/bw1222/data/CMU_Camera_3DPW/subjects/24/24_1_2.pkl", CocoPose.CocoPose(), framerate=60, filename_prefix="Plots/output")
    
    plt.show()
