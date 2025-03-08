import Procedural_Cameras.Generate_Dataset as gen
from Procedural_Cameras.Pose_Implementations import PoseImpl, H36mPose, CocoPose
from Procedural_Cameras.Plotting import *

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # gen.generate_dataset(CocoPose.CocoPose(), H36mPose.H36mPose())
    scene = load_scene("../Datasets/CMU_Camera/subjects/01/01_1_1.pkl")
    # plot_human_and_cam_pose(scene["pose_3d_cam"][0], CocoPose.CocoPose(), scene['cam_obj_sequence'][0])
    
    # joints = parse_asf("../Datasets/CMU/subjects/01/01.asf")
    # pose_impl = CocoPose.CocoPose()
    # pose_impl.set_joints(joints)
    # motions = parse_amc("../Datasets/CMU/subjects/01/01_01.amc")
    # joints['root'].set_motion(motions[0])
    # pose_impl.set_pose(joints['root'])

    # plot_human_and_cam_pose(pose_impl.get_joint_locs(), pose_impl, scene['cam_obj_sequence'][0])

    # joints = parse_asf("../Datasets/CMU/subjects/01/01.asf")
    # pose_impl2 = H36mPose.H36mPose()
    # pose_impl2.set_joints(joints)
    # motions = parse_amc("../Datasets/CMU/subjects/01/01_01.amc")
    # joints['root'].set_motion(motions[0])
    # pose_impl2.set_pose(joints['root'])

    # plot_human_and_cam_pose(pose_impl2.get_joint_locs(), pose_impl2, scene['cam_obj_sequence'][0])

    render_scene("../Datasets/CMU_Camera/subjects/01/01_4_5.pkl", CocoPose.CocoPose(), framerate=15, filename_prefix="Plots/test_projection")

    plt.show()