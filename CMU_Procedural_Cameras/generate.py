import Procedural_Cameras.Generate_Dataset as gen
from Procedural_Cameras.Pose_Implementations import Pose_Impl, H36M_Pose
from Procedural_Cameras.Plotting import *

import matplotlib.pyplot as plt

def rotate_roll_3d(poses, angle_degrees):
    """
    Rotates all 3D joint positions around the X-axis (roll) by a given angle.

    Parameters:
        poses (np.ndarray): (frames, joints, 3) 3D array of poses.
        angle_degrees (float): The roll rotation angle in degrees.

    Returns:
        np.ndarray: The rotated pose array of shape (frames, joints, 3).
    """
    theta = np.radians(angle_degrees)  # Convert degrees to radians
    c, s = np.cos(theta), np.sin(theta)

    # Rotation matrix around X-axis (roll)
    R = np.array([[1,  0,  0],
                  [0,  c, -s],
                  [0,  s,  c]])

    # Apply rotation: (frames, joints, 3) @ (3,3) -> (frames, joints, 3)
    rotated_poses = np.einsum('ij,fkj->fki', R, poses)  # Efficient batch rotation

    return rotated_poses

if __name__ == "__main__":
    gen.generate_dataset(H36M_Pose.H36M_Pose())
    scene = load_scene("../Datasets/CMU_Camera/subjects/01/01_2_1.pkl")
    plot_human_and_cam_pose(scene["pose_3d_cam_coords"][0], H36M_Pose.H36M_Pose(), scene['cam_obj_sequence'][0])
    
    plot_human_and_cam_pose(rotate_roll_3d(scene["pose_3d_cam_coords"], -60)[0], H36M_Pose.H36M_Pose(), scene['cam_obj_sequence'][0])
    
    plot_human_and_cam_pose(scene["pose_3d"][0], H36M_Pose.H36M_Pose(), scene['cam_obj_sequence'][0])

    render_scene("../Datasets/CMU_Camera/subjects/01/01_2_1.pkl", H36M_Pose.H36M_Pose(), framerate=15, filename_prefix="Plots/test_projection")

    plt.show()