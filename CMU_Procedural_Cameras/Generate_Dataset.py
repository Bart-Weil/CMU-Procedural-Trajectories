import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.

from AMCParser.amc_parser import *
from Camera import *
from Procedural_Cameras import *
from Plotting import *

from tqdm import tqdm

import pickle
import os, sys

def generate_data_for_sequence(filename_prefix, joints_file, motions_file):
    joints = parse_asf(joints_file)
    motions = parse_amc(motions_file)
    n_paths = 5

    poses = []
    for i in range(len(motions)):
        joints['root'].set_motion(motions[i])
        poses.append(joints['root'].get_pose().to_numpy())

    cam_seqs = generate_cam_seqs(poses, n_paths, np.random.randint(0, 100000))
    
    for path in range(n_paths):
        cam_seq = cam_seqs[path]
        initial_opt_center = cam_seq[0].opt_center
        scene_cam_seq = {"opt_center": np.array([cam.opt_center - initial_opt_center for cam in cam_seq]),
                         "cam_look_at": np.array([cam.cam_look_at for cam in cam_seq]),
                         "cam_up": np.array([cam.cam_up for cam in cam_seq]),
                         "cam_intrinsic": np.array([cam.cam_intrinsic for cam in cam_seq])
                        }

        scene_2d_pose = np.array([cam_seq[i].project_points(poses[i]) for i in range(len(cam_seq))])

        scene_3d_pose = np.array([poses[i] - cam_seq[i].opt_center for i in range(len(cam_seq))])

        scene = {"cam_sequence": scene_cam_seq, "pose_2d": scene_2d_pose, "pose_3d": scene_3d_pose}

        with open(filename_prefix + (f"_%0{len(str(n_paths))}d" % (path+1)) + ".pkl", "wb") as scene_pickle:
            pickle.dump(scene, scene_pickle)

def generate_dataset():
    CMU_Dir = "../Datasets/CMU"
    write_dataset_dir = "../Datasets/CMU_Camera"
    
    subjects_dir = os.path.join(CMU_Dir, "subjects")
    
    if not os.path.exists(subjects_dir):
        print(f"Error: Directory '{subjects_dir}' not found.")
        return
    
    for subject in tqdm(os.listdir(subjects_dir)):
        subject_path = os.path.join(subjects_dir, subject)
        
        if not os.path.isdir(subject_path):
            continue

        CMU_contents = os.listdir(subject_path)
        joints_files = [f for f in CMU_contents if f.endswith(".asf")]
        motion_files = [f for f in CMU_contents if f.endswith(".amc")]

        if not joints_files:
            print(f"Warning: No .asf file found for subject '{subject}', skipping.")
            continue

        if not motion_files:
            print(f"Warning: No .amc files found for subject '{subject}', skipping.")
            continue
        
        joints_file = os.path.join(subject_path, joints_files[0])

        for i, motion in enumerate(motion_files):
            motion_file = os.path.join(subject_path, motion)
            
            # Ensure output directory exists
            output_dir = os.path.join(write_dataset_dir, "subjects", subject)
            os.makedirs(output_dir, exist_ok=True)
            
            prefix = os.path.join(output_dir, subject + "_" + f"{i+1}")
            generate_data_for_sequence(prefix, joints_file, motion_file)

def render_scene(scene_path, framerate=60, filename_prefix=""):
    step = 120 // framerate

    scene_file = open(scene_path, "rb")
    scene_data = pickle.load(scene_file)
    scene_file.close()
    
    scene_2d_poses = scene_data["pose_2d"][::step]

    scene_cams = scene_data["cam_sequence"]

    cams = [Camera(scene_cams["opt_center"][i], 
                   scene_cams["cam_look_at"][i],
                   scene_cams["cam_up"][i],
                   scene_cams["cam_intrinsic"][i]) for i in range(0, len(scene_cams["opt_center"]), step)]

    n_frames = len(scene_2d_poses)
    for i in tqdm(range(n_frames)):
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.set_xlim(0, cams[i].screen_w)
        ax.set_ylim(0, cams[i].screen_h)

        plot_projected_pose(scene_2d_poses[i], fig, ax)

        filename = filename_prefix + (f"%0{len(str(n_frames))}d" % i) + ".png"
        plt.savefig(filename)
        plt.close()

if __name__ == "__main__":
    # generate_dataset()
    render_scene("../Datasets/CMU_Camera/subjects/01/01_2_1.pkl", framerate=15, filename_prefix="Plots/test_projection")

