import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.
from tqdm import tqdm

import pickle
import os, sys

from Procedural_Cameras.AMCParser.amc_parser import *
from Procedural_Cameras.Camera import *
from Procedural_Cameras.Procedural_Cameras import *
from Procedural_Cameras.Pose_Implementations.Pose_Impl import *

def generate_data_for_sequence(filename_prefix, joints_file, motions_file, pose_impl: Pose_Impl, rng):
    joints = parse_asf(joints_file)
    pose_impl.set_joints(joints)

    motions = parse_amc(motions_file)
    n_paths = 5

    poses = []
    for i in range(len(motions)):
        joints['root'].set_motion(motions[i])
        pose_impl.set_pose(joints['root'])
        poses.append(pose_impl.get_joint_locs())

    cam_seqs = generate_cam_seqs(poses, n_paths, rng)
    
    for path in range(n_paths):
        cam_seq = cam_seqs[path]
        initial_opt_center = cam_seq[0].opt_center
        cam_sequence = {'opt_center': np.array([cam.opt_center - initial_opt_center for cam in cam_seq]),
                         'cam_look_at': np.array([cam.cam_look_at - initial_opt_center for cam in cam_seq]),
                         'cam_up': np.array([cam.cam_up for cam in cam_seq]),
                         'cam_intrinsic': np.array([cam.cam_intrinsic for cam in cam_seq])
                        }

        pose_2d = np.array([cam_seq[i].project_points(poses[i]) for i in range(len(cam_seq))])

        pose_3d = [poses[i] - initial_opt_center for i in range(len(cam_seq))]

        hom_3d_pose = [np.hstack((pose_3d[i], np.ones((pose_3d[i].shape[0], 1)))) for i in range(len(pose_3d))]

        pose_3d = np.array(pose_3d)

        pose_3d_cam_coords = np.array([hom_3d_pose[i] @ cam_seq[i].cam_ext.T for i in range(len(cam_seq))])

        scene = {'cam_sequence': cam_sequence, 'pose_2d': pose_2d, 'pose_3d': pose_3d, 'pose_3d_cam_coords': pose_3d_cam_coords}

        with open(filename_prefix + (f'_%0{len(str(n_paths))}d' % (path+1)) + '.pkl', 'wb') as scene_pickle:
            pickle.dump(scene, scene_pickle)

def generate_dataset(pose_impl: Pose_Impl, seed=42):
    CMU_Dir = '../Datasets/CMU'
    write_dataset_dir = '../Datasets/CMU_Camera'
    
    subjects_dir = os.path.join(CMU_Dir, 'subjects')
    
    rng = np.random.default_rng(seed)

    if not os.path.exists(subjects_dir):
        print(f"Error: Directory '{subjects_dir}' not found.")
        return
    
    for subject in tqdm(os.listdir(subjects_dir)):
        subject_path = os.path.join(subjects_dir, subject)
        
        if not os.path.isdir(subject_path):
            continue

        CMU_contents = os.listdir(subject_path)
        joints_files = [f for f in CMU_contents if f.endswith('.asf')]
        motion_files = [f for f in CMU_contents if f.endswith('.amc')]

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
            output_dir = os.path.join(write_dataset_dir, 'subjects', subject)
            os.makedirs(output_dir, exist_ok=True)
            
            prefix = os.path.join(output_dir, subject + '_' + f'{i+1}')
            generate_data_for_sequence(prefix, joints_file, motion_file, pose_impl, rng)

def load_scene(scene_path):
    scene_file = open(scene_path, 'rb')
    scene_data = pickle.load(scene_file)
    scene_file.close()
    
    pose_2ds = scene_data['pose_2d']
    pose_3ds = scene_data['pose_3d']
    pose_3d_cam_coords = scene_data['pose_3d_cam_coords']


    scene_cams = scene_data['cam_sequence']

    cams = [Camera(scene_cams['opt_center'][i], 
                   scene_cams['cam_look_at'][i],
                   scene_cams['cam_up'][i],
                   scene_cams['cam_intrinsic'][i]) for i in range(len(scene_cams['opt_center']))]

    return {'cam_obj_sequence': cams, 'pose_2d': pose_2ds, 'pose_3d': pose_3ds, 'pose_3d_cam_coords': pose_3d_cam_coords}
