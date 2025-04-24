import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, even if not used directly.
from tqdm import tqdm

import pickle
import os, sys

from Procedural_Cameras.AMCParser.amc_parser import *
from Procedural_Cameras.Camera import *
from Procedural_Cameras.Procedural_Cameras import *
from Procedural_Cameras.Pose_Implementations.PoseImpl import *

def read_mocap(joints_file, motions_file, pose_impl_2d: PoseImpl, pose_impl_3d: PoseImpl):
    joints = parse_asf(joints_file)
    pose_impl_2d.set_joints(joints)
    pose_impl_3d.set_joints(joints)

    motions = parse_amc(motions_file)

    proj_poses_3d = []
    poses_3d = []

    for i in range(len(motions)):
        joints['root'].set_motion(motions[i])
        pose_impl_2d.set_pose(joints['root'])
        pose_impl_3d.set_pose(joints['root'])

        proj_poses_3d.append(pose_impl_2d.get_joint_locs())
        poses_3d.append(pose_impl_3d.get_joint_locs())

    return np.array(proj_poses_3d), np.array(poses_3d)


def pickle_scene(cam_seq, proj_poses_3d, poses_3d, filename):
    cam_param_seq = {
        'cam_extrinsic': np.array([cam.cam_ext for cam in cam_seq]),
        'cam_intrinsic': np.array([cam.cam_intrinsic for cam in cam_seq])
    }

    scene_pose_2d = [cam_seq[i].project_points(proj_poses_3d[i]) for i in range(len(cam_seq))]

    scene_pose_3d = [poses_3d[i] for i in range(len(cam_seq))]

    scene_pose_3d_cam = []
    scene_pose_2d_cam = []
    for i in range(len(cam_seq)):
        cam_ext = cam_seq[i].cam_ext
        
        R = cam_ext[:, :3]
        opt_center = cam_seq[i].opt_center

        scene_pose_3d_cam.append((poses_3d[i] - opt_center)@R.T)

        scene_pose_2d_cam.append((proj_poses_3d[i] - opt_center)@R.T)

    scene = {
        'cam_sequence': cam_param_seq,
        # 'pose_2d': np.array(scene_pose_2d),
        # 'pose_2d_cam': np.array(scene_pose_2d_cam),
        # 'pose_3d': np.array(scene_pose_3d),
        'pose_3d_cam': np.array(scene_pose_3d_cam)
    }

    with open(filename, 'wb') as scene_pickle:
        pickle.dump(scene, scene_pickle)


def generate_benchmark_fom_mocap(filename_prefix,
                                 joints_file,
                                 motions_file,
                                 pose_impl_2d: PoseImpl,
                                 pose_impl_3d: PoseImpl,
                                 rng):
    n_paths = 10
    
    proj_poses_3d, poses_3d = read_mocap(joints_file, motions_file, pose_impl_2d, pose_impl_3d)

    cam_seqs = generate_benchmark_cam_seqs(poses_3d, n_paths, rng)

    for path in range(n_paths):
        cam_seq = cam_seqs[path]
        filename = filename_prefix + (f'_%0{len(str(n_paths))}d' % (path+1)) + '.pkl'
        pickle_scene(cam_seq, proj_poses_3d, poses_3d, filename)


def generate_from_mocap(filename_prefix,
                        joints_file,
                        motions_file,
                        pose_impl_2d: PoseImpl,
                        pose_impl_3d: PoseImpl,
                        rng):
    n_paths = 15

    proj_poses_3d, poses_3d = read_mocap(joints_file, motions_file, pose_impl_2d, pose_impl_3d)

    if (SEQ_LEN > proj_poses_3d.shape[0]):
        return

    cam_seqs, seq_starts, seq_ends = generate_cam_seqs(poses_3d, n_paths, rng)
    
    for path in range(n_paths):
        cam_seq = cam_seqs[path]
        seq_start, seq_end = seq_starts[path], seq_ends[path]

        frames = np.floor(np.arange(seq_start, seq_end, CMU_FPS/cam_fps)).astype('int32')
        
        proj_poses_3d_cam_frames = proj_poses_3d[frames, ...]
        poses_3d_cam_frames = poses_3d[frames, ...]

        filename = filename_prefix + (f'_%0{len(str(n_paths))}d' % (path+1)) + '.pkl'
        pickle_scene(cam_seq, proj_poses_3d_cam_frames, poses_3d_cam_frames, filename)

"""
Generate a dataset of 2D and 3D poses from CMU motion capture data.
pose_impl_2d: PoseImpl instance for 2D poses (model input)
pose_impl_3d: PoseImpl instance for 3D poses (model GT)
benchmark: If True, generates a simplified dataset for benchmarking purposes.
"""
def generate_dataset(pose_impl_2d: PoseImpl, pose_impl_3d: PoseImpl, benchmark: bool=False, seed=42):
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
            if not benchmark:

                generate_from_mocap(prefix, joints_file, motion_file, pose_impl_2d, pose_impl_3d, rng)
            else:
                generate_benchmark_fom_mocap(prefix + '_benchmark', joints_file, motion_file, pose_impl_2d, pose_impl_3d, rng)


def load_scene(scene_path):
    scene_file = open(scene_path, 'rb')
    scene_data = pickle.load(scene_file)
    scene_file.close()
    
    pose_2ds = scene_data['pose_2d']
    pose_3ds = scene_data['pose_3d']
    scene_pose_3d_cam = scene_data['pose_3d_cam']

    scene_cams = scene_data['cam_sequence']

    cams = [Camera(scene_cams['cam_extrinsic'][i],
                   scene_cams['cam_intrinsic'][i]) for i in range(len(scene_cams['cam_extrinsic']))]

    return {'cam_obj_sequence': cams, 'pose_2d': pose_2ds, 'pose_3d': pose_3ds, 'pose_3d_cam': scene_pose_3d_cam}
