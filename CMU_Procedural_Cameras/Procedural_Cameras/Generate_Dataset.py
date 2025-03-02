import transforms3d
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

import torch

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=len(q.shape)-1)
    uuv = torch.cross(qvec, uv, dim=len(q.shape)-1)
    return (v + 2 * (q[..., :1] * uv + uuv))
    
    
def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return torch.cat((w, -xyz), dim=len(q.shape)-1)

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate

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
        cam_sequence = {
            'opt_center': np.array([cam.opt_center - initial_opt_center for cam in cam_seq]),
            'cam_look_at': np.array([cam.cam_look_at - initial_opt_center for cam in cam_seq]),
            'cam_up': np.array([cam.cam_up for cam in cam_seq]),
            'cam_intrinsic': np.array([cam.cam_intrinsic for cam in cam_seq])
        }

        mean_cam_v = 240*np.linalg.norm(np.diff(cam_sequence['opt_center'], axis=0), axis=1)

        pose_2d = [cam_seq[i].project_points(poses[i]) for i in range(len(cam_seq))]

        pose_3d = [poses[i] - initial_opt_center for i in range(len(cam_seq))]

        # Compute Pc = (Pw - t) * R.T
        pose_3d_cam_coords = []
        
        for i in range(len(cam_seq)):
            cam_ext = cam_seq[i].cam_ext
            R = cam_ext[:, :3]
            opt_center = cam_seq[i].opt_center
            pose_3d_cam_coords.append((poses[i]-opt_center)@R.T)

        scene = {
            'cam_sequence': cam_sequence,
            'pose_2d': np.array(pose_2d),
            'pose_3d': np.array(pose_3d),
            'pose_3d_cam_coords': np.array(pose_3d_cam_coords),
            'mean_cam_v': mean_cam_v
        }

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
