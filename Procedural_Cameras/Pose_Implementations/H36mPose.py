import numpy as np
import numpy.typing as npt
import transforms3d.euler as euler

from abc import ABC, abstractmethod
from typing import List, Dict

from Procedural_Cameras.AMCParser.amc_parser import *
from Procedural_Cameras.Pose_Implementations.PoseImpl import *

class H36mPose(CMU_Pose):
    # Depth first ordering of CMU joints
    joint_names = ['Pelvis', 'RHip', 'RKnee',
              'RAnkle', 'LHip', 'LKnee',
              'LAnkle', 'Spine1', 'Neck',
              'Head', 'Site', 'LShoulder',
              'LElbow', 'LWrist', 'RShoulder',
              'RElbow', 'RWrist']

    adjacency = {
        'Pelvis': ['RHip', 'LHip', 'Spine1'],
        'RHip': ['RKnee'],
        'RKnee': ['RAnkle'],
        'RAnkle': [],
        'LHip': ['LKnee'],
        'LKnee': ['LAnkle'],
        'LAnkle': [],
        'Spine1': ['Neck'],
        'Neck': ['LShoulder', 'RShoulder', 'Head'],
        'Head': ['Site'],
        'Site': [],
        'LShoulder': ['LElbow'],
        'LElbow': ['LWrist'],
        'LWrist': [],
        'RShoulder': ['RElbow'],
        'RElbow': ['RWrist'],
        'RWrist': []
    }

    def set_joints(self, cmu_joints):
        self.head_offset_factor = 0.65*self.m_conversion*cmu_joints['head'].length

        av_shoulder_length = (cmu_joints['lclavicle'].length + cmu_joints['rclavicle'].length)/2
        self.shoulder_offset_factor = 0.25*self.m_conversion*av_shoulder_length

        av_hip_length = (cmu_joints['lhipjoint'].length + cmu_joints['rhipjoint'].length)/2
        self.hip_offset_factor = 0.1*self.m_conversion*av_hip_length

    def set_pose(self, root_joint: Joint):
        self.cmu_pose = root_joint.to_dict()
        xs, ys, zs = [], [], []
        for joint in self.cmu_pose.values():
            xs.append(self.m_conversion * joint.coordinate[0, 0])
            ys.append(self.m_conversion * joint.coordinate[1, 0])
            zs.append(self.m_conversion * joint.coordinate[2, 0])
        # Quirky axis order used by .asf

        # Obtaining head pose (offset in look direction)
        head_matrix = self.cmu_pose['head'].matrix # Transform from default direction to current direction
        
        head_dir = np.squeeze(self.cmu_pose['head'].direction) # default direction
        
        look_dir = np.cross(head_dir, np.array([-1, 0, 0])) # default look direction (fwd perpendicular to head direction)
        unit_look_dir = look_dir/np.linalg.norm(look_dir)

        pose_look = (head_matrix@unit_look_dir)[[2, 0, 1]] # transform and reorder axes according to .asf convention

        cmu_joint_locs = np.array([zs, xs, ys]).T
        cmu_joints = dict(zip(super().joint_names, [cmu_joint_locs[i, :] for i in range(cmu_joint_locs.shape[0])]))

        # Obtaining shoulder offset
        l_shoulder = (self.cmu_pose['lclavicle'].matrix @ self.cmu_pose['lclavicle'].direction)[[2, 0, 1]]
        r_shoulder = (self.cmu_pose['rclavicle'].matrix @ self.cmu_pose['rclavicle'].direction)[[2, 0, 1]]

        l_shoulder_offset = np.squeeze(self.shoulder_offset_factor*l_shoulder)
        r_shoulder_offset = np.squeeze(self.shoulder_offset_factor*r_shoulder)

        # Obtaining hip offset
        l_hip = (self.cmu_pose['lclavicle'].matrix @ self.cmu_pose['lclavicle'].direction)[[2, 0, 1]]
        r_hip = (self.cmu_pose['rclavicle'].matrix @ self.cmu_pose['rclavicle'].direction)[[2, 0, 1]]

        l_hip_offset = np.squeeze(self.hip_offset_factor*l_shoulder)
        r_hip_offset = np.squeeze(self.hip_offset_factor*r_shoulder)

        self.joint_locs = {
            'Pelvis': (cmu_joints['lhipjoint'] + cmu_joints['rhipjoint'])/2,
            'RHip': cmu_joints['rhipjoint'] + r_hip_offset,
            'RKnee': cmu_joints['rfemur'] + r_hip_offset,
            'RAnkle': cmu_joints['rfoot'] + r_hip_offset,
            'LHip': cmu_joints['lhipjoint'] + l_hip_offset,
            'LKnee': cmu_joints['lfemur'] + l_hip_offset,
            'LAnkle': cmu_joints['lfoot'] + l_hip_offset,
            'Spine1': (cmu_joints['lowerback'] + cmu_joints['upperback'])/2,
            'Neck': cmu_joints['lowerneck'],
            'Head': cmu_joints['upperneck'] + self.head_offset_factor*pose_look,
            'Site': cmu_joints['head'],
            'LShoulder': cmu_joints['lclavicle'] - l_shoulder_offset,
            'LElbow': cmu_joints['lhumerus'] - l_shoulder_offset,
            'LWrist': cmu_joints['lwrist'] - l_shoulder_offset,
            'RShoulder': cmu_joints['rclavicle'] - r_shoulder_offset,
            'RElbow': cmu_joints['rhumerus'] - r_shoulder_offset,
            'RWrist': cmu_joints['rwrist'] - r_shoulder_offset
        }

    def get_joint_locs(self):
        return np.array(np.vstack(list(self.joint_locs.values())))

    def set_joint_locs(self, joint_locs: npt.NDArray[np.float64]):
        self.joint_locs = dict(zip(self.joint_names, [joint_locs[i, :] for i in range(joint_locs.shape[0])]))

    def plot_3D(self, fig, ax):
        xs, ys, zs = [], [], []
        for joint in self.joint_names:
            joint_coord = self.joint_locs[joint]
            xs.append(joint_coord[0])
            ys.append(joint_coord[1])
            zs.append(joint_coord[2])
            plt.plot(xs, ys, zs, 'b.')
            for child in self.adjacency[joint]:
                child_coord = self.joint_locs[child]
                xs = [child_coord[0], joint_coord[0]]
                ys = [child_coord[1], joint_coord[1]]
                zs = [child_coord[2], joint_coord[2]]
                plt.plot(xs, ys, zs, 'r')

    def plot_2D(self, fig, ax):
        xs, ys = [], []
        for joint_name in self.joint_names:
            joint = self.joint_locs[joint_name]
            is_occluded = joint[2]
            joint_coord = joint[:2]
            if not is_occluded:
                xs.append(joint_coord[0])
                ys.append(joint_coord[1])
                plt.plot(xs, ys, 'b.')
                for child in self.adjacency[joint_name]:
                    child_coord = self.joint_locs[child]
                    xs = [child_coord[0], joint_coord[0]]
                    ys = [child_coord[1], joint_coord[1]]
                    plt.plot(xs, ys, 'r')
