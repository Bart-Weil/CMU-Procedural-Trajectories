import numpy as np
import numpy.typing as npt
import transforms3d.euler as euler

from abc import ABC, abstractmethod
from typing import List, Dict

from Procedural_Cameras.AMCParser.amc_parser import *
from Procedural_Cameras.Pose_Implementations.PoseImpl import *

class SMPLPose(CMU_Pose):
    joint_names = ['Pelvis', 'LHip', 'RHip',
              'Thorax', 'LKnee', 'RKnee',
              'Sternum', 'LAnkle', 'RAnkle',
              'Chest', 'LFoot', 'RFoot',
              'Neck', 'LChest', 'RChest',
              'Head', 'LShoulder', 'RShoulder',
              'LElbow', 'RElbow', 'LWrist',
              'RWrist', 'LHand', 'RHand']

    adjacency = {
        'Pelvis': ['LHip', 'RHip', 'Thorax'],
        'LHip': ['LKnee'],
        'RHip': ['RKnee'],
        'Thorax': ['Sternum'],
        'LKnee': ['LAnkle'],
        'RKnee': ['RAnkle'],
        'Sternum': ['Chest'],
        'LAnkle': ['LFoot'],
        'RAnkle': ['RFoot'],
        'Chest': ['Neck', 'LChest', 'RChest'],
        'LFoot': [],
        'RFoot': [],
        'Neck': ['Head'],
        'LChest': ['LShoulder'],
        'RChest': ['RShoulder'],
        'Head': [],
        'LShoulder': ['LElbow'],
        'RShoulder': ['RElbow'],
        'LElbow': ['LWrist'],
        'RElbow': ['RWrist'],
        'LWrist': ['LHand'],
        'RWrist': ['RHand'],
        'LHand': [],
        'RHand': []
    }

    def set_joints(self, cmu_joints):
        self.chest_offset_factor = 0.7*self.m_conversion*cmu_joints['lowerback'].length
        self.neck_offset_factor = 0.5*self.m_conversion*cmu_joints['head'].length
        self.head_offset_factor = 1.7*self.m_conversion*cmu_joints['head'].length


    def set_pose(self, root_joint: Joint):
        self.cmu_pose = root_joint.to_dict()
        xs, ys, zs = [], [], []
        for joint in self.cmu_pose.values():
            xs.append(self.m_conversion * joint.coordinate[0, 0])
            ys.append(self.m_conversion * joint.coordinate[1, 0])
            zs.append(self.m_conversion * joint.coordinate[2, 0])

        # Obtaining head pose (offset in look direction)
        chest_matrix = self.cmu_pose['thorax'].matrix # Transform from default direction to current direction
        chest_dir_default = np.squeeze(self.cmu_pose['thorax'].direction) # default direction

        chest_dir = chest_matrix@chest_dir_default
        chest_dir = (chest_dir/np.linalg.norm(chest_dir))[[2, 0, 1]]

        neck_matrix = self.cmu_pose['upperneck'].matrix # Transform from default direction to current direction
        neck_dir_default = np.squeeze(self.cmu_pose['upperneck'].direction) # default direction

        neck_dir = neck_matrix@neck_dir_default
        neck_dir = (neck_dir/np.linalg.norm(neck_dir))[[2, 0, 1]]

        cmu_joint_locs = np.array([zs, xs, ys]).T
        cmu_joints = dict(zip(super().joint_names, [cmu_joint_locs[i, :] for i in range(cmu_joint_locs.shape[0])]))

        # Quirky axis order used by .asf
        self.joint_locs = {
            'Pelvis': cmu_joints['root'],
            'LHip': cmu_joints['lhipjoint'],
            'RHip': cmu_joints['rhipjoint'],
            'Thorax': cmu_joints['lowerback'],
            'LKnee': cmu_joints['lfemur'],
            'RKnee': cmu_joints['rfemur'],
            'Sternum': cmu_joints['upperback'],
            'LAnkle': cmu_joints['lfoot'],
            'RAnkle': cmu_joints['rfoot'],
            'Chest': cmu_joints['thorax'] - self.chest_offset_factor*chest_dir,
            'LFoot': cmu_joints['ltoes'],
            'RFoot': cmu_joints['rtoes'],
            'Neck': cmu_joints['lowerneck'] - self.neck_offset_factor*neck_dir,
            'LChest': (cmu_joints['thorax'] + cmu_joints['lclavicle'])/2,
            'RChest': (cmu_joints['thorax'] + cmu_joints['rclavicle'])/2,
            'Head': cmu_joints['upperneck'] - self.head_offset_factor*neck_dir,
            'LShoulder': cmu_joints['lclavicle'],
            'RShoulder': cmu_joints['rclavicle'],
            'LElbow': cmu_joints['lhumerus'],
            'RElbow': cmu_joints['rhumerus'],
            'LWrist': cmu_joints['lwrist'],
            'RWrist': cmu_joints['rwrist'],
            'LHand': cmu_joints['lhand'],
            'RHand': cmu_joints['rhand']
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
