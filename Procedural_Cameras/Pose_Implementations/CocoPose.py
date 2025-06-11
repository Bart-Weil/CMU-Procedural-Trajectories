import numpy as np
import numpy.typing as npt
import transforms3d.euler as euler

from abc import ABC, abstractmethod
from typing import List, Dict

from Procedural_Cameras.AMCParser.amc_parser import *
from Procedural_Cameras.Pose_Implementations.PoseImpl import *

class CocoPose(CMU_Pose):
    joint_names = [
        'NOSE', 'NECK', 'R_SHOULDER',
        'R_ELBOW', 'R_WRIST', 'L_SHOULDER',
        'L_ELBOW', 'L_WRIST', 'R_HIP',
        'R_KNEE', 'R_ANKLE', 'L_HIP',
        'L_KNEE', 'L_ANKLE', 'R_EYE',
        'L_EYE', 'R_EAR', 'L_EAR',
    ]

    adjacency = {
        'NOSE': ['L_EYE', 'R_EYE', 'L_EAR', 'R_EAR', 'NECK'],
        'NECK': ['L_SHOULDER', 'R_SHOULDER', 'L_HIP', 'R_HIP'],
        'R_SHOULDER': ['R_ELBOW'],
        'R_ELBOW': ['R_WRIST'],
        'R_WRIST': [],
        'L_SHOULDER': ['L_ELBOW'],
        'L_ELBOW': ['L_WRIST'],
        'L_WRIST': [],
        'R_HIP': ['R_KNEE'],
        'R_KNEE': ['R_ANKLE'],
        'R_ANKLE': [],
        'L_HIP': ['L_KNEE'],
        'L_KNEE': ['L_ANKLE'],
        'L_ANKLE': [],
        'R_EYE': [],
        'L_EYE': [],
        'R_EAR': [], 
        'L_EAR': [],
    }

    def set_joints(self, cmu_joints):
        self.nose_fwd_factor = 0.5*self.m_conversion*cmu_joints['head'].length
        
        self.ear_widtvert_factor = 0.6*self.m_conversion*cmu_joints['head'].length
        
        self.eye_fwd_factor = 0.3*self.m_conversion*cmu_joints['head'].length
        self.eye_widtvert_factor = 0.3*self.m_conversion*cmu_joints['head'].length
        self.eye_up_factor = 0.4*self.m_conversion*cmu_joints['head'].length

        self.head_up_factor = 0.2*self.m_conversion*cmu_joints['head'].length

        av_shoulder_length = (cmu_joints['lclavicle'].length + cmu_joints['rclavicle'].length)/2
        self.shoulder_offset_factor = 0.25*self.m_conversion*av_shoulder_length

        av_hip_length = (cmu_joints['lhipjoint'].length + cmu_joints['rhipjoint'].length)/2
        self.hip_offset_factor = 0.15*self.m_conversion*av_hip_length

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
        
        head_up = np.squeeze(self.cmu_pose['head'].direction) # default direction
        head_up = head_up/np.linalg.norm(head_up)

        head_fwd = np.cross(head_up, np.array([-1, 0, 0]))

        head_right = np.cross(head_up, head_fwd)

        # rotate each default dir and reorder axes according to .asf convention (y, z, x) -> (x. y, z), z vertical
        nose_offset = (head_matrix@(self.nose_fwd_factor*head_fwd))[[2, 0, 1]]
        
        # Ear offsets
        l_ear_offset = self.ear_widtvert_factor*(head_matrix@(-head_right))[[2, 0, 1]]
        r_ear_offset = self.ear_widtvert_factor*(head_matrix@head_right)[[2, 0, 1]]

        # Eye offsets
        l_eye_dir = self.eye_fwd_factor*head_fwd - self.eye_widtvert_factor*head_right + self.eye_up_factor*head_up
        r_eye_dir = self.eye_fwd_factor*head_fwd + self.eye_widtvert_factor*head_right + self.eye_up_factor*head_up
        l_eye_offset = (head_matrix@l_eye_dir)[[2, 0, 1]]
        r_eye_offset = (head_matrix@r_eye_dir)[[2, 0, 1]]

        cmu_joint_locs = np.array([zs, xs, ys]).T
        cmu_joints = dict(zip(super().joint_names, [cmu_joint_locs[i, :] for i in range(cmu_joint_locs.shape[0])]))

        # Obtaining shoulder offset
        l_shoulder = (self.cmu_pose['lclavicle'].matrix @ self.cmu_pose['lclavicle'].direction)[[2, 0, 1]]
        r_shoulder = (self.cmu_pose['rclavicle'].matrix @ self.cmu_pose['rclavicle'].direction)[[2, 0, 1]]

        l_shoulder_offset = np.squeeze(self.shoulder_offset_factor*l_shoulder)
        r_shoulder_offset = np.squeeze(self.shoulder_offset_factor*r_shoulder)

        head_offset = self.head_up_factor * (head_matrix@head_up)[[2, 0, 1]]
        coco_head = cmu_joints['upperneck'] + head_offset

        self.joint_locs = {
            'NOSE': coco_head + nose_offset,
            'NECK': cmu_joints['lowerneck'],
            'R_SHOULDER': cmu_joints['rclavicle'] - r_shoulder_offset,
            'R_ELBOW': cmu_joints['rhumerus'] - r_shoulder_offset,
            'R_WRIST': cmu_joints['rwrist'] - r_shoulder_offset,
            'L_SHOULDER': cmu_joints['lclavicle'] - l_shoulder_offset,
            'L_ELBOW': cmu_joints['lhumerus'] - l_shoulder_offset,
            'L_WRIST': cmu_joints['lwrist'] - l_shoulder_offset,
            'R_HIP': cmu_joints['rhipjoint'],
            'R_KNEE': cmu_joints['rfemur'],
            'R_ANKLE': cmu_joints['rfoot'],
            'L_HIP': cmu_joints['lhipjoint'],
            'L_KNEE': cmu_joints['lfemur'],
            'L_ANKLE': cmu_joints['lfoot'],
            'R_EYE': coco_head + r_eye_offset,
            'L_EYE': coco_head + l_eye_offset,
            'R_EAR': coco_head + r_ear_offset,
            'L_EAR': coco_head + l_ear_offset,
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
