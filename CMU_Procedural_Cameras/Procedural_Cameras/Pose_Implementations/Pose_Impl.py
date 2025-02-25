import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from typing import List

from Procedural_Cameras.AMCParser.amc_parser import *

class Pose_Impl(ABC):
    m_conversion = 0.056444 
    @abstractmethod
    def set_joints(self, cmu_joints):
        pass

    @abstractmethod
    def set_pose(self, root_joint: Joint):
        pass

    @abstractmethod
    def get_joint_locs(self) -> npt.NDArray[np.float64]:
        pass

    @abstractmethod
    def set_joint_locs(self, joint_locs: npt.NDArray[np.float64]):
        pass

    @abstractmethod
    def plot_3D(self, fig, ax):
        pass

    @abstractmethod
    def plot_3D(self, fig, ax):
        pass

class CMU_Pose(Pose_Impl):
    # Depth first ordering of CMU joints
    joint_names = ['root', 'lhipjoint', 'lfemur',
              'ltibia', 'lfoot', 'ltoes',
              'rhipjoint', 'rfemur','rtibia',
              'rfoot', 'rtoes', 'lowerback',
              'upperback', 'thorax', 'lowerneck',
              'upperneck', 'head', 'lclavicle',
              'lhumerus', 'lradius', 'lwrist',
              'lhand', 'lfingers', 'lthumb',
              'rclavicle', 'rhumerus', 'rradius',
              'rwrist', 'rhand', 'rfingers',
              'rthumb']

    adjacency = {
    'root': ['lhipjoint', 'rhipjoint', 'lowerback'],
    'lhipjoint': ['lfemur'],
    'lfemur': ['ltibia'],
    'ltibia': ['lfoot'],
    'lfoot': ['ltoes'],
    'ltoes': [],
    'rhipjoint': ['rfemur'],
    'rfemur': ['rtibia'],
    'rtibia': ['rfoot'],
    'rfoot': ['rtoes'],
    'rtoes': [],
    'lowerback': ['upperback'],
    'upperback': ['thorax'],
    'thorax': ['lowerneck', 'lclavicle', 'rclavicle'],
    'lowerneck': ['upperneck'],
    'upperneck': ['head'],
    'head': [],
    'lclavicle': ['lhumerus'],
    'lhumerus': ['lradius'],
    'lradius': ['lwrist'],
    'lwrist': ['lhand'],
    'lhand': ['lfingers', 'lthumb'],
    'lfingers': [],
    'lthumb': [],
    'rclavicle': ['rhumerus'],
    'rhumerus': ['rradius'],
    'rradius': ['rwrist'],
    'rwrist': ['rhand'],
    'rhand': ['rfingers', 'rthumb'],
    'rfingers': [],
    'rthumb': []
    }

    def set_joints(self, cmu_joints):
        pass

    def set_pose(self, root_joint: Joint):
        self.cmu_pose = root_joint.to_dict()
        xs, ys, zs = [], [], []
        for joint in self.cmu_pose.values():
            xs.append(self.m_conversion * joint.coordinate[0, 0])
            ys.append(self.m_conversion * joint.coordinate[1, 0])
            zs.append(self.m_conversion * joint.coordinate[2, 0])
        # Quirky axis order used by .asf
        self.set_joint_locs(np.array([zs, xs, ys]).T)

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
