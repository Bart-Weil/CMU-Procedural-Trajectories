import numpy as np

import numpy.typing as npt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from transforms3d.euler import euler2mat

from typing import List

class CMU_Pose:
  # Depth first ordering of CMU joints
  joints = ['root', 'lhipjoint', 'lfemur',
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

  def __init__(self, joint_locs: npt.NDArray[np.float64]):
    self.joint_locs = dict(zip(self.joints, [joint_locs[i, :] for i in range(joint_locs.shape[0])]))

  def to_numpy(self):
    return np.array(np.vstack(list(self.joint_locs.values())))

  def plot_3D(self, fig, ax):
    xs, ys, zs = [], [], []
    for joint in self.joints:
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
    for joint_name in self.joints:
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

class Joint:
  def __init__(self, name, direction, length, axis, dof, limits):
    """
    Definition of basic joint. The joint also contains the information of the
    bone between it's parent joint and itself. Refer
    [here](https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html)
    for detailed description for asf files.

    Parameter
    ---------
    name: Name of the joint defined in the asf file. There should always be one
    root joint. String.

    direction: Default direction of the joint(bone). The motions are all defined
    based on this default pose.

    length: Length of the bone.

    axis: Axis of rotation for the bone.

    dof: Degree of freedom. Specifies the number of motion channels and in what
    order they appear in the AMC file.

    limits: Limits on each of the channels in the dof specification

    """
    self.name = name
    self.direction = np.reshape(direction, [3, 1])
    self.length = length
    axis = np.deg2rad(axis)
    self.C = euler2mat(*axis)
    self.Cinv = np.linalg.inv(self.C)
    self.limits = np.zeros([3, 2])
    for lm, nm in zip(limits, dof):
      if nm == 'rx':
        self.limits[0] = lm
      elif nm == 'ry':
        self.limits[1] = lm
      else:
        self.limits[2] = lm
    self.parent = None
    self.children = []
    self.coordinate = None
    self.matrix = None

  def set_motion(self, motion):
    if self.name == 'root':
      self.coordinate = np.reshape(np.array(motion['root'][:3]), [3, 1])
      rotation = np.deg2rad(motion['root'][3:])
      self.matrix = self.C.dot(euler2mat(*rotation)).dot(self.Cinv)
    else:
      idx = 0
      rotation = np.zeros(3)
      for axis, lm in enumerate(self.limits):
        if not np.array_equal(lm, np.zeros(2)):
          rotation[axis] = motion[self.name][idx]
          idx += 1
      rotation = np.deg2rad(rotation)
      self.matrix = self.parent.matrix.dot(self.C).dot(euler2mat(*rotation)).dot(self.Cinv)
      self.coordinate = (self.parent.coordinate + self.length * self.matrix.dot(self.direction))
    for child in self.children:
      child.set_motion(motion)

  def get_pose(self):
    joints = self.to_dict()
    xs, ys, zs = [], [], []
    for joint in joints.values():
      xs.append(0.056444 * joint.coordinate[0, 0])
      ys.append(0.056444 * joint.coordinate[1, 0])
      zs.append(0.056444 * joint.coordinate[2, 0])
    # Quirky axis order used by .asf
    return CMU_Pose(np.array([zs, xs, ys]).T)

  def to_dict(self):
    ret = {self.name: self}
    for child in self.children:
      ret.update(child.to_dict())
    return ret

  def pretty_print(self):
    print('===================================')
    print('joint: %s' % self.name)
    print('direction:')
    print(self.direction)
    print('limits:', self.limits)
    print('parent:', self.parent)
    print('children:', self.children)


def read_line(stream, idx):
  if idx >= len(stream):
    return None, idx
  line = stream[idx].strip().split()
  idx += 1
  return line, idx


def parse_asf(file_path):
  '''read joint data only'''
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    # meta infomation is ignored
    if line == ':bonedata':
      content = content[idx+1:]
      break

  # read joints
  joints = {'root': Joint('root', np.zeros(3), 0, np.zeros(3), [], [])}
  idx = 0
  while True:
    # the order of each section is hard-coded

    line, idx = read_line(content, idx)

    if line[0] == ':hierarchy':
      break

    assert line[0] == 'begin'

    line, idx = read_line(content, idx)
    assert line[0] == 'id'

    line, idx = read_line(content, idx)
    assert line[0] == 'name'
    name = line[1]

    line, idx = read_line(content, idx)
    assert line[0] == 'direction'
    direction = np.array([float(axis) for axis in line[1:]])

    # skip length
    line, idx = read_line(content, idx)
    assert line[0] == 'length'
    length = float(line[1])

    line, idx = read_line(content, idx)
    assert line[0] == 'axis'
    assert line[4] == 'XYZ'

    axis = np.array([float(axis) for axis in line[1:-1]])

    dof = []
    limits = []

    line, idx = read_line(content, idx)
    if line[0] == 'dof':
      dof = line[1:]
      for i in range(len(dof)):
        line, idx = read_line(content, idx)
        if i == 0:
          assert line[0] == 'limits'
          line = line[1:]
        assert len(line) == 2
        mini = float(line[0][1:])
        maxi = float(line[1][:-1])
        limits.append((mini, maxi))

      line, idx = read_line(content, idx)

    assert line[0] == 'end'
    joints[name] = Joint(
      name,
      direction,
      length,
      axis,
      dof,
      limits
    )

  # read hierarchy
  assert line[0] == ':hierarchy'

  line, idx = read_line(content, idx)

  assert line[0] == 'begin'

  while True:
    line, idx = read_line(content, idx)
    if line[0] == 'end':
      break
    assert len(line) >= 2
    for joint_name in line[1:]:
      joints[line[0]].children.append(joints[joint_name])
    for nm in line[1:]:
      joints[nm].parent = joints[line[0]]

  return joints


def parse_amc(file_path):
  with open(file_path) as f:
    content = f.read().splitlines()

  for idx, line in enumerate(content):
    if line == ':DEGREES':
      content = content[idx+1:]
      break

  frames = []
  idx = 0
  line, idx = read_line(content, idx)
  assert line[0].isnumeric(), line
  EOF = False
  while not EOF:
    joint_degree = {}
    while True:
      line, idx = read_line(content, idx)
      if line is None:
        EOF = True
        break
      if line[0].isnumeric():
        break
      joint_degree[line[0]] = [float(deg) for deg in line[1:]]
    frames.append(joint_degree)
  return frames


def test_all():
  import os
  lv0 = './data'
  lv1s = os.listdir(lv0)
  for lv1 in lv1s:
    lv2s = os.listdir('/'.join([lv0, lv1]))
    asf_path = '%s/%s/%s.asf' % (lv0, lv1, lv1)
    print('parsing %s' % asf_path)
    joints = parse_asf(asf_path)
    motions = parse_amc('./nopose.amc')
    joints['root'].set_motion(motions[0])
    joints['root'].draw()

    # for lv2 in lv2s:
    #   if lv2.split('.')[-1] != 'amc':
    #     continue
    #   amc_path = '%s/%s/%s' % (lv0, lv1, lv2)
    #   print('parsing amc %s' % amc_path)
    #   motions = parse_amc(amc_path)
    #   for idx, motion in enumerate(motions):
    #     print('setting motion %d' % idx)
    #     joints['root'].set_motion(motion)


if __name__ == '__main__':
  test_all()
  # asf_path = './133.asf'
  # amc_path = './133_01.amc'
  # joints = parse_asf(asf_path)
  # motions = parse_amc(amc_path)
  # frame_idx = 0
  # joints['root'].set_motion(motions[frame_idx])
  # joints['root'].draw()
