"""
This file contains some important definitions of the mpii data format.
In particular it defines the joint indices and bone pairs.
"""

joints =      [            0,           1,          2,         3,          4,           5,        6,       7,           8,          9,           10,           11,              12,             13,          14,          15]
joint_names = ['right_ankle','right_knee','right_hip','left_hip','left_knee','left_ankle', 'pelvis','thorax','upper_neck', 'head_top','right_wrist','right_elbow','right_shoulder','left_shoulder','left_elbow','left_wrist']
root_index = 6
bones = [[0,1],[1,2],[3,4],[4,5],[2,6],[3,6],[6,7],[7,8],[8,9],[8,12],[10,11],[11,12],[8,13],[13,14],[14,15]]
