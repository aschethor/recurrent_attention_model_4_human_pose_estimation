"""
This file contains some important definitions of the h36m data format.
In particular it defines the joint indices and bone pairs.
"""

joints =      [    0,             1,          2,           3,            4,         5,           6,       7,     8,      9,        10,        11,            12,         13,         14,             15,         16 ]
joint_names = ['hip','right_up_leg','right_leg','right_foot','left_up_leg','left_leg', 'left_foot','spine1','neck', 'head','head-top','left-arm','left_forearm','left_hand','right_arm','right_forearm','right_hand']
root_index = 0
bones = [[0, 1], [1, 2], [2, 3],[0, 4], [4, 5], [5, 6],[0, 7], [7, 8], [8, 9], [9, 10],[8, 14], [14, 15], [15, 16],[8, 11], [11, 12], [12, 13]]
