"""
This file contains some important definitions of the h36m data format.
In particular it defines the joint indices and bone pairs.
"""
joints =      [         0,     1,           2,              3,           4,          5,             6,          7,             8,         9,           10,           11,        12,         13,    14 ]
joint_names = ['head-top','head', 'right_arm','right_forearm','right_hand', 'left_arm','left_forearm','left_hand','right_up_leg','right_leg','right_foot','left_up_leg','left_leg','left_foot','spine']

root_index = 14
bones = [[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,14],[14,8],[8,9],[9,10],[14,11],[11,12],[12,13]]
