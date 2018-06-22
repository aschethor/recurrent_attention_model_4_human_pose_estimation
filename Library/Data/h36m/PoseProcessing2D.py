"""
This File provides you:
- Joint Probability Densities from 2D Pose Data (use annotations = 'annotation_2D_cap' in Datasets)
- loss function: cross_entropy between prediction and JPD generated from Pose Data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# define height and width of image and number of joints to detect
_height, _width, _n_joints = 1080,1920,17
# generate x,y mesh grid
_x,_y = None,None
# create uniform distribution (needed in joint_probability_density)
_uniform = None

def init(height=1080,width=1920,n_joints=17):
    """init global variables"""
    global _height,_width,_n_joints,_x,_y,_uniform
    _height = height
    _width = width
    _n_joints = n_joints
    x = np.linspace(0,width-1,width)
    y = np.linspace(0,height-1,height)
    x,y = np.meshgrid(x,y,indexing='xy')
    _x = (torch.Tensor(x)*torch.ones([n_joints,1,1])).cuda()
    _y = (torch.Tensor(y)*torch.ones([n_joints,1,1])).cuda()
    _uniform = torch.ones([n_joints,height,width]).cuda()/(width*height)
    return

def gaussian(mu,sigma):
    """
    generate gaussian probability densities from pose data (mu)
    :mu: joint positions (n_joints x 2)
    :sigma: sigma of gaussian
    :return: density images (n_joints x height x width)
    """
    sigma_sqrd = sigma*sigma
    xv = _x-mu[:,0].contiguous().unsqueeze(1).unsqueeze(2).type(torch.FloatTensor).cuda()
    yv = _y-mu[:,1].contiguous().unsqueeze(1).unsqueeze(2).type(torch.FloatTensor).cuda()
    return 1/(2*np.pi*sigma_sqrd)*torch.exp(-(xv*xv+yv*yv)/(2*sigma_sqrd))

def joint_probability_density(poses_batch, n_misses=1, sigma=20):
    """
    create joint probability densities of batch of 2d poses (use 'annotation_2D_cap' in OpenPoseDataset)
    :poses_batch: batch of poses (batch_size x n_persons x n_joints x 2)
    :n_misses: number of persons in image not registered in poses
    :return: density images (batch_size x n_joints x height x width)
    """

    density = torch.zeros([poses_batch.shape[0],_n_joints,_height,_width]).cuda()
    for i,poses in enumerate(poses_batch):
        for pose in poses:
            density[i] += gaussian(pose,sigma)
        density[i] += n_misses*_uniform
    return density

def cross_entropy(p_output, p_reality):
    """
    returns cross entropy of p_output with p_reality
    This corresponds somewhat to the log likelihood of p_output being correct (?!)
    """
    return -torch.sum(p_reality*torch.log(p_output))

