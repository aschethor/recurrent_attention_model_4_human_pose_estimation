"""
This file provides plotting functionality for 2d pose data
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_2d_poses(image, poses, filename):
    """
    visualize given poses next to image and save into file...
    :image: pytorch tensor (on cpu) of image [3 x h x w]
    :poses: list of poses (a pose is a 17 x 3 matrix containing the coordiantes of person vertices)
    :filename: filename to store visualization
    """
    image = image.permute(1,2,0)

    # plot image
    plt.figure(1)
    plt.clf()
    plt.imshow(image.numpy(),origin='upper',extent=[0,image.shape[1],0,image.shape[0]])

    for pose in poses:
        plt.plot([p[0] for p in pose],[-p[1]+image.shape[0] for p in pose],'o', alpha=0.5)

    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)

    # save figure
    plt.savefig(filename,dpi=600)

# default colors
_colors = torch.FloatTensor([[1,0,0],[0,1,0],[0,0,1],[0.5,0.5,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.25,0.25],[0.25,0.5,0.25],[0.25,0.25,0.5],[0.75,0.25,0],[0.75,0,0.25],[0.25,0.75,0],[0,0.75,0.25],[0.25,0,0.75],[0,0.25,0.75],[0.66,0.33,0.33],[0.33,0.66,0.33]]).unsqueeze(1).unsqueeze(1).cuda()

def plot_joint_probability_density(image, density, filename, colors=None):
    """
    plt joint probability densities
    :image: image to plot densities up on (on cpu) [3 x h x w]
    :density: joint densities (n_joints x height x width) (on gpu)
    :filename: filename to save plot
    :colors: tensor containing colors to plot of size (n_joints x 3)
    """
    image = image.permute(1,2,0)

    if colors is None:
        colors = _colors

    # plot image
    plt.figure(1)
    plt.clf()
    density_plot = torch.sum(density.unsqueeze(3)*colors, dim=0).cpu().numpy()
    plt.imshow((image.numpy()*100+(200/np.max(density_plot))*density_plot).astype(np.uint8),origin='upper',extent=[0,image.shape[1],0,image.shape[0]])

    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)

    # save figure
    plt.savefig(filename,dpi=600)

