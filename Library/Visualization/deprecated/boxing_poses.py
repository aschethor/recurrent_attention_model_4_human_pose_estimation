import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from Library.Visualization import util
import pickle
from scipy import misc

def visualize(image, poses, filename):
    """
    visualize given poses next to image and save into file...
    :image: pytorch tensor of image [w x h x 3] (TODO: ?!)
    :poses: list of poses (a pose is a 17 x 3 matrix containing the coordiantes of person vertices)
    :filename: filename to store visualization
    """

    # plot image
    fig = plt.figure(0)
    ax_img  = fig.add_subplot(1,2,1)
    plt.xlim([0.0,image.shape[1]])
    plt.ylim([0.0,image.shape[0]])
    ax_img.set_axis_off()
    ax_img.imshow(image.numpy())

    # plot poses
    ax_3d   = fig.add_subplot(122, projection='3d')
    ax_3d.xaxis.set_visible(False)
    ax_3d.yaxis.set_visible(False)
    ax_3d.set_axis_off()
    ax_3d.axis('equal')
    plt.hold(True)
    
    for pose in poses:

        # configure pose
        pose = pose.numpy()
        pose_3d_in = np.mat(pose)
        pose_3d_in = pose_3d_in.reshape([-1,3]).T
        pose_3d_in = np.multiply(pose_3d_in,np.mat([1,-1,1]).T)
        pose_3d    = pose_3d_in

        util.plot_3Dpose(ax_3d, pose_3d, bones = util.bones_h36m, set_limits=False)

    # set limits of x / y / z axis
    for axis, setter in [(ax_3d.xaxis, ax_3d.set_xlim), (ax_3d.yaxis, ax_3d.set_ylim), (ax_3d.zaxis, ax_3d.set_zlim)]:
        vmin, vmax = axis.get_data_interval()
        vrange = vmax-vmin
        vmin -= vrange*0.5#*0.05
        vmax += vrange*0.5#*0.05
        setter([vmin, vmax])

    # save figure
    plt.savefig(filename,dpi=600)
