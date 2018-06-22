"""
This file provides plotting functionality for 3d pose data
"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pickle
from scipy import misc
from scipy.linalg import norm

def plot_3d_cylinder(ax, p0, p1, radius=5, color=(0.5, 0.5, 0.5)):
    """
    plots a 3d cylinder from p0 to p1 into ax.
    :radius: radius of cylinder to be plotted
    :color: color of cylinder to be plotted
    """
    num_samples = 8
    origin = np.array([0, 0, 0])
    #vector in direction of axis
    v = p1 - p0
    mag = norm(v)
    if mag==0: # prevent division by 0 for bones of length 0
        return np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0)),np.zeros((0,0))
    #unit vector in direction of axis
    v = v / mag
    #make some vector not in the same direction as v
    not_v = np.array([1, 0, 0])
    eps = 0.00001
    if norm(v-not_v)<eps:
        not_v = np.array([0, 1, 0])
    #make vector perpendicular to v
    n1 = np.cross(v, not_v)
    n1 /= norm(n1)
    #make unit vector perpendicular to v and n1
    n2 = np.cross(v, n1)
    #surface ranges over t from 0 to length of axis and 0 to 2*pi
    t = np.linspace(0, mag, 2)
    theta = np.linspace(0, 2 * np.pi, num_samples)
    #use meshgrid to make 2d arrays
    t, theta = np.meshgrid(t, theta)
    #generate coordinates for surface
    X, Y, Z = [p0[i] + v[i] * t + radius * np.sin(theta) * n1[i] + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
    #ax.plot_surface(X, Y, Z, color=color, alpha=0.25, shade=True)
    c = np.ones( (list(X.shape)+[4]) )
    c[:,:] = color #(1,1,1,0) #color
    return X, Y, Z, c

def plot_3d_pose(ax, pose_3d, bones, radius=10, colormap='gist_rainbow', color_order=[0, 5, 9, 15, 2, 10, 12, 4, 14, 13, 11, 3, 7, 8, 6, 1]):
    """
    plots 3d pose into ax.
    :pose_3d: numpy array containing joints in 3d (size: 3 x number of joints)
    :bones: list of index pairs connecting corresponding joints (bones)
    """
    
    pose_3d = np.reshape(pose_3d, (3, -1))

    ax.view_init(elev=0, azim=-90)
    
    cmap = plt.get_cmap(colormap)

    X,Y,Z = np.squeeze(np.array(pose_3d[0,:])), np.squeeze(np.array(pose_3d[2,:])), np.squeeze(np.array(pose_3d[1,:]))
    
    XYZ = np.vstack([X,Y,Z])
    
    # dummy bridge that connects different components (set to transparent) 
    def bridge_vertices(xs,ys,zs,cs, x,y,z,c):
        num_samples = x.shape[0]
        if num_samples == 0: # don't build a bridge if there is no data
            return
        if len(cs) > 0:
            x_bridge = np.hstack([xs[-1][:,-1].reshape(num_samples,1), x[:,0].reshape(num_samples,1)])
            y_bridge = np.hstack([ys[-1][:,-1].reshape(num_samples,1), y[:,0].reshape(num_samples,1)])
            z_bridge = np.hstack([zs[-1][:,-1].reshape(num_samples,1),z[:,0].reshape(num_samples,1)])
            c_bridge = np.ones( (num_samples,2,4) )
            c_bridge[:,:] = np.array([0,0,0,0])
            xs.append(x_bridge)
            ys.append(y_bridge)
            zs.append(z_bridge)
            cs.append(c_bridge)
        xs.append(x)
        ys.append(y)
        zs.append(z)
        cs.append(c)
        return
        
    maximum = max(color_order) #len(bones)
    xs = []
    ys = []
    zs = []
    cs = []
    for i, bone in enumerate(bones):
        assert i < len(color_order)
        colorIndex = (color_order[i] * cmap.N / float(maximum))
        color = cmap(int(colorIndex))
        x,y,z,c = plot_3d_cylinder(ax, XYZ[:,bone[0]], XYZ[:,bone[1]], radius=radius, color=color)
        bridge_vertices(xs,ys,zs,cs, x,y,z,c)
        
    if len(xs) == 0:
        return
        
    # merge all sufaces together to one big one
    x_full = np.hstack(xs)
    y_full = np.hstack(ys)
    z_full = np.hstack(zs)
    c_full = np.hstack(cs)

    ax.plot_surface(x_full, y_full, z_full, rstride=1, cstride=1, facecolors=c_full, linewidth=0, antialiased=True)

def plot_3d_poses(image, poses, bones, filename):
    """
    visualize given poses next to image and save into file...
    :image: pytorch tensor on cpu of image [3 x h x w]
    :poses: list of poses (a pose is a 17 x 3 matrix containing the coordiantes of person vertices)
    :filename: filename to store visualization
    """
    image = image.permute(1,2,0)

    # plot image
    fig = plt.figure(0)
    ax_img  = fig.add_subplot(1,2,1)
    plt.xlim([0.0,image.shape[1]])
    plt.ylim([0.0,image.shape[0]])
    ax_img.set_axis_off()
    ax_img.imshow(image.numpy(),origin='upper',extent=[0,image.shape[1],0,image.shape[0]])

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

        plot_3d_pose(ax_3d, pose_3d, bones = bones)

    # set limits of x / y / z axis
    for axis, setter in [(ax_3d.xaxis, ax_3d.set_xlim), (ax_3d.yaxis, ax_3d.set_ylim), (ax_3d.zaxis, ax_3d.set_zlim)]:
        vmin, vmax = axis.get_data_interval()
        vrange = vmax-vmin
        vmin -= vrange*0.5#*0.05
        vmax += vrange*0.5#*0.05
        setter([vmin, vmax])

    # save figure
    plt.savefig(filename,dpi=600)
