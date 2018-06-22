"""
goal: after a fixed amount of steps, there should be an affine transformation centering a head in an image
idea: inspired by eye-movement that searches for heads in an image
"""

model_name = 'NeuralNet_2_mpii'
epoch = 53
load_state = 'epoch_{}_final'.format(epoch)
n_recurrences = 10
n_joints = 16

import sys
sys.path.insert(0, '/home/wandel/code')

print("""
    description:
    test {}_{} to predict {}
    """.format(model_name,load_state,sys.argv[1]))

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
from torch import nn
from PIL import Image
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Library.Logging import Logger
Logger.init()
from Library.Data.mpii import Definitions
import Library.Data.utility as util

folder = 'fotos/'

# load neural network
neural_net, _ = Logger.load_state(model_name,'{}'.format(load_state))
neural_net = neural_net.cuda()
neural_net.eval()

# load image
image = Image.open('{}{}.jpg'.format(folder,sys.argv[1]))
transform = transforms.Compose([
    transforms.CenterCrop(max(image.size)),
    transforms.Resize([1000,1000]),
    #transforms.RandomRotation([90,90]), # this operation should be done directly on the image: > convert myimg.jpg -rotate -90 myimg_rotated.jpg
    transforms.ToTensor()
])
image = transform(image).unsqueeze(0)
image_in = Variable(image, requires_grad=False,volatile=True).cuda()

# initialize start values
start_zooms = Variable(-0*torch.ones(1,n_joints,1,1).cuda(), requires_grad=False,volatile=True)
start_positions = Variable(torch.zeros(1,n_joints,2,1).cuda(), requires_grad=False,volatile=True)
start_hidden = None

image = torch.squeeze(image_in.data.cpu(),0)

# plot predicted positions during glimpses
plt.figure(1,figsize=(4*n_recurrences,4))
plt.clf()

def plot_rectangles(positions,zooms):
    exp_zooms = torch.exp(zooms).data
    go_right_up = torch.Tensor([-1,1]).cuda().unsqueeze(0).unsqueeze(1).unsqueeze(3)
    go_left_up = torch.Tensor([-1,-1]).cuda().unsqueeze(0).unsqueeze(1).unsqueeze(3)
    left_top = util.relative2pixel_positions(positions.data+go_left_up*exp_zooms,1000).cpu().numpy()
    right_top = util.relative2pixel_positions(positions.data+go_right_up*exp_zooms,1000).cpu().numpy()
    left_down = util.relative2pixel_positions(positions.data-go_right_up*exp_zooms,1000).cpu().numpy()
    right_down = util.relative2pixel_positions(positions.data-go_left_up*exp_zooms,1000).cpu().numpy()
    for i in range(left_top.shape[1]):
        color = 'b'
        if 'right' in Definitions.joint_names[i]:
            color = 'g'
        if 'left' in Definitions.joint_names[i]:
            color = 'r'
            
        plt.plot([  left_top[0,i,0,0], right_top[0,i,0,0]],[  left_top[0,i,1,0], right_top[0,i,1,0]],'-{}'.format(color))
        plt.plot([ right_top[0,i,0,0],right_down[0,i,0,0]],[ right_top[0,i,1,0],right_down[0,i,1,0]],'-{}'.format(color))
        plt.plot([right_down[0,i,0,0], left_down[0,i,0,0]],[right_down[0,i,1,0], left_down[0,i,1,0]],'-{}'.format(color))
        plt.plot([ left_down[0,i,0,0],  left_top[0,i,0,0]],[ left_down[0,i,1,0],  left_top[0,i,1,0]],'-{}'.format(color))
        plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)


def plot_sceleton(image,positions):
    pixel_positions = util.relative2pixel_positions(positions.data,1000).cpu().numpy()
    plt.imshow(image.float().permute(1,2,0).numpy(),origin='upper')
    plt.axis('off')
    for bone in Definitions.bones:
        if 'right' in Definitions.joint_names[bone[0]] or 'right' in Definitions.joint_names[bone[1]]:
            plt.plot([pixel_positions[0,bone[0],0,0],pixel_positions[0,bone[1],0,0]],[pixel_positions[0,bone[0],1,0],pixel_positions[0,bone[1],1,0]],'-g',linewidth=1)
        elif 'left' in Definitions.joint_names[bone[0]] or 'left' in Definitions.joint_names[bone[1]]:
            plt.plot([pixel_positions[0,bone[0],0,0],pixel_positions[0,bone[1],0,0]],[pixel_positions[0,bone[0],1,0],pixel_positions[0,bone[1],1,0]],'-r',linewidth=1)
        else:
            plt.plot([pixel_positions[0,bone[0],0,0],pixel_positions[0,bone[1],0,0]],[pixel_positions[0,bone[0],1,0],pixel_positions[0,bone[1],1,0]],'-b',linewidth=1)

# do several "glimpses"
for i in range(n_recurrences):
    plt.subplot(1,n_recurrences+1,i+1)
    plot_sceleton(image,start_positions)
    plot_rectangles(start_positions,start_zooms)
    plt.xlim([0,1000])
    plt.ylim([0,1000])
    plt.gca().invert_yaxis()
    
    # push to neural net
    start_positions, start_zooms, start_hidden = neural_net(image_in,positions=start_positions,zooms=start_zooms,hidden=start_hidden)

plt.subplot(1,n_recurrences+1,n_recurrences+1)
plot_sceleton(image,start_positions)
plot_rectangles(start_positions,start_zooms)
plt.savefig('{}{}_prediction_mpii_{}_glimpses.png'.format(folder,sys.argv[1],epoch),dpi=600)

# plot final positions
plt.figure(2,figsize=(6,6))
plt.clf()
plot_sceleton(image,start_positions)
plt.savefig('{}{}_prediction_mpii_{}.png'.format(folder,sys.argv[1],epoch),dpi=600)

print("done :)")
