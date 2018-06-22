"""
goal: after a fixed amount of steps, there should be an affine transformation centering a head in an image
idea: inspired by eye-movement that searches for heads in an image
"""

model_name = 'NeuralNet_3'
body_part = 'right_hand'#'head-top' #
epoch = 15
n_recurrences = 10
n_joints = 17
dataset = 'boxing'
load_state = '{}_{}_epoch_{}_final'.format(dataset,body_part,epoch)

print("""
    description:
    test {}_{} to predict {}
    """.format(model_name,load_state,body_part))

import sys
sys.path.insert(0, '/home/wandel/code')

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
from torch import nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Library.Logging import Logger
Logger.init()
from Library.Data.boxing.Datasets import OpenPoseDataset
import Library.Data.boxing.Definitions as Def
import Library.Data.utility as util

# load dataset
batch_size = 30
train_data = OpenPoseDataset(sequences=['5'])
Loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True)

n_samples = len(train_data)
n_batches = len(Loader)
avg_pool = nn.AvgPool2d(5)
affine_size = 5*20

# load neural network
neural_net, _ = Logger.load_state(model_name,load_state)
neural_net = neural_net.cuda()
neural_net.eval()

# pull batch
index = None
frame = None
camera = None
poses = None
image = None
for batch in Loader:
    index, frame, camera, poses, image = batch
    break

# preprocess batch
batch_size = int(image.shape[0])
real_position_0 = poses[:,0,Def.joint_names.index(body_part),:].unsqueeze(2).float().cuda()
real_position_0 = Variable(real_position_0, requires_grad=False)
real_position_1 = poses[:,1,Def.joint_names.index(body_part),:].unsqueeze(2).float().cuda()
real_position_1 = Variable(real_position_1, requires_grad=False)
image_in = Variable(image, requires_grad=False).cuda()

# initialize start values
start_zoom = Variable(torch.zeros(batch_size,1,1).cuda())
start_position = Variable(torch.zeros(batch_size,2,1).cuda())
start_hidden = None

theta_position = Variable(torch.Tensor([[[0,0,1],[0,0,1]]])).cuda()
theta_zoom     = Variable(torch.Tensor([[[1,0,0],[0,1,0]]])).cuda()

def plot_affine_images(images,position,zoom,i):
    theta = position*theta_position+torch.exp(zoom)*theta_zoom
    grid = F.affine_grid(theta,torch.Size((batch_size,3,affine_size,affine_size)))
    x = avg_pool(F.grid_sample(images,grid))

    # plot transformed image views
    for j in range(batch_size):
        plt.subplot(batch_size,n_recurrences+2,i+j*(n_recurrences+2)+2)
        plt.imshow(x[j].data.permute(1,2,0).cpu().numpy(),origin='upper')
        plt.axis('off')

#start plotting
plt.figure(1,figsize=(20,7))
plt.clf()
for i in range(batch_size):
    plt.subplot(batch_size,n_recurrences+2,i*(n_recurrences+2)+1)
    plt.imshow(image_in[i].data.permute(1,2,0).cpu().numpy(),origin='upper')
    plt.axis('off')
plot_affine_images(image_in,start_position,start_zoom,0)

# do several "glimpses"
for i in range(n_recurrences):
    
    # push to neural net
    start_position, start_zoom, start_hidden = neural_net(image_in,position=start_position,zoom=start_zoom,hidden = start_hidden)

    plot_affine_images(image_in,start_position,start_zoom,i+1)
plt.savefig('images/{}/{}_eye_movement.png'.format(body_part,model_name),dpi=600, bbox_inches='tight')

# plot real and predicted positions
real_position_0 = util.relative2pixel_positions(real_position_0.data,1000).cpu().numpy()
real_position_1 = util.relative2pixel_positions(real_position_1.data,1000).cpu().numpy()
pixel_position = util.relative2pixel_positions(start_position.data,1000).cpu().numpy()

def plot_rectangle(position,zoom):
    exp_zoom = torch.exp(zoom).data
    go_right_up = torch.Tensor([-1,1]).cuda().unsqueeze(0).unsqueeze(2)
    go_left_up = torch.Tensor([-1,-1]).cuda().unsqueeze(0).unsqueeze(2)
    left_top = util.relative2pixel_positions(position.data+go_left_up*exp_zoom,1000).cpu().numpy()
    right_top = util.relative2pixel_positions(position.data+go_right_up*exp_zoom,1000).cpu().numpy()
    left_down = util.relative2pixel_positions(position.data-go_right_up*exp_zoom,1000).cpu().numpy()
    right_down = util.relative2pixel_positions(position.data-go_left_up*exp_zoom,1000).cpu().numpy()
    
    color = 'r'
    
    plt.plot([  left_top[0,0,0], right_top[0,0,0]],[  left_top[0,1,0], right_top[0,1,0]],'-{}'.format(color))
    plt.plot([ right_top[0,0,0],right_down[0,0,0]],[ right_top[0,1,0],right_down[0,1,0]],'-{}'.format(color))
    plt.plot([right_down[0,0,0], left_down[0,0,0]],[right_down[0,1,0], left_down[0,1,0]],'-{}'.format(color))
    plt.plot([ left_down[0,0,0],  left_top[0,0,0]],[ left_down[0,1,0],  left_top[0,1,0]],'-{}'.format(color))
    #plt.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)

plt.figure(2,figsize=(4,4))
for batch_index in range(batch_size):
    print('.',end='')
    plt.clf()
    plt.imshow(image[batch_index].float().permute(1,2,0).numpy(),origin='upper')
    plt.plot(real_position_0[batch_index,0,0],real_position_0[batch_index,1,0],'g+')
    plt.plot(real_position_1[batch_index,0,0],real_position_1[batch_index,1,0],'g+')
    plt.plot(pixel_position[batch_index,0,0],pixel_position[batch_index,1,0],'r+')
    plot_rectangle(start_position[batch_index],start_zoom[batch_index])
    plt.axis('off')
    plt.savefig('images/{}/{}_prediction_{}.png'.format(body_part,model_name,batch_index),dpi=600)

print("done :)")
