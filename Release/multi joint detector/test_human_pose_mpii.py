"""
goal: after a fixed amount of steps, there should be an affine transformation centering a head in an image
idea: inspired by eye-movement that searches for heads in an image
"""

model_name = 'NeuralNet_1'
epoch = 109
load_state = 'mpii_epoch_{}_final'.format(epoch)
n_recurrences = 10
n_joints = 16

print("""
    description:
    test {}_{} to predict human pose (MPII)
    """.format(model_name,load_state))

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
from Library.Data.mpii.Datasets import MPIIDataset
from Library.Data.mpii import Definitions
from Library.Data.utility import accuracy_pckh
from Library.Data.utility import n_accurate_predictions_pckh
import Library.Data.utility as util

# load dataset
batch_size = 30
img_size = 1000
#TODO: test set often contains labels for only one person -> returns wrong accuracy if other persons are found
train_data = MPIIDataset(img_size=img_size, train_test_valid='valid_victor')
Loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True)

n_samples = len(train_data)
n_batches = len(Loader)

# load neural network
#neural_net, _ = Logger.load_state(model_name,'{}'.format(load_state))
neural_net = torch.load("Logger/states/{}_{}.mdl".format(model_name,load_state))
neural_net = neural_net.cuda(0)
neural_net.eval()
#neural_net.train()

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
image_in = Variable(image, requires_grad=False,volatile=True).cuda(0)

# initialize start values
start_zooms = Variable(torch.zeros(batch_size,n_joints,1,1).cuda(0),requires_grad=False,volatile=True)
start_positions = Variable(torch.zeros(batch_size,n_joints,2,1).cuda(0),requires_grad=False,volatile=True)
start_hidden = None

# do several "glimpses"
for i in range(n_recurrences):
    # push to neural net
    start_positions, start_zooms, start_hidden = neural_net(image_in,positions=start_positions,zooms=start_zooms,hidden=start_hidden)

# plot real and predicted positions
pixel_positions = util.relative2pixel_positions(start_positions.data,img_size).cpu().numpy()

def print_pckh_metrics(metrics,header = "joint accuracies"):
    print(header)
    for i,joint_name in enumerate(Definitions.joint_names):
        print("{}: {}".format(joint_name,metrics[i]))

def print_pckh_hits(metrics,header = "joint hits"):
    print(header)
    for i,joint_name in enumerate(Definitions.joint_names):
        print("{}: {} of {}".format(joint_name,metrics[0][i],metrics[1][i]))
    

plt.figure(2,figsize=(6,6))
for batch_index in range(batch_size):
    print_pckh_metrics(accuracy_pckh(poses[batch_index].unsqueeze(0),start_positions.data[batch_index,:,:,0].unsqueeze(0).cpu(),outlier_detection=True),"correct joints in image {}".format(batch_index))
    print_pckh_hits(n_accurate_predictions_pckh(poses[batch_index].unsqueeze(0),start_positions.data[batch_index,:,:,0].unsqueeze(0).cpu(),outlier_detection=True),"correct joint hits in image {}".format(batch_index))
    plt.clf()
    plt.imshow(image[batch_index].float().permute(1,2,0).numpy(),origin='upper')
    for i in range(poses.shape[1]):
        real_positions = poses[:,i,:,:].unsqueeze(3).float().cuda(0)
        real_positions = Variable(real_positions, requires_grad=False)
        real_positions = util.relative2pixel_positions(real_positions.data,img_size).cpu().numpy()
        plt.plot(real_positions[batch_index,:,0,0],real_positions[batch_index,:,1,0],'+')
    plt.plot(pixel_positions[batch_index,:,0,0],pixel_positions[batch_index,:,1,0],'+')
    for bone in Definitions.bones:
        if 'right' in Definitions.joint_names[bone[0]] or 'right' in Definitions.joint_names[bone[1]]:
            plt.plot([pixel_positions[batch_index,bone[0],0,0],pixel_positions[batch_index,bone[1],0,0]],[pixel_positions[batch_index,bone[0],1,0],pixel_positions[batch_index,bone[1],1,0]],'-g',linewidth=2)
        elif 'left' in Definitions.joint_names[bone[0]] or 'left' in Definitions.joint_names[bone[1]]:
            plt.plot([pixel_positions[batch_index,bone[0],0,0],pixel_positions[batch_index,bone[1],0,0]],[pixel_positions[batch_index,bone[0],1,0],pixel_positions[batch_index,bone[1],1,0]],'-r',linewidth=2)
        else:
            plt.plot([pixel_positions[batch_index,bone[0],0,0],pixel_positions[batch_index,bone[1],0,0]],[pixel_positions[batch_index,bone[0],1,0],pixel_positions[batch_index,bone[1],1,0]],'-b',linewidth=2)
    plt.axis('off')
    plt.savefig('images/mpi_{}_prediction_{}_{}_valid.png'.format(model_name,batch_index,epoch),dpi=600)

print_pckh_metrics(accuracy_pckh(poses,start_positions.data[:,:,:,0].cpu(),outlier_detection=True))
print_pckh_hits(n_accurate_predictions_pckh(poses,start_positions.data[:,:,:,0].cpu(),outlier_detection=True))
    
print("done :)")
