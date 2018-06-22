"""
derivative of NeuralNet_12 but with:
- more /deeper resnet channels
- different C in loss function
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

cross_connectivity = 20

class ResnetBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ResnetBlock, self).__init__()
        
        momentum = 0.01
        self.downsample = None
        if stride !=1 or in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels,momentum = momentum)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels,momentum = momentum)
        
    def forward(self,x):
        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))+residual)
        return x

class AffineNet(nn.Module):
    def __init__(self):
        super(AffineNet, self).__init__()
        
        self.affine_size = 37
        self.affine_avrg = 5
        
        # average pooling
        self.avg_pool = nn.AvgPool2d(self.affine_avrg)
        
        # resnet layers
        resnets = []
        resnets.append(ResnetBlock(3,25))
        resnets.append(ResnetBlock(25,20))
        resnets.append(ResnetBlock(20,15))
        resnets.append(ResnetBlock(15,25,stride=2))
        resnets.append(ResnetBlock(25,20))
        resnets.append(ResnetBlock(20,15))
        resnets.append(ResnetBlock(15,30,stride=2))
        resnets.append(ResnetBlock(30,30))
        self.resnet = nn.Sequential(*resnets)
        
        self.maxPool = nn.MaxPool2d(2)
        
        self.conv = nn.Conv2d(30,100, kernel_size = 5)
        
        self.theta_position = Variable(torch.Tensor([[[0,0,1],[0,0,1]]]),requires_grad=False).cuda()
        self.theta_zoom     = Variable(torch.Tensor([[[1,0,0],[0,1,0]]]),requires_grad=False).cuda()
        
    def forward(self, images, position, zoom):
        
        batch_size = images.shape[0]
        
        # generate affine grid
        theta = position*self.theta_position+torch.exp(zoom)*self.theta_zoom
        grid = F.affine_grid(theta,torch.Size((batch_size,3,self.affine_size*self.affine_avrg,self.affine_size*self.affine_avrg)))
        
        # apply affine grid and average
        x = F.grid_sample(images,grid)
        x = self.avg_pool(x)
        
        # resnet layers
        x = self.resnet(x)
        
        x = self.maxPool(x)
        
        x = F.sigmoid(self.conv(x))
        
        x = x.view(-1,100)
        return x

class DeltaPoseNet(nn.Module):
    def __init__(self, n_joints):
        super(DeltaPoseNet, self).__init__()
        
        self.n_joints = n_joints
        
        # "recurrent" linear layers
        self.lin1 = nn.Linear(100+(n_joints-1)*cross_connectivity+n_joints*3, 100)
        self.gru2 = nn.GRU(100, 100)
        self.drp2 = nn.Dropout(0.2)
        self.gru3 = nn.GRU(100, 100)
        self.drp3 = nn.Dropout(0.2)
        self.gru4 = nn.GRU(100, 100)
        self.drp4 = nn.Dropout(0.2)
        self.lin5 = nn.Linear(200, 3)
        
    def forward(self, affine_net_input, cross_connects, relative_position, relative_zoom, hidden = None):
        """
        :affine_net_input: cuda Tensor [batch_size x 100]
        :relative_position: cuda Tensor [batch_size x n_joints x 2 x 1]
        :relative_zoom: cuda Tensor [batch_size x n_joints x 1 x 1]
        :return: cuda Tensor [batch_size x 1 x 3 x 1]
        """
        
        batch_size = affine_net_input.shape[0]
        
        if hidden is None:
            hidden = [Variable(torch.zeros(1,batch_size,100)).cuda(),Variable(torch.zeros(1,batch_size,100)).cuda(),Variable(torch.zeros(1,batch_size,100)).cuda()]
        
        h2 = hidden[0]
        h3 = hidden[1]
        h4 = hidden[2]
        
        x = torch.cat([relative_position, relative_zoom],dim=2).view(-1,self.n_joints*3)
        x = torch.cat([affine_net_input,cross_connects,x],dim=1)
        
        # fully connected layer
        x_tmp = F.sigmoid(self.lin1(x))
        x,h2 = self.gru2(x_tmp.unsqueeze(0),h2)
        x = self.drp2(x)
        x,h3 = self.gru3(x,h3)
        x = self.drp3(x)
        x,h4 = self.gru4(x,h4)
        x = self.drp4(x)
        x = torch.squeeze(x,dim=0)
        x = F.tanh(self.lin5(torch.cat([x,x_tmp],dim=1)))
        
        return x.unsqueeze(1).unsqueeze(3),[h2,h3,h4]

class NeuralNet(nn.Module):
    def __init__(self, n_joints = 17):
        super(NeuralNet, self).__init__()
        
        self.n_joints = n_joints
        self.affine_nets = [AffineNet() for i in range(n_joints)]
        self.delta_pose_nets = [DeltaPoseNet(n_joints) for i in range(n_joints)]
        self.share_weights()
        
    def named_children(self):
        for i,an in enumerate(self.affine_nets):
            yield 'an_{}'.format(i),an
        for i,dpn in enumerate(self.delta_pose_nets):
            yield 'dpn_{}'.format(i),dpn
    
    def share_weights(self):
        """
        share weights between affine nets
        this method has to be called after loading the network (?!)
        """
        shared_resnet = self.affine_nets[0].resnet
        for i in range(1,len(self.affine_nets)):
            self.affine_nets[i].resnet = shared_resnet
    
    def forward(self,images,positions=None, zooms=None, hidden=None):
        """
        :images: image batch [batch_size x n_channels x height x width]
        :positions: vector of joints positions: [batch_size x n_joints x 2 x 1]
        :zooms: vector of joint zooms: [batch_size x n_joints x 1 x 1]
        :hidden: hidden values
        """
        
        if hidden is None:
            hidden = [None for i in range(self.n_joints)]
        
        # apply affine_nets (transformation + first layers)
        tmp = []
        for i,affine_net in enumerate(self.affine_nets):
            tmp.append(affine_net(images,positions[:,i,:,:],zooms[:,i,:,:]))
        
        # calculate relative positions and apply delta_pose_nets
        delta_pose = []
        return_hidden = []
        for i,delta_pose_net in enumerate(self.delta_pose_nets):
            
            # relative position / zoom to other joints
            relative_poses = positions - positions[:,i:(i+1),:,:]
            relative_poses = relative_poses / torch.exp(zooms[:,i:(i+1),:,:])
            relative_zooms = zooms - zooms[:,i:(i+1),:,:]
            
            # absolute position / zoom of own joint
            relative_poses[:,i,:,:] = positions[:,i,:,:]
            relative_zooms[:,i,:,:] = zooms[:,i,:,:]
            
            # cross connections
            cross_connects = torch.cat([t[:,0:cross_connectivity] for t in tmp],dim=1)
            
            new_delta_pose, new_hidden = delta_pose_net(tmp[i][:,cross_connectivity:100],cross_connects,relative_poses,relative_zooms,hidden[i])
            delta_pose.append(new_delta_pose)
            return_hidden.append(new_hidden)
        delta_pose = torch.cat(delta_pose,dim=1)
        
        # update positions
        positions = positions + torch.exp(zooms) * delta_pose[:,:,0:2,:]
        zooms = zooms + delta_pose[:,:,2:3,:]
        
        return positions, zooms, return_hidden
