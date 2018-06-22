"""
exactly like NeuralNet_4 (in old) but without cost input and confidence output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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

# define recurrent convolutional spatial transformer network
# -> later also with recurrent units (GRU / LSTM)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.affine_size = 5*37
        
        # average pooling
        self.avg_pool = nn.AvgPool2d(5)
        
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
        
        # "recurrent" linear layers
        self.lin1 = nn.Linear(100+3, 100)
        self.lin2 = nn.Linear(100, 100)
        self.lin3 = nn.Linear(100, 100)
        self.lin4 = nn.Linear(100, 100)
        self.lin5 = nn.Linear(200, 3)
        
        
        self.theta_position = Variable(torch.Tensor([[[0,0,1],[0,0,1]]])).cuda()
        self.theta_zoom     = Variable(torch.Tensor([[[1,0,0],[0,1,0]]])).cuda()

    def forward(self, images, position, zoom, hidden = None):
        
        batch_size = images.shape[0]
        
        # do affine transformation
        theta = position*self.theta_position+torch.exp(zoom)*self.theta_zoom
        grid = F.affine_grid(theta,torch.Size((batch_size,3,self.affine_size,self.affine_size)))
        x = F.grid_sample(images,grid)
        x = self.avg_pool(x)
        
        # resnet layers
        x = self.resnet(x)
        x = self.maxPool(x)
        x = F.sigmoid(self.conv(x))
        x = x.view(-1,100)

        # add recursion
        x = torch.cat([x,torch.squeeze(torch.cat([position,zoom],dim=1),dim=2)],dim=1)
        
        # fully connected layer
        x_tmp = F.sigmoid(self.lin1(x))
        x = F.sigmoid(self.lin2(x_tmp))
        x = F.sigmoid(self.lin3(x))
        x = F.sigmoid(self.lin4(x))
        x = F.tanh(self.lin5(torch.cat([x,x_tmp],dim=1)))
        
        # update recurrence / position / zoom
        x = x.unsqueeze(2)
        position = position+torch.exp(zoom)*x[:,0:2]
        zoom = zoom+x[:,2:3]
        
        return position, zoom, hidden
