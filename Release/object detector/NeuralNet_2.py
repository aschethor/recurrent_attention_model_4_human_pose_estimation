"""
exactly like NeuralNet_1
but with gru units replaced by simple linear units
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# define recurrent convolutional spatial transformer network
# -> later also with recurrent units (GRU / LSTM)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.affine_size = 5*20
        
        # average pooling
        self.avg_pool = nn.AvgPool2d(5)
        
        # "convolutional" layers
        self.lin_conv1 = nn.Linear(20*20*3, 1000)
        self.lin_conv2 = nn.Linear(1000, 100)
        self.lin_conv3 = nn.Linear(200, 100)
        
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

        # "convolutional" neural network
        x_tmp = F.sigmoid(self.lin_conv1(x.view(-1,20*20*3)))
        x = F.sigmoid(self.lin_conv2(x_tmp))
        x = F.sigmoid(self.lin_conv3(torch.cat([x,x_tmp[:,0:100]],dim=1)))
        
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
