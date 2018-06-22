import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import math

def neighbor_connected_2D(images,weights,bias,stride=[1,1]):
    """
    applies a neighbor connected layer
    remark: output_height = (image_height-kernel_height+1)//stride[0] and output_width = (image_width-kernel_width+1)//stride[1]
    :images: Tensor containing batch of images: [batchsize x n_in_channels x image_height x image_width]
    :weights: Tensor containing weights: [n_in_channels x n_out_channels x output_height x output_width x kernel_height x kernel_width]
    :bias: Tensor containing bias values: [n_out_channels x output_height x output_width]
    :returns: batch of images: [batchsize x n_out_channels x output_height x output_width]
    """
    images = images.unsqueeze(2)
    weights = weights.unsqueeze(0)
    output_height = weights.shape[3]
    output_width = weights.shape[4]
    n_out_channels = weights.shape[2]
    output = bias.unsqueeze(0).expand(images.shape[0],-1,-1,-1)
    for i in range(weights.shape[5]):
        for j in range(weights.shape[6]):
            tmp = torch.sum(images[:,:,:,i:(i+output_height*stride[0]):stride[0],j:(j+output_width*stride[1]):stride[1]]*weights[:,:,:,:,:,i,j],dim=1)
            output = output + tmp
    return output


class NeighborLayer2D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,image_shape,stride=[1,1]):
        """
        implementation of a neighbor Layer:
        corresponds to a sparsely connected Layer, that connects the cells of the input layer within the range of kernel_size to the output layer
        the stride can be used to sparsen the output layer and get more global connections - it works similar to the stride in a convolutional layer
        :in_channels: number of channels of the input
        :out_channels: number of channels to output
        :kernel_size: kernel_size of neighbor layer - range of connections from input layer to output layer (height and width)
        :image_shape: height and width of the input layer (needed for neighborLayer in contrast to convolutional layer)
        :stride: stride of the neighbor layer
        """
        super(NeighborLayer2D, self).__init__()
        output_height = (image_shape[0]-kernel_size[0]+1)//stride[0]
        output_width = (image_shape[1]-kernel_size[1]+1)//stride[1]
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.weight = Parameter(torch.Tensor(in_channels,out_channels,output_height,output_width,kernel_size[0],kernel_size[1]))
        self.bias = Parameter(torch.Tensor(out_channels,output_height,output_width))
        self.stride = stride
        self.reset_parameters()

    def reset_parameters(self):
        """
        resets weight and bias values to a random uniform distribution
        """
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self,x):
        """
        calculates output of neighbor layer
        :x: Tensor containing batch of images: [batchsize x n_in_channels x image_height x image_width]
        :return: batch of images: [batchsize x n_out_channels x output_height x output_width]
        remark: output_height = (image_height-kernel_height+1)//stride[0] and output_width = (image_width-kernel_width+1)//stride[1]
        """
        return neighbor_connected_2D(x,self.weight,self.bias,self.stride)
