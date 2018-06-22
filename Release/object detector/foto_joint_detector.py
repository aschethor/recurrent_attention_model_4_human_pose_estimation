"""
goal: after a fixed amount of steps, there should be an affine transformation centering a head in an image
idea: inspired by eye-movement that searches for heads in an image
"""

model_name = 'NeuralNet_3'
body_part = 'head-top' #'right_hand'#'head-top' #
epoch = 15
n_recurrences = 10
n_joints = 17
dataset = 'boxing'
load_state = '{}_{}_epoch_{}_final'.format(dataset,body_part,epoch)

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
neural_net, _ = Logger.load_state(model_name,load_state)
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
start_zoom = Variable(torch.zeros(1,1,1).cuda(), requires_grad=False,volatile=True)
start_position = Variable(torch.zeros(1,2,1).cuda(), requires_grad=False,volatile=True)
start_hidden = None

image = torch.squeeze(image_in.data.cpu(),0)

# plot predicted positions during glimpses
plt.figure(1,figsize=(4*n_recurrences,4))
plt.clf()

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


# do several "glimpses"
for i in range(n_recurrences):
    plt.subplot(1,n_recurrences+1,i+1)
    plt.imshow(image.float().permute(1,2,0).numpy(),origin='upper')
    plt.axis('off')
    plot_rectangle(start_position,start_zoom)
    plt.xlim([0,1000])
    plt.ylim([0,1000])
    plt.gca().invert_yaxis()
    
    # push to neural net
    start_position, start_zoom, start_hidden = neural_net(image_in,position=start_position,zoom=start_zoom,hidden=start_hidden)

plt.subplot(1,n_recurrences+1,n_recurrences+1)
plt.imshow(image.float().permute(1,2,0).numpy(),origin='upper')
plt.axis('off')
plot_rectangle(start_position,start_zoom)
plt.savefig('{}{}_{}_prediction_{}_{}_{}_glimpses.png'.format(folder,sys.argv[1],model_name,body_part,dataset,epoch),dpi=600)

print("done :)")
