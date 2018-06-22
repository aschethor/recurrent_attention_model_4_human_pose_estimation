"""
goal: after a fixed amount of steps, there should be an affine transformation centering a head in an image
idea: inspired by eye-movement that searches for heads in an image
"""

model_name = 'NeuralNet_4'
body_part = 'right_hand'#'head-top' #
full_data = True
load_state = None #'epoch_9_final' #
n_recurrences = 10
n_joints = 17
n_max_persons = 2
C = 2
dataset = 'boxing'

print("""
    description:
    training of {} on {} with full_data: {}
    """.format(model_name,body_part,full_data))

import sys
sys.path.insert(0, '/home/wandel/code')

import torch
torch.manual_seed(0)
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Library.Logging import Logger
Logger.init()
from Library.Data.boxing.Datasets import OpenPoseDataset
from Library.Data.boxing.Datasets import ParisDataset
from Library.Data.boxing import Definitions
from Library.Data.utility import Concat
from Library.Data.utility import accuracy_pckh
from Library.Data.utility import n_accurate_predictions_pckh
NeuralNet = __import__(model_name).NeuralNet
import time
import util


# time logging
Logger.t_step()
t_startup = 0
t_data_load = 0
t_to_gpu = 0
t_neural_net = 0
t_backward = 0
t_testing = 0

# load neural network and optimizer
if load_state is None:
    # generate neural network
    neural_net = NeuralNet()
    neural_net = neural_net.cuda()

    # define optimizer
    optimizer = optim.Adam(neural_net.parameters())
else:
    print("load network: {}_{}_final".format(load_state,body_part))
    # load neural network
    neural_net, optimizer = Logger.load_state(model_name,'{}_{}'.format(body_part,load_state))
    neural_net = neural_net.cuda()
    try:
        neural_net.share_weights()
    except AttributeError:
        print("weights can not be shared")

# load dataset
joint_index = Definitions.joint_names.index(body_part)
head_index = Definitions.joint_names.index('head-top')
neck_index = Definitions.joint_names.index('head')
batch_size = 16
if full_data:
    train_data_1 = OpenPoseDataset(sequences=['1','2','4','8','9'],transform=transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.3))
    train_data_2 = ParisDataset()
    train_data = Concat([train_data_1,train_data_2])
    test_data = OpenPoseDataset(sequences=['5'])
else:
    train_data = OpenPoseDataset(sequences=['1'],cameras=['0'])
    test_data = OpenPoseDataset(sequences=['5'],cameras=['0'])
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=True)

n_train_samples = len(train_data)
n_train_batches = len(train_loader)
n_test_samples = len(test_data)
n_test_batches = len(test_loader)

print("startup time: {}".format(Logger.t_step()))

# train network
for epoch in range(20):
    print("starting with epoch: {}".format(epoch+1))
    
    # training
    neural_net.train()
    for i,batch in enumerate(train_loader):
        
        # preprocess batch
        index, frame, camera, poses, image = batch
        t_data_load += Logger.t_step()
        
        # preprocess batch
        batch_size = int(image.shape[0])
        real_position = []
        for j in range(n_max_persons):
            real_position.append(Variable(poses[:,j,joint_index,:].unsqueeze(2).float().cuda(), requires_grad=False))
        image_in = Variable(image).cuda()

        # time to move data to GPU
        t_to_gpu += Logger.t_step()

        # initialize start values
        start_zoom = Variable(0.2*(torch.rand(batch_size,1,1).cuda()-0.5)) # Variable(torch.zeros(batch_size,1,1).cuda())
        start_position = Variable(0.2*(torch.zeros(batch_size,2,1).cuda()-0.5)) # Variable(torch.zeros(batch_size,2,1).cuda())
        start_hidden = None
        
        # initialize loss
        loss = []
        for j in range(n_max_persons):
            loss.append(Variable(torch.zeros(batch_size,1,1)).cuda())
        
        # do several "glimpses"
        for j in range(n_recurrences):
            
            # push to neural net
            start_position, start_zoom, start_hidden = neural_net(image_in,position=start_position,zoom=start_zoom,hidden = start_hidden)

            # loss function
            exp_start_zoom = torch.exp(start_zoom)
            
            for k in range(n_max_persons):
                # temporary variables necessary to circumvene unlabeled joints
                loss_tmp_1 = (real_position[k]-start_position)/exp_start_zoom
                loss_tmp_1[real_position[k]==-1]=0
                loss_tmp_2 = start_zoom.clone()
                loss_tmp_2[real_position[k][:,0:1,:]==-1]=0
                loss[k] = loss[k]+torch.sum((C*C/4*torch.sum(torch.pow(loss_tmp_1,2),2).unsqueeze(2)+loss_tmp_2),1)
            
            del loss_tmp_1, loss_tmp_2
            
            # "cut" gradient backpropagation on position and zoom
            start_position = Variable(start_position.data)
            start_zoom = Variable(start_zoom.data)
        
        #take person with minimal loss
        total_loss = loss[0]
        for j in range(1,n_max_persons):
            # ignore non-labeled persons -> set their loss value very high
            loss[j][loss[j].data==0]=100000
            total_loss = torch.min(total_loss,loss[j])
        total_loss = torch.mean(total_loss)
        
        # time for forward pass
        t_neural_net += Logger.t_step()
        
        # weight update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # time for backpropagation
        t_backward += Logger.t_step()
        
        # log metrics
        Logger.log("{}_boxing_train_loss".format(model_name),body_part,epoch*n_train_batches+i,total_loss.data[0])
        
        if (i%10)==0:
            accuracy = accuracy_pckh(poses,start_position.unsqueeze(1).repeat(1,n_joints,1,1).data[:,:,:,0].cpu(),head_index=head_index,neck_index=neck_index,outlier_detection=False)
            Logger.log("{}_boxing_train_acc".format(model_name),body_part,epoch*n_train_batches+i,accuracy[joint_index])
            
            hits, of = n_accurate_predictions_pckh(poses,start_position.unsqueeze(1).repeat(1,n_joints,1,1).data[:,:,:,0].cpu(),head_index=head_index,neck_index=neck_index,outlier_detection=False)
            Logger.log("{}_boxing_train_hits".format(model_name),body_part,epoch*n_train_batches+i,"{},{}".format(hits[joint_index],of[joint_index]))
        
        print("train step {} / {} done, t_data_load = {}, t_to_gpu = {}, t_neural_net = {}, t_backward = {}".format(i+1,n_train_batches,t_data_load,t_to_gpu,t_neural_net,t_backward))
        
        # save model
        #if (i+1)%2000==0:
        #    Logger.save_state(model_name,'{}_{}_epoch_{}_batch_{}'.format(dataset,body_part,epoch+1,i+1),neural_net,optimizer)
    
    del index, frame, camera, poses, image, image_in, start_position, real_position, exp_start_zoom, start_zoom, start_hidden, loss, total_loss
    torch.cuda.empty_cache()
    
    # testing
    neural_net.eval()
    for i,batch in enumerate(test_loader):
        
        # preprocess batch
        index, frame, camera, poses, image = batch
        
        # preprocess batch
        batch_size = int(image.shape[0])
        real_position = []
        for j in range(n_max_persons):
            real_position.append(Variable(poses[:,j,joint_index,:].unsqueeze(2).float().cuda(), requires_grad=False,volatile=True))
        image_in = Variable(image, requires_grad=False,volatile=True).cuda()

        # initialize start values
        start_zoom = Variable(torch.zeros(batch_size,1,1).cuda(), requires_grad=False,volatile=True)
        start_position = Variable(torch.zeros(batch_size,2,1).cuda(), requires_grad=False,volatile=True)
        start_hidden = None
        
        # initialize loss
        loss = []
        for j in range(n_max_persons):
            loss.append(Variable(torch.zeros(batch_size,1,1), requires_grad=False,volatile=True).cuda())
        
        # do several "glimpses"
        for j in range(n_recurrences):
            
            # push to neural net
            start_position, start_zoom, start_hidden = neural_net(image_in,position=start_position,zoom=start_zoom,hidden=start_hidden)
        
            # loss function
            exp_start_zoom = torch.exp(start_zoom)
            
            for k in range(n_max_persons):
                # temporary variables necessary to circumvene unlabeled joints
                loss_tmp_1 = (real_position[k]-start_position)/exp_start_zoom
                loss_tmp_1[real_position[k]==-1]=0
                loss_tmp_2 = start_zoom.clone()
                loss_tmp_2[real_position[k][:,0:1,:]==-1]=0
                #loss[k] = loss[k]+torch.sum((torch.sum(torch.pow(loss_tmp_1,2),2).unsqueeze(2)+4*loss_tmp_2),1)
                loss[k] = loss[k]+torch.sum((C*C/4*torch.sum(torch.pow(loss_tmp_1,2),2).unsqueeze(2)+loss_tmp_2),1)
            
            del loss_tmp_1, loss_tmp_2

            # "cut" gradient backpropagation on position and zoom
            start_position = Variable(start_position.data, requires_grad=False,volatile=True)
            start_zoom = Variable(start_zoom.data, requires_grad=False,volatile=True)
        
        total_loss = loss[0]
        for j in range(1,n_max_persons):
            # ignore non-labeled persons
            loss[j][loss[j].data==0]=100000
            total_loss = torch.min(total_loss,loss[j])
        total_loss = torch.mean(total_loss)
        
        # log metrics
        Logger.log("{}_boxing_test_loss".format(model_name),body_part,(epoch+1)*n_train_batches,total_loss.data[0])
        
        accuracy = accuracy_pckh(poses,start_position.unsqueeze(1).repeat(1,n_joints,1,1).data[:,:,:,0].cpu(),head_index=head_index,neck_index=neck_index,outlier_detection=False)
        Logger.log("{}_boxing_test_acc".format(model_name),body_part,(epoch+1)*n_train_batches,accuracy[joint_index])

        hits, of = n_accurate_predictions_pckh(poses,start_position.unsqueeze(1).repeat(1,n_joints,1,1).data[:,:,:,0].cpu(),head_index=head_index,neck_index=neck_index,outlier_detection=False)
        Logger.log("{}_boxing_test_hits".format(model_name),body_part,(epoch+1)*n_train_batches,"{},{}".format(hits[joint_index],of[joint_index]))
        
        # time for testing
        t_testing += Logger.t_step()
        
        print("test step {} / {} done, t_testing = {}".format(i+1,n_test_batches,t_testing))
    
    del index, frame, camera, poses, image, image_in, start_position, real_position, exp_start_zoom, start_zoom, start_hidden, loss, total_loss
    torch.cuda.empty_cache()
    
    # save model
    Logger.save_state(model_name,'{}_{}_epoch_{}_final'.format(dataset,body_part,epoch+1),neural_net,optimizer)

Logger.save_state(model_name,'{}_{}_final'.format(dataset,body_part),neural_net,optimizer)

print("done :)")
