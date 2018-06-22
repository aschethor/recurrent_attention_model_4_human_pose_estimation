"""
goal: estimate 2D joint locations in an image
"""

model_name = 'NeuralNet_1'
load_state = None #'1_epoch_89_final' #
n_recurrences = 10
n_joints = 16
n_max_persons = 4
C = 2
outlier_bound = 300

print("""
    description:
    training of {} with MPII dataset
    """.format(model_name))

import sys
sys.path.insert(0, '/home/wandel/code')

import torch
torch.manual_seed(0)
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Library.Logging import Logger
from Library.Logging.GPUtility import get_gpu_memory_map
Logger.init()
from Library.Data.mpii.Datasets import MPIIDataset
from Library.Data.mpii import Definitions
from Library.Data.utility import accuracy_pckh
from Library.Data.utility import n_accurate_predictions_pckh
NeuralNet = __import__(model_name).NeuralNet

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
    neural_net = NeuralNet(n_joints)
    neural_net = neural_net.cuda()

    # define optimizer
    optimizer = optim.Adam(neural_net.parameters())
else:
    print("load network: {}".format(load_state))
    # load neural network
    neural_net, optimizer = Logger.load_state(model_name,load_state)
    neural_net = neural_net.cuda()
    try:
        neural_net.share_weights()
    except AttributeError:
        print("weights can not be shared")

# load dataset
batch_size = 16 # 16 for Net 5 # 32 for Net 4,6 # 64 for Net 1,2,3
train_data = MPIIDataset(transform=transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.3),n_max_persons = n_max_persons,train_test_valid='train_victor')
test_data = MPIIDataset(n_max_persons = n_max_persons,train_test_valid='valid_victor')
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8, shuffle=True)

n_train_samples = len(train_data)
n_train_batches = len(train_loader)
n_test_samples = len(test_data)
n_test_batches = len(test_loader)

print("startup time: {}".format(Logger.t_step()))

# train network
outlier_detection = False

for epoch in range(200):
    print("starting with epoch: {}".format(epoch+1))
    
    if epoch+1 == 10:
        outlier_detection = True
        print("starting outlier detection")
    
    # training
    neural_net.train()
    for i,batch in enumerate(train_loader):
        
        # preprocess batch
        index, frame, camera, poses, image = batch
        t_data_load += Logger.t_step()
        
        # preprocess batch
        batch_size = int(image.shape[0])
        real_positions = []
        for j in range(n_max_persons):
            real_positions.append(Variable(poses[:,j,:,:].unsqueeze(3).float().cuda(), requires_grad=False))
        image_in = Variable(image).cuda()

        # time to move data to GPU
        t_to_gpu += Logger.t_step()

        # initialize start values
        start_zooms = Variable(0.2*(torch.rand(batch_size,1,1,1).cuda()-0.5)+0.1*(torch.rand(batch_size,n_joints,1,1).cuda()-0.5)) # Variable(torch.zeros(batch_size,n_joints,1,1).cuda())
        start_positions = Variable(0.2*(torch.zeros(batch_size,1,2,1).cuda()-0.5)+0.1*(torch.zeros(batch_size,n_joints,2,1).cuda()-0.5)) # Variable(torch.zeros(batch_size,n_joints,2,1).cuda())
        start_hidden = None
        
        # initialize loss
        loss = []
        for j in range(n_max_persons):
            loss.append(Variable(torch.zeros(batch_size,1,1)).cuda())
        
        # do several "glimpses"
        for j in range(n_recurrences):
            
            # push to neural net
            start_positions, start_zooms, start_hidden = neural_net(image_in,positions=start_positions,zooms=start_zooms,hidden=start_hidden)
        
            # loss function
            exp_start_zooms = torch.exp(start_zooms)
            
            for k in range(n_max_persons):
                # temporary variables necessary to circumvene unlabeled joints
                loss_tmp_1 = (real_positions[k]-start_positions)/exp_start_zooms
                loss_tmp_1[real_positions[k]==-1]=0
                loss_tmp_2 = start_zooms.clone()
                loss_tmp_2[real_positions[k][:,:,0:1,:]==-1]=0
                loss[k] = loss[k]+torch.sum((C*C/4*torch.sum(torch.pow(loss_tmp_1,2),2).unsqueeze(2)+loss_tmp_2),1)
            
            del loss_tmp_1, loss_tmp_2
				
            # "cut" gradient backpropagation on position and zoom
            start_positions = Variable(start_positions.data)
            start_zooms = Variable(start_zooms.data)
        
        #take person with minimal loss
        total_loss = loss[0]
        for j in range(1,n_max_persons):
            # ignore non-labeled persons -> set their loss value very high
            loss[j][loss[j].data==0]=100000
            total_loss = torch.min(total_loss,loss[j])
        if outlier_detection:
            total_loss[total_loss.data>outlier_bound]=outlier_bound
        total_loss = torch.mean(total_loss)
        
        # time for forward pass
        t_neural_net += Logger.t_step()
        
        # backprop and weight update
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # time for backpropagation
        t_backward += Logger.t_step()
        
        # log metrics
        Logger.log("{}_mpii_train".format(model_name),'loss',epoch*n_train_batches+i,total_loss.data[0])
        
        if (i%10)==0:
            accuracy = accuracy_pckh(poses,start_positions.data[:,:,:,0].cpu(),outlier_detection=outlier_detection)
            for j,joint_name in enumerate(Definitions.joint_names):
                Logger.log("{}_mpii_train_acc".format(model_name),joint_name,epoch*n_train_batches+i,accuracy[j])
            
            hits, of = n_accurate_predictions_pckh(poses,start_positions.data[:,:,:,0].cpu(),outlier_detection=outlier_detection)
            for j,joint_name in enumerate(Definitions.joint_names):
                Logger.log("{}_mpii_train_hits".format(model_name),joint_name,epoch*n_train_batches+i,"{},{}".format(hits[j],of[j]))
        
        print("train step {} / {} done, t_data_load = {}, t_to_gpu = {}, t_neural_net = {}, t_backward = {}".format(i+1,n_train_batches,t_data_load,t_to_gpu,t_neural_net,t_backward))
    
    del index, frame, camera, poses, image, image_in, start_positions, real_positions, exp_start_zooms, start_zooms, start_hidden, loss, total_loss
    torch.cuda.empty_cache()
    
    # testing
    # TODO: be careful, it seems there might be an issue with batch_norm:
    # https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/16
    neural_net.eval()
    for i,batch in enumerate(test_loader):
        
        # preprocess batch
        index, frame, camera, poses, image = batch
        
        # preprocess batch
        batch_size = int(image.shape[0])
        real_positions = []
        for j in range(n_max_persons):
            real_positions.append(Variable(poses[:,j,:,:].unsqueeze(3).float().cuda(), requires_grad=False,volatile=True))
        image_in = Variable(image, requires_grad=False,volatile=True).cuda()

        # initialize start values
        start_zooms = Variable(torch.zeros(batch_size,n_joints,1,1).cuda(), requires_grad=False,volatile=True)
        start_positions = Variable(torch.zeros(batch_size,n_joints,2,1).cuda(), requires_grad=False,volatile=True)
        start_hidden = None
        
        # initialize loss
        loss = []
        for j in range(n_max_persons):
            loss.append(Variable(torch.zeros(batch_size,1,1), requires_grad=False,volatile=True).cuda())
        
        # do several "glimpses"
        for j in range(n_recurrences):
            
            # push to neural net
            start_positions, start_zooms, start_hidden = neural_net(image_in,positions=start_positions,zooms=start_zooms,hidden=start_hidden)
        
            # loss function
            exp_start_zooms = torch.exp(start_zooms)
            
            for k in range(n_max_persons):
                # temporary variables necessary to circumvene unlabeled joints
                loss_tmp_1 = (real_positions[k]-start_positions)/exp_start_zooms
                loss_tmp_1[real_positions[k]==-1]=0
                loss_tmp_2 = start_zooms.clone()
                loss_tmp_2[real_positions[k][:,:,0:1,:]==-1]=0
                #loss[k] = loss[k]+torch.sum((torch.sum(torch.pow(loss_tmp_1,2),2).unsqueeze(2)+4*loss_tmp_2),1)
                loss[k] = loss[k]+torch.sum((C*C/4*torch.sum(torch.pow(loss_tmp_1,2),2).unsqueeze(2)+loss_tmp_2),1)
            
            del loss_tmp_1, loss_tmp_2

            # "cut" gradient backpropagation on position and zoom
            start_positions = Variable(start_positions.data, requires_grad=False,volatile=True)
            start_zooms = Variable(start_zooms.data, requires_grad=False,volatile=True)
        
        total_loss = loss[0]
        for j in range(1,n_max_persons):
            # ignore non-labeled persons
            loss[j][loss[j].data==0]=100000
            total_loss = torch.min(total_loss,loss[j])
        if outlier_detection:
            total_loss[total_loss.data>outlier_bound]=outlier_bound
        total_loss = torch.mean(total_loss)
        
        # log metrics
        Logger.log("{}_mpii_test".format(model_name),'loss',(epoch+1)*n_train_batches,total_loss.data[0])
        
        accuracy = accuracy_pckh(poses,start_positions.data[:,:,:,0].cpu(),outlier_detection=outlier_detection)
        for j,joint_name in enumerate(Definitions.joint_names):
            Logger.log("{}_mpii_test_acc".format(model_name),joint_name,(epoch+1)*n_train_batches,accuracy[j])

        hits, of = n_accurate_predictions_pckh(poses,start_positions.data[:,:,:,0].cpu(),outlier_detection=outlier_detection)
        for j,joint_name in enumerate(Definitions.joint_names):
            Logger.log("{}_mpii_test_hits".format(model_name),joint_name,(epoch+1)*n_train_batches,"{},{}".format(hits[j],of[j]))
        
        # time for testing
        t_testing += Logger.t_step()
        
        print("test step {} / {} done, t_testing = {}".format(i+1,n_test_batches,t_testing))
    
    del index, frame, camera, poses, image, image_in, start_positions, real_positions, exp_start_zooms, start_zooms, start_hidden, loss, total_loss
    torch.cuda.empty_cache()
    
    # save model
    if (epoch+1)%1==0:
        Logger.save_state(model_name,'mpii_epoch_{}_final'.format(epoch+1),neural_net,optimizer)

Logger.save_state(model_name,'mpii_final',neural_net,optimizer)

print("done :)")
