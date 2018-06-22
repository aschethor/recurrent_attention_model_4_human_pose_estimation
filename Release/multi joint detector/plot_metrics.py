"""
plots loss and accuracy metrics of training and testing set
TODO: move to Logger
CODO: use plot.subplots_adjust(...)
"""

model_name = 'NeuralNet_1'
dataset = 'mpii'
n_recurrences = 10
n_joints = 16

print("""
    description:
    plot metrics of {} on {}
    """.format(model_name,dataset))

import sys
sys.path.insert(0, '/home/wandel/code')

import matplotlib
matplotlib.use('agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
Definitions = __import__("Library.Data.{}.Definitions".format(dataset), fromlist=[''])

for filtered in [True,False]:

    plt.figure(1,figsize=(24,12))
    plt.clf()
    
    # plot train loss
    plt.subplot(2,2,1)
    plt.title('train')
    plt.ylabel('loss')
    plt.xlabel('batches')
    train_loss = np.loadtxt(open("Logger/logs/{}_{}_train_loss.log".format(model_name,dataset), "rb"), delimiter=",")
    if filtered:
        train_loss = lowess(train_loss[:,1],train_loss[:,0], is_sorted=True, frac=0.025, it=0)
    plt.plot(train_loss[:,0],train_loss[:,1])
    mean = np.mean(train_loss[:,1])
    std = np.std(train_loss[:,1])
    plt.ylim([mean-2*std,mean+4*std])

    # plot train accuracies
    plt.subplot(2,2,3)
    plt.ylabel('accuracy')
    plt.xlabel('batches')
    for joint_name in Definitions.joint_names:
        train_accuracy = np.loadtxt(open("Logger/logs/{}_{}_train_acc_{}.log".format(model_name,dataset,joint_name), "rb"), delimiter=",")
        if filtered:
            train_accuracy = lowess(train_accuracy[:,1],train_accuracy[:,0], is_sorted=True, frac=0.025, it=0)
        plt.plot(train_accuracy[:,0],train_accuracy[:,1])
    plt.ylim([0,1])
    plt.legend(Definitions.joint_names,loc=2)

    # plot test loss
    plt.subplot(2,2,2)
    plt.title('test')
    plt.ylabel('loss')
    plt.xlabel('batches')
    test_loss = np.loadtxt(open("Logger/logs/{}_{}_test_loss.log".format(model_name,dataset), "rb"), delimiter=",")
    epoch_size = test_loss[0,0]
    test_size = test_loss[test_loss[:,0]==epoch_size,0].shape[0]
    test_loss_mean = []
    test_loss_std = []
    for i in range(1,int(test_loss.shape[0]/test_size)+1):
        data = test_loss[((i-1)*test_size):(i*test_size),1][:-1] # remark: crop last batch sinze it could contain a smaller amount of samples
        test_loss_mean.append([i*epoch_size,np.mean(data)])
        test_loss_std.append([i*epoch_size,np.std(data)])
    test_loss_mean = np.asarray(test_loss_mean)
    test_loss_std = np.asarray(test_loss_std)
    plt.errorbar(test_loss_mean[:,0],test_loss_mean[:,1],test_loss_std[:,1])

    # plot test accuracies
    plt.subplot(2,2,4)
    plt.ylabel('accuracy')
    plt.xlabel('batches')
    for joint_name in Definitions.joint_names:
        test_accuracy = np.loadtxt(open("Logger/logs/{}_{}_test_acc_{}.log".format(model_name,dataset,joint_name), "rb"), delimiter=",")
        test_hits = np.loadtxt(open("Logger/logs/{}_{}_test_hits_{}.log".format(model_name,dataset,joint_name), "rb"), delimiter=",")
        epoch_size = test_accuracy[0,0]
        test_size = test_accuracy[test_accuracy[:,0]==epoch_size,0].shape[0]
        test_accuracy_mean = []
        test_accuracy_std = []
        test_hit_percentage = []
        for i in range(1,int(test_accuracy.shape[0]/test_size)+1):
            data = test_accuracy[((i-1)*test_size):(i*test_size),1][:-1] # remark: crop last batch sinze it could contain a smaller amount of samples
            test_accuracy_mean.append([i*epoch_size,np.mean(data)])
            test_accuracy_std.append([i*epoch_size,np.std(data)])
            data_a = test_hits[((i-1)*test_size):(i*test_size),1][:-1]
            data_b = test_hits[((i-1)*test_size):(i*test_size),2][:-1]
            test_hit_percentage.append([i*epoch_size,np.sum(data_a)/np.sum(data_b)])
        test_accuracy_mean = np.asarray(test_accuracy_mean)
        test_accuracy_std = np.asarray(test_accuracy_std)
        test_hit_percentage = np.asarray(test_hit_percentage)
        # plt.errorbar(test_accuracy_mean[:,0],test_accuracy_mean[:,1],test_accuracy_std[:,1])
        plt.plot(test_hit_percentage[:,0],test_hit_percentage[:,1])
    plt.ylim([0,1])
    plt.legend(Definitions.joint_names,loc=2)
   
    plt.suptitle('Training metrics of {} on {} dataset'.format(model_name,dataset))
 
    # save plots
    if filtered:
        plt.savefig('Logger/plots/{}_{}_filtered.png'.format(model_name,dataset),dpi=400)
    else:
        plt.savefig('Logger/plots/{}_{}.png'.format(model_name,dataset),dpi=400)

print("done :)")
