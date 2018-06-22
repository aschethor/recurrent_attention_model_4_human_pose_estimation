import os
import time
import torch


def init():
    """
    make directories needed for logging
    """
    if not os.path.exists('Logger/logs'):
        os.makedirs('Logger/logs')
    if not os.path.exists('Logger/states'):
        os.makedirs('Logger/states')


def log(name,item,index,value):
    """
    logs index value couple into csv file
    """
    filename = 'Logger/logs/{}_{}.log'.format(name,item)

    if os.path.exists(filename):
        append_write = 'a'
    else:
        append_write = 'w'

    with open(filename, append_write) as log_file:
        log_file.write("{}, {}\n".format(index,value))


t_start = 0
t_end = 0


def t_step():
    """
    returns delta t from last call of t_step()
    """
    global t_start,t_end
    t_end = time.clock()
    delta_t = t_end-t_start
    t_start = t_end
    return delta_t


def save_state(name,index,model,optimizer):
    """
    saves state of model and optimizer
    """
    torch.save(model,'Logger/states/{}_{}.mdl'.format(name,index))
    torch.save(optimizer,'Logger/states/{}_{}.opt'.format(name,index))


def load_state(name,index):
    """
    loads state of model and optimizer
    """
    model = torch.load('Logger/states/{}_{}.mdl'.format(name,index))
    optimizer = torch.load('Logger/states/{}_{}.opt'.format(name,index))
    return model,optimizer
