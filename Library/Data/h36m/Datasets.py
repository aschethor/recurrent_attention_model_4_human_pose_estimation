"""
This file allows you to load the h36m datasets given in /cvlabdata2/cvlab/dataset_boxing/
Furthermore it provides functionality to concatenate multiple of these datasets and 
transform the image / annotations to fit the neural network using it
"""

import torch
import torch.utils.data as data
from torchvision import transforms
import pickle
import numpy as np
from scipy import misc
from PIL import Image
from Library.Data.utility import pixel2relative_positions

class H36MDataset(data.Dataset):
    """H36M data loads image and position data of h36m dataset"""
    
    def __init__(self,transform = None, n_max_persons = 4, img_size = 1000, annotations='annotations_2d',topics = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'Walking', 'WalkingDog', 'WalkTogether'],testing=False):
        
        if testing:
            self.data = pickle.load(open('/cvlabdata2/cvlab/Human36m/OpenPose/val_data.pkl','rb'))
        else:
            self.data = pickle.load(open('/cvlabdata2/cvlab/Human36m/OpenPose/train_data.pkl','rb'))
        
        self.img_size = img_size
        if transform == None:
            self.transform = transforms.Compose([transforms.CenterCrop(self.img_size),transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transform,transforms.CenterCrop(self.img_size),transforms.ToTensor()])
        self.n_max_persons = n_max_persons
        
        self.annotations = annotations
        
        self.frames = list()
        
        for topic in topics:
            for dataset in self.data[topic].keys():
                for subset in self.data[topic][dataset].keys():
                    for frame in self.data[topic][dataset][subset].keys():
                        if annotations in self.data[topic][dataset][subset][frame].keys():
                            self.frames.append([topic, dataset, subset, frame])
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self,index):
        topic, dataset, subset, frame = self.frames[index]
        
        annotations = self.data[topic][dataset][subset][frame][self.annotations]
        
        image_file_name = '/cvlabdata2/cvlab/Human36m/OpenPose/S{}/Images/{}_000000{}.jpg'.format(dataset,subset,frame)
        image = Image.open(image_file_name)
        im_size = image.size
        ratio = self.img_size/max(im_size)
        image_transform = transforms.Compose([transforms.Resize([int(im_size[1]*ratio), int(im_size[0]*ratio)]),self.transform])
        image = image_transform(image)
        
        annotations = [annotations]
        for i in range(self.n_max_persons-len(annotations)):
            annotations.append(np.zeros([15,2]))
        annotations = annotations[0:self.n_max_persons]
        
        annotations = torch.from_numpy(np.asarray(annotations)).float()
        annotations = pixel2relative_positions(annotations,im_size[1],im_size[0])
        
        return torch.from_numpy(np.asarray([index])), frame, 0, annotations, image
