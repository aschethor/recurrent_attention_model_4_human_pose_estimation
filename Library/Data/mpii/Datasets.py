"""
This file allows you to load the mpii datasets given in /cvlabdata1/home/rhodin/datasets/mpii/
"""

import sys
sys.path.insert(0, '/home/wandel/code')

import torch
import torch.utils.data as data
from torchvision import transforms
import h5py
import numpy as np
from PIL import Image
from Library.Data.utility import pixel2relative_positions

class MPIIDataset(data.Dataset):
    """MPIIDataset loads image and position annotations of the mpii dataset"""
    
    def __init__(self, transform = None, n_max_persons = 4, img_size = 1000, train_test_valid='train'):
        """
        :transform: transformations on the images
        """
        
        self.directory = '/cvlabdata1/home/rhodin/datasets/mpii/'
        self.img_size = img_size
        if transform == None:
            self.transform = transforms.Compose([transforms.CenterCrop(self.img_size),transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transform,transforms.CenterCrop(self.img_size),transforms.ToTensor()])
        self.n_max_persons = n_max_persons
        
        # load h5 data
        if train_test_valid == 'train' or train_test_valid == 'test' or train_test_valid == 'valid':
            data = h5py.File('{}annot/{}.h5'.format(self.directory,train_test_valid),'r')
        elif train_test_valid == 'train_victor' or train_test_valid == 'valid_victor':
            data = h5py.File('/home/wandel/code/Library/Data/mpii/{}.h5'.format(train_test_valid),'r')
        
        self.frames = {}
        
        for i,img_name in enumerate(data['imgname']):
            if img_name in self.frames.keys():
                self.frames[img_name].append(data['part'][i])
            else:
                self.frames[img_name] = [data['part'][i]]
        
        return
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        """
        returns index, frame, associated poses and image data
        :image: image is returned as PIL-image (format: 1080 x 1920)
        :annotations: annotations of image; FloatTensor of size (#actors x n_joints x (depends on annotation))
        """
        # am I really taking all annotations? - yes...
        frame,annotations = list(self.frames.items())[index]
        frame = frame.decode('utf-8')
        image = Image.open('{}images/{}'.format(self.directory,frame))
        im_size = image.size
        ratio = self.img_size/max(im_size)
        image_transform = transforms.Compose([transforms.Resize([int(im_size[1]*ratio), int(im_size[0]*ratio)]),self.transform])
        image = image_transform(image)
        for i in range(self.n_max_persons-len(annotations)):
            annotations.append(np.zeros([16,2]))
        annotations = annotations[0:self.n_max_persons]
        annotations = torch.from_numpy(np.asarray(annotations)).float()
        annotations = pixel2relative_positions(annotations,im_size[1],im_size[0])
        return torch.from_numpy(np.asarray([index])), frame, 0, annotations, image
