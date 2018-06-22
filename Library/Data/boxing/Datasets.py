"""
This file allows you to load the boxing datasets given in /cvlabdata2/cvlab/dataset_boxing/
"""

import sys
sys.path.insert(0, '/home/wandel/code')

import torch
import torch.utils.data as data
from torchvision import transforms
import pickle
import numpy as np
from scipy import misc
from PIL import Image
from Library.Data.utility import Concat
from Library.Data.utility import pixel2relative_positions

class BoxingPoseData(data.Dataset):
    """BoxingPoseData loads image and position annotations of specified directory and camera"""
    
    def __init__(self, directory, camera, transform = None, n_max_persons = 4, img_size = 1000, annotations = 'annotation_2D_cap'):
        """
        :directory: describes path to pose data and images, e.g.:
            /cvlabdata2/cvlab/dataset_boxing/OpenPose/Processed/MPI/Sequence1/
        :camera: describes which camera perspective to load from
        :annotations: what annotations to load from pickle file, e.g.:
            annotations_3D, openpose_heatmap, annotations_3D_cap, annotation_2D_cap
        """
        
        self.directory = directory
        self.annotations = annotations
        self.img_size = img_size
        if transform == None:
            self.transform = transforms.Compose([transforms.CenterCrop(self.img_size),transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transform,transforms.CenterCrop(self.img_size),transforms.ToTensor()])
        self.n_max_persons = n_max_persons
        
        # load pickle data
        data_file_name = 'fixed_data.pickle'
        self.data = pickle.load(open(self.directory+data_file_name,'rb'))['actors']
        
        self.actors = list(self.data.keys())
        
        self.camera = camera
        
        self.frames = list()
        
        for frame in sorted(self.data[self.actors[0]].keys()):
            if camera in self.data[self.actors[0]][frame].keys():
                if annotations in self.data[self.actors[0]][frame][camera].keys():
                    if annotations in self.data[self.actors[1]][frame][camera].keys():
                        self.frames.append(frame)
        return
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, index):
        """
        returns index, frame, associated poses and image data
        :image: image is returned as PIL-image (format: 1080 x 1920)
        :annotations: annotations of image; FloatTensor of size (#actors x n_joints x (depends on annotation))
        """
        camera = self.camera
        frame = self.frames[index]
        image_file_name = 'Images/cam_{}/frame_{}.jpg'.format(camera,frame)
        image = Image.open(self.directory+image_file_name)
        im_size = image.size
        ratio = self.img_size/max(im_size)
        image_transform = transforms.Compose([transforms.Resize([int(im_size[1]*ratio), int(im_size[0]*ratio)]),self.transform])
        image = image_transform(image)
        
        annotations = [self.data[actor][frame][camera][self.annotations] for actor in self.actors]
        for i in range(self.n_max_persons-len(annotations)):
            annotations.append(np.zeros([17,2]))
        annotations = annotations[0:self.n_max_persons]
        annotations = torch.from_numpy(np.asarray(annotations)).float()
        annotations = pixel2relative_positions(annotations,im_size[1],im_size[0])
        return torch.from_numpy(np.asarray([index])), frame, camera, annotations, image

class OpenPoseDataset(data.Dataset):
    """OpenPoseDataset is a class to load data from the OpenPose dataset"""
    
    def __init__(self, sequences=None, cameras=None, transform = None, n_max_persons = 4, img_size = 1000, annotations = 'annotation_2D_cap'):
        if sequences is None:
            sequences = ['1','2','4','5','8','9']
        if cameras is None:
            cameras = ['0','1','2','3','4','5','6']
        
        OpenPoseDirectory = "/cvlabdata2/cvlab/dataset_boxing/OpenPose/Processed/MPI/Sequence"
        self.datasets = list()
        for s in sequences:
            for c in cameras:
                self.datasets.append(BoxingPoseData(OpenPoseDirectory+s+'/',c, transform, n_max_persons,img_size,annotations))
        self.dataset = Concat(self.datasets)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)
    
    def __len__(self):
        return self.dataset.__len__()

class ParisDataset(data.Dataset):
    """ParisDataset is a class to load data from the Paris dataset"""
    
    def __init__(self, sequences=None, cameras=None, transform = None, n_max_persons = 4, img_size = 1000, annotations = 'annotation_2D_cap'):
        if sequences is None:
            sequences = ['1','2']
        if cameras is None:
            cameras = ['0','1','2','3','4','5','6']
        
        ParisDirectory = "/cvlabdata2/cvlab/dataset_boxing/Paris/Processed/MPI/Paris_Sequence"
        self.datasets = list()
        for s in sequences:
            for c in cameras:
                self.datasets.append(BoxingPoseData(ParisDirectory+s+'/',c,transform, n_max_persons,img_size,annotations))
        self.dataset = Concat(self.datasets)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)
    
    def __len__(self):
        return self.dataset.__len__()
