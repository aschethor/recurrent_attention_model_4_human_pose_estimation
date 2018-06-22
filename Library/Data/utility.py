"""
This file provides functionality to concatenate multiple of datasets and more...
"""

import torch
import torch.utils.data as data
from torchvision import transforms
import numpy as np

def accuracy_pckh(real_positions,pred_position,head_index=9,neck_index=8,head_size_ratio = 0.5,outlier_detection=False):
    """
    calculate pckh accuracy (predicted joints matching real joints up to half the head size)
    :real_positions: batch of real positions [batch_size x n_persons x n_joints x 2]
    :pred_position: batch of predicted positions [batch_size x n_joints x 2]
    :head_index: index of top head joint
    :neck_index: index of neck joint
    :return: accuracy of each joint [n_joints]
    """
    hits, of = n_accurate_predictions_pckh(real_positions,pred_position,head_index,neck_index,head_size_ratio,outlier_detection)
    accuracy = hits/of
    # set nan values to 0
    accuracy[accuracy!=accuracy]=0
    return accuracy

def n_accurate_predictions_pckh(real_positions,pred_position,head_index=9,neck_index=8,head_size_ratio = 0.5,outlier_detection=False):
    """
    calculate number of accurately predicted joints (predicted joints matching real joints up to half the head size)
    and number of detectable joints (the ratio of these two numbers should give a better accuracy_pckh)
    :real_positions: batch of real positions [batch_size x n_persons x n_joints x 2]
    :pred_position: batch of predicted positions [batch_size x n_joints x 2]
    :head_index: index of top head joint
    :neck_index: index of neck joint
    :outlier_detection: if True: sets number of predictable joints to 0 for predictions that don't hit a single joint within PCKh @ 0.5
    :return: number of accurately predicted joints [n_joints], number of predictable joints [n_joints]
    """
    batch_size = real_positions.shape[0]
    n_joints = real_positions.shape[2]
    pred_position = pred_position.unsqueeze(1)
    pred_dist_sq = torch.sum(torch.pow(real_positions-pred_position,2),dim=3)
    allowed_dist_sq = torch.sum(torch.pow((real_positions[:,:,head_index,:]-real_positions[:,:,neck_index,:])*head_size_ratio,2),dim=2)
    allowed_dist_sq = allowed_dist_sq.unsqueeze(2)
    allowed_outlier_dist_sq = torch.sum(torch.pow((real_positions[:,:,head_index,:]-real_positions[:,:,neck_index,:])*0.5,2),dim=2)
    allowed_outlier_dist_sq = allowed_dist_sq.unsqueeze(2)
    # find best person fit per sample in batch
    _,person = torch.max(torch.sum(pred_dist_sq<=allowed_dist_sq,dim=2),dim=1)
    n_found_joints = torch.zeros(n_joints)
    for i in range(batch_size):
        n_found_joints += (pred_dist_sq[i,person[i],:]<=allowed_dist_sq[i,person[i],:]).float()
    n_predictable_joints = torch.zeros(n_joints)
    for i in range(batch_size):
        if outlier_detection==False or torch.sum((pred_dist_sq[i,person[i],:]<=allowed_outlier_dist_sq[i,person[i],:]).float())>0:
            n_predictable_joints += (real_positions[i,person[i],:,0]>-1).float()
    return n_found_joints, n_predictable_joints

def pixel2relative_positions(pixel_positions, height, width):
    """
    takes pixel coordinates [0,height] x [0,width] and transforms it into relative coordinates [-1,1]^2 of centercropped image
    :pixel_position: tensor of size [batch_size x n_joints x 2]
    :height_width: height and width of a rectangular image
    :return: tensor of size [batch_size x n_joints x 2]
    """
    max_h_w = max(height,width)
    pixel_positions[:,:,0] = (2*pixel_positions[:,:,0]+max(height-width,0))/max_h_w-1
    pixel_positions[:,:,1] = (2*pixel_positions[:,:,1]+max(width-height,0))/max_h_w-1
    pixel_positions[((pixel_positions[:,:,0]==-1)|(pixel_positions[:,:,1]==-1)).unsqueeze(2).repeat(1,1,2)]=-1
    
    return pixel_positions

def relative2pixel_positions(pixel_position, height_width):
    """
    takes relative coordinates [-1,1]^2 and transforms it into pixel coordinates [0,height_width]^2
    :pixel_position: tensor of size [batch_size x 2]
    :height_width: height and width of a rectangular (centercropped) image
    :return: tensor of size [batch_size x 2]
    """
    return (pixel_position+1)*height_width/2

class Concat(data.Dataset):
    """Concat can be used to concatenate datasets. taken from https://discuss.pytorch.org/t/combine-concat-dataset-instances/1184"""
    
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.offsets = np.cumsum(self.lengths)
        self.length = np.sum(self.lengths)
    
    def __getitem__(self, index):
        for i, offset in enumerate(self.offsets):
            if index < offset:
                if i>0:
                    index -= self.offsets[i-1]
                return self.datasets[i][index]
        raise IndexError(f'{index} exceeds {self.length}')
    
    def __len__(self):
        return self.length

class DataTransform(data.Dataset):
    """
    deprecated - now included directly into datasets
    transforms BoxingPoseData set into wanted format
    """
    
    def __init__(self, dataset, image_transform=transforms.ToTensor(), annotation_scale=None, annotation_translate=None):
        """
        takes input from BoxingPoseData set and applies the following transformations:
        :dataset: a BoxingPoseData set to transform
        :image_transform: torchvision transform to convert PIL-data into torch tensor (and apply some augmentation techniques)
                          if None: no transformation will be applied
        :annotation_scale: torch tensor to scale annotation data
        :annotation_translate: torch tensor to translate annotation data (after scaling)
        """
        self.dataset = dataset
        self.image_transform = image_transform
        self.annotation_scale = annotation_scale
        self.annotation_translate = annotation_translate
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        index, frame, camera, pose, image = self.dataset[index]
        
        if self.annotation_scale is not None:
            pose = pose*self.annotation_scale
        if self.annotation_translate is not None:
            pose = pose+self.annotation_translate
        if self.image_transform is not None:
            image = self.image_transform(image)
        
        return index, frame, camera, pose, image

class ToGPU(data.Dataset):
    """
    deprecated - Doesn't work :( -> need to use spawn() method ?!
    moves pose and image data onto GPU -> parallelizes transfer when Dataset is loaded by mulitple workers
    """

    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        index, frame, camera, pose, image = self.dataset[index]
        pose = pose.cuda()
        image = image.cuda()
        return index, frame, camera, pose, image
