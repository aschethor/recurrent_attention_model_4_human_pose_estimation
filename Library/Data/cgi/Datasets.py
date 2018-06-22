
import torch
import torch.utils.data as data
import numpy as np

class RectangleDataset(data.Dataset):
    """RectangleData creates images of a Rectangle at a random position"""
    
    def __init__(self, n_samples, image_shape, rectangle_size, padding = 0, rectangle_color=torch.FloatTensor([1,0,0])):
        """
        :n_samples: size of generated dataset
        :image_shape: size of output images (height x width)
        :rectangle_size: length of rectangles edges (height x width)
        :padding: border to leave free at the edges
        """
        self.n_samples = n_samples
        self.image_shape = image_shape
        self.rectangle_size = rectangle_size/2
        self.padding = padding
        self.rectangle_color = rectangle_color.unsqueeze(1).unsqueeze(2)
        return
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        """
        returns:
        :index: index of item
        :frame: (0) for compatibility with BoxingDataSet
        :camera: (0) -||-
        :position: position of the center of the rectangle
        :image: images of shape (3 x height x width)
        """
        image = torch.zeros(3,self.image_shape[0],self.image_shape[1])
        position_x = np.random.randint(self.rectangle_size[1]+self.padding,self.image_shape[1]-self.rectangle_size[1]-self.padding)
        position_y = np.random.randint(self.rectangle_size[0]+self.padding,self.image_shape[0]-self.rectangle_size[0]-self.padding)
        image[:,position_y-self.rectangle_size[0]:position_y+self.rectangle_size[0],
              position_x-self.rectangle_size[1]:position_x+self.rectangle_size[1]] += self.rectangle_color
        return index, 0, 0, torch.FloatTensor([[[position_x,position_y]]]), image
        
        
