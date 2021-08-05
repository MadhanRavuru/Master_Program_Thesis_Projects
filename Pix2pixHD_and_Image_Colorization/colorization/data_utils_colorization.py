"""Data utility functions."""
import os
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
import _pickle as pickle
import albumentations as A  

class ColorizationData(data.Dataset):

    def __init__(self, image_paths_file, to_target_indices, aug_transforms=None):
        self.aug_transforms = aug_transforms
        self.to_target_indices = to_target_indices
        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()
          
            
    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.get_item_from_index(key)
        else:
            raise TypeError("Invalid argument type.")

    def __len__(self):
        return len(self.image_names)
 
    
    def get_item_from_index(self, index):
        
        image_path_bg_gray = self.image_names[index]
       
        img = Image.open(image_path_bg_gray)           #PIL image
        gray_img_bf = np.array(img)
        
    
        target_path_bf = self.image_names[index].replace('bf_gray', 'bf')
        if not os.path.isfile(target_path_bf):
            target_path_bf = target_path_bf.replace('jai', 'JAI')
        img = Image.open(target_path_bf)           #PIL image
        rgb_img_bf = np.array(img)
        lab_img_bf = rgb2lab(rgb_img_bf)
        lab_img_bf = lab_img_bf[np.newaxis,:,:,:]                 #(1,256,256,3)
        images_ab = lab_img_bf[:,:,:,1:]

        images_d = self.to_target_indices(images_ab)
        images_d = np.squeeze(images_d)                                      #(256, 256) 
        
        if self.aug_transforms is not None:
            transformed = self.aug_transforms(image=gray_img_bf, mask=images_d)
            gray_img_bf = transformed['image']
            images_d = transformed['mask']
        
        #gray_img_bf = gray_img_bf[np.newaxis,:,:]         # for training
        gray_img_bf = gray_img_bf[np.newaxis,:,:,0]        # for testing, with result from GAN
        gray_img_bf = np.float32(gray_img_bf)/255
       
  
   
        gray_img_bf = torch.from_numpy(gray_img_bf)
        
        images_d = np.array(images_d, dtype=np.int64)   
        images_d = torch.from_numpy(images_d.copy())       #bin indice starting from zero
        
        return gray_img_bf, images_d

def albumentation_transforms():
    return A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(),    # default p = 0.5
    A.Transpose(p=0.5)
   # A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.05)
])
    
# Median frequency balancing
def compute_class_weights(train_data, num_classes): 
    
    classPixelCount = np.zeros(num_classes)
    classTotalCount = np.zeros(num_classes)
    
    for k,(img, target) in enumerate(train_data):
        target = target.numpy()
        perImageFrequencies = np.bincount(target.flatten())
        perImageFrequencies.resize(classPixelCount.shape)
        #print(perImageFrequencies)
        classPixelCount = np.add(classPixelCount, perImageFrequencies)
        
        nPixelsInImage = target.shape[0]*target.shape[1]
        #print(classPixelCount)
        for i, freq in enumerate(perImageFrequencies,0):
            if freq > 0:
                classTotalCount[i] = classTotalCount[i] + nPixelsInImage
        #print(classTotalCount)
    
    frequency = classPixelCount/classTotalCount  
    #print(frequency)
    median = np.median(frequency)
    #print(median)
    class_weights = median/frequency
    #class_weights = 1.0/frequency
    return class_weights
    
