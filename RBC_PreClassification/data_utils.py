"""Data utility functions."""
import os

import matplotlib.pyplot as plt
import numpy as np
import csv
import cv2
import json
import torch
import torch.utils.data as data
from PIL import Image

import albumentations as A
import _pickle as pickle
        

class PreClassificationDataset(data.Dataset):
    
    def __init__(self, image_paths_file, to_tensor_center, to_tensor_full, aug_transforms=None):
        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()
        self.aug_transforms = aug_transforms
        self.to_tensor_center = to_tensor_center
        self.to_tensor_full = to_tensor_full
        
     
                                             
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
        
        image_path = self.image_names[index]
        
        img_full = cv2.imread(image_path)   
        img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB) ## opencv reads the color channels in reverse order :(
      
        
        image_path_good = image_path.replace('unlabelled_images_1','center_cell_good')
        if os.path.isfile(image_path_good):
            target = 0
            img_center = cv2.imread(image_path_good)   
            img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
                     
        else:
            image_path_bad = image_path_good.replace('center_cell_good','center_cell_bad')
            target = 1
            img_center = cv2.imread(image_path_bad)   
            img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB) 
        
        img = np.concatenate((img_center,img_full), axis=2)   # concatenation along channels
        
        if self.aug_transforms is not None:
            img = self.aug_transforms(image = img)['image']
        
        img = np.split(img, 2, axis =2)       # split transformed img into 2 parts (center and full crop)
        
        img_center = self.to_tensor_center(img[0]) 
        img_full = self.to_tensor_full(img[1])
        img = torch.cat((img_center, img_full), 0)     # concatenate tensors along channels
     
        return img, target

def albumentation_transforms():
    return A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(),
    A.Transpose(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.Blur(blur_limit = 3, always_apply = False, p = 0.1)     
])
    
# Median frequency balancing
def compute_class_weights(train_data): 
    
    target_labels = []
    
    for (img, target) in train_data:
        
        target_labels.append(target)
      
    
    target_labels_freq = np.bincount(np.array(target_labels))/len(train_data)
    #print(np.bincount(np.array(target_labels)))
    target_label_weights = np.median(target_labels_freq)/target_labels_freq

    return target_label_weights

