"""Data utility functions."""
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import albumentations as A
import _pickle as pickle


#SEG_LABELS_LIST = [
#    {"id": 0,  "name": "Platelet",     "rgb_values": [240, 170,  215]},
#    {"id": 0,  "name": "Background",   "rgb_values": [  0,   0,    0]},
#    {"id": 1,  "name": "Whitening_R",  "rgb_values": [ 95,  95,   95]},
#    {"id": 1,  "name": "Whitening",    "rgb_values": [255, 255,  255]},
#    {"id": 1,  "name": "Whitening2",   "rgb_values": [128, 128,  128]},
#    {"id": 1,  "name": "Whitening3",   "rgb_values": [225, 225,  225]},
#    {"id": 2,  "name": "Hemoglobin",   "rgb_values": [220,  20,   60]},
#    {"id": 2,  "name": "Hemoglobin2",  "rgb_values": [165,  42,   42]},
#    {"id": 2,  "name": "Hemoglobin3",  "rgb_values": [238, 105,   43]},
#    {"id": 3,  "name": "Nucleus",      "rgb_values": [ 70, 130,  180]},
#    {"id": 4,  "name": "Rna",          "rgb_values": [ 10,   0,  180]},
#    {"id": 5,  "name": "Reticulocyte", "rgb_values": [100, 238,  238]},
#    {"id": 6,  "name": "WBC",          "rgb_values": [150,   0,  170]}]

SEG_LABELS_LIST = [                                                        # we considered only two segmentation classes
   
    {"id": 0,  "name": "Background",   "rgb_values": [  0,   0,    0]},
    {"id": 1,  "name": "Whitening_R",  "rgb_values": [ 95,  95,   95]},
    {"id": 1,  "name": "Whitening",    "rgb_values": [255, 255,  255]},
    {"id": 1,  "name": "Whitening2",   "rgb_values": [128, 128,  128]},
    {"id": 1,  "name": "Whitening3",   "rgb_values": [225, 225,  225]},
    {"id": 1,  "name": "Hemoglobin",   "rgb_values": [220,  20,   60]},
    {"id": 1,  "name": "Hemoglobin2",  "rgb_values": [165,  42,   42]},
    
    {"id": 1,  "name": "Platelet",     "rgb_values": [240, 170,  215]},
    {"id": 1,  "name": "Nucleus",      "rgb_values": [ 70, 130,  180]},
    {"id": 1,  "name": "Rna",          "rgb_values": [ 10,   0,  180]},
    {"id": 1,  "name": "Reticulocyte", "rgb_values": [100, 238,  238]},
    {"id": 1,  "name": "WBC",          "rgb_values": [150,   0,  170]},
    {"id": 1,  "name": "Hemoglobin3",  "rgb_values": [238, 105,   43]}]

def label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)  #(H,W,C)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)



class SegmentationData(data.Dataset):

    def __init__(self, image_paths_file, transforms=None, border_detection=None):
        with open(image_paths_file) as f:
            self.image_names = f.read().splitlines()
            self.transforms = transforms
            self.border_detection = border_detection
            
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
    
    def pad_mask(self, target):      #pad first/last row/col of mask generated from pixel annotation tool with border duplicate
        pad_t = pad_b = pad_r = pad_l = 1
        skip_row = skip_col = 1
        target_rows = target_cols = target.shape[0] - skip_row
        target = target[skip_row:target_rows, 
                            skip_col:target_cols]
        cur_mask = cv2.copyMakeBorder(target, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        return cur_mask
    
    def get_item_from_index(self, index):
        
        to_tensor = transforms.Compose( [transforms.ToTensor(),
                                         transforms.Normalize((0.6958, 0.6409, 0.6662), (0.0779, 0.1226, 0.0763))])

        image_path = self.image_names[index]
        
        img = cv2.imread(image_path)   
        img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB) ## opencv reads the color channels in reverse order :(
       
        #img = cv2.copyMakeBorder(img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0) ## Zero padding to get shape 128*128*3
        
        
        if self.border_detection is None:
        
            target_path = self.image_names[index].replace('images', 'targets')
            target_path = target_path.replace('.png','_color_mask.png')
            
            target = cv2.imread(target_path)   
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB) ## opencv reads the color channels in reverse order :(
            
            if self.transforms is not None:
                transformed = self.transforms(image=img, mask=target)
                img = transformed['image']
                target = transformed['mask']
            
            img = to_tensor(img)
            
            target = self.pad_mask(target)
            
            target = np.array(target, dtype=np.int64)
            
            target_labels = target[..., 0]
            for label in SEG_LABELS_LIST:
                mask = np.all(target == label['rgb_values'], axis=2)
                target_labels[mask] = label['id']
            
            target_labels = torch.from_numpy(target_labels.copy())
            
        else:
            
            target_path = self.image_names[index].replace('images', 'borders')
            target_path = target_path.replace('.png','_color_mask_border.png')
            
            target = cv2.imread(target_path)   
            target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB) ## opencv reads the color channels in reverse order :(
            
            if self.transforms is not None:
                transformed = self.transforms(image=img, mask=target)
                img = transformed['image']
                target = transformed['mask']
            
            img = to_tensor(img)
            
            
            target = np.array(target, dtype=np.int64)
            
            target_labels = target[..., 0]
            for label in SEG_LABELS_LIST:
                mask = np.all(target == label['rgb_values'], axis=2)
                target_labels[mask] = label['id']
            
            target_labels = torch.from_numpy(target_labels.copy())            
            
        
        return img, target_labels

def get_train_transforms():
    return A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(),    # default p = 0.5
    A.Transpose(p=0.5),
    A.RandomBrightnessContrast(),  # default p = 0.5  
])

# Median frequency balancing
##### What if train split targets is missing one particular label(highly unusual), i.e.,we get 'nan' value in frequency??
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
    return class_weights
    
