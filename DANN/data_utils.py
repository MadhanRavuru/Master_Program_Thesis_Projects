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

class AttributesDataset():         # contains labels for all variations and mapping between sring names and IDs
    def __init__(self, data_file):
        size_labels = []
        shape_labels = []
        hemo_dist_labels = []
        inclusion_labels = []
        
        
        with open(data_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                size_labels.append(row['size'])
                shape_labels.append(row['shape'])
                hemo_dist_labels.append(row['hemo_dist'])
                inclusion_labels.append(row['inclusion'].split(', '))
           
        
                
        self.size_labels = np.unique(size_labels)
        self.shape_labels = np.unique(shape_labels)
        self.hemo_dist_labels = np.unique(hemo_dist_labels)
     
        
        inclusion_set = []
        for i in inclusion_labels:
                if i not in inclusion_set:
                    inclusion_set.append(i)
        inc_labels = {x for l in inclusion_set for x in l}
        self.inclusion_labels = sorted(list(inc_labels))
        
        
        self.num_sizes = len(self.size_labels)
        self.num_shapes = len(self.shape_labels)
        self.num_hemo_dist = len(self.hemo_dist_labels)
        self.num_inclusion = len(self.inclusion_labels)
        
        self.size_id_to_name = dict(zip(range(self.num_sizes),self.size_labels))
        self.size_name_to_id = dict(zip(self.size_labels,range(self.num_sizes)))
        
        self.shape_id_to_name = dict(zip(range(self.num_shapes),self.shape_labels))
        self.shape_name_to_id = dict(zip(self.shape_labels,range(self.num_shapes)))
        
        self.hemo_dist_id_to_name = dict(zip(range(self.num_hemo_dist),self.hemo_dist_labels))
        self.hemo_dist_name_to_id = dict(zip(self.hemo_dist_labels,range(self.num_hemo_dist)))
        

class ClassificationDataset(data.Dataset):
    
    def __init__(self, data_path_center, data_path_full, data_file, attributes, \
                 to_tensor_center, to_tensor_full, aug_transforms=None):
        
        self.aug_transforms = aug_transforms
        self.to_tensor_center = to_tensor_center
        self.to_tensor_full = to_tensor_full
        
        self.attr = attributes
        
        self.img_names = []
        
        self.size_labels = []
        self.shape_labels = []
        self.hemo_dist_labels = []
        self.inclusion_labels = []
        
        self.data_path_center = data_path_center
        self.data_path_full = data_path_full
        
        with open(data_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.img_names.append(row['image_name'])
                self.size_labels.append(self.attr.size_name_to_id[row['size']])
                self.shape_labels.append(self.attr.shape_name_to_id[row['shape']])
                self.hemo_dist_labels.append(self.attr.hemo_dist_name_to_id[row['hemo_dist']])
                
                item = row['inclusion'].split(', ')
                vector = [cls in item for cls in self.attr.inclusion_labels]
                self.inclusion_labels.append(np.array(vector, dtype=float))
                                             
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

    def get_item_from_index(self, index):
        
        if not os.path.isfile(os.path.join(self.data_path_center, self.img_names[index])):
            self.img_names[index] = self.img_names[index].replace('jai','JAI')     # DIC crop file names have 'JAI' string
        
        image_path_center = os.path.join(self.data_path_center, self.img_names[index])
        image_path_full = os.path.join(self.data_path_full, self.img_names[index])
        
        img_center = cv2.imread(image_path_center)   
        img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB) ## opencv reads the color channels in reverse order :(
        
        img_full = cv2.imread(image_path_full)   
        img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
 
        img = np.concatenate((img_center,img_full), axis=2)   # concatenation along channels
        
        if self.aug_transforms is not None:
            img = self.aug_transforms(image = img)['image']
        
        img = np.split(img, 2, axis =2)       # split transformed img into 2 parts (center and full crop)
        
        img_center = self.to_tensor_center(img[0])   
        img_full = self.to_tensor_full(img[1])
        img = torch.cat((img_center, img_full), 0)     # concatenate tensors along channels
        
        dict_data = {
                     'img_center_cell': img_center,
                     'img_full_crop': img_full,
                     'img': img,
                     'labels':{
                         'size_label':self.size_labels[index],
                         'shape_label':self.shape_labels[index],
                         'hemo_dist_label':self.hemo_dist_labels[index],
                         'inclusion_label':self.inclusion_labels[index]                     
                        }
                    }
        return dict_data

    def __len__(self):
        return len(self.img_names)

def albumentation_transforms():
    return A.Compose([
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(),
    A.Transpose(p=0.5),
    #A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    #A.Blur(blur_limit = 3, always_apply = False, p = 0.1)     
])
    
# Median frequency balancing
def compute_class_weights(train_data, attributes): 
    
    size_labels = []
    shape_labels = []
    hemo_dist_labels =[]
    inclusion_labels = []

    for dict_item in train_data:
        size_label = dict_item['labels']['size_label']               # getting all IDs of variations
        shape_label = dict_item['labels']['shape_label']
        hemo_dist_label = dict_item['labels']['hemo_dist_label']
        inclusion_label = dict_item['labels']['inclusion_label']
        
        size_labels.append(size_label)
        shape_labels.append(shape_label)
        hemo_dist_labels.append(hemo_dist_label)
        inclusion_labels.append(inclusion_label)
      
    
    size_labels_freq = np.bincount(np.array(size_labels))/len(train_data)
    size_label_weights = np.median(size_labels_freq)/size_labels_freq
    size_label_weights = dict(zip(attributes.size_labels, size_label_weights))
    
    shape_labels_freq = np.bincount(np.array(shape_labels))/len(train_data)
    shape_label_weights = np.median(shape_labels_freq)/shape_labels_freq
    shape_label_weights = dict(zip(attributes.shape_labels, shape_label_weights))
    
    hemo_dist_labels_freq = np.bincount(np.array(hemo_dist_labels))/len(train_data)
    hemo_dist_label_weights = np.median(hemo_dist_labels_freq)/hemo_dist_labels_freq
    hemo_dist_label_weights = dict(zip(attributes.hemo_dist_labels, hemo_dist_label_weights))
    
    #median freq balancing
    inclusion_label_ones_weights = len(train_data)/(2*np.sum(np.array(inclusion_labels), axis=0)) 
    inclusion_label_zeros_weights = len(train_data)/(2*(len(train_data)-np.sum(np.array(inclusion_labels), axis=0)))
    inclusion_label_ones_weights = dict(zip(attributes.inclusion_labels, inclusion_label_ones_weights))
    inclusion_label_zeros_weights = dict(zip(attributes.inclusion_labels, inclusion_label_zeros_weights))

    
    return size_label_weights, shape_label_weights, hemo_dist_label_weights, inclusion_label_ones_weights, inclusion_label_zeros_weights

