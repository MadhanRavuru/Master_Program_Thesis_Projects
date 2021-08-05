import glob
import os
import numpy as np
import sys
from file_loader import *
from sklearn.model_selection import train_test_split

data = "segmentation_data/images/"
files = get_all_files(data, 'png')

train_files, val_files= train_test_split(files, test_size=0.2, random_state=2)

file_train = open("segmentation_data/train.txt", "w")  
file_val = open("segmentation_data/val.txt", "w")  

# Get train and valifdation data files
for file in train_files:
    file_train.write( file + "\n")  

for file in val_files:
    file_val.write( file + "\n")  

file_train.close()
file_val.close()

