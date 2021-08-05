import glob
import os
import numpy as np
import sys
from file_loader import *
from sklearn.model_selection import train_test_split

data = "unlabelled_images_1/"
files = get_all_files(data, 'png')

train_files, val_files= train_test_split(files, test_size=0.2, random_state=2)

file_train = open("train.txt", "w")  
file_val = open("val.txt", "w")  

for file in train_files:
    file_train.write( file + "\n")  

for file in val_files:
    file_val.write( file + "\n")  


file_train.close()
file_val.close()

