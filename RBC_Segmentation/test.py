import glob
import os
import numpy as np
import sys
from file_loader import *

data = "labelled_test_data/images"
files = get_all_files(data, 'png')


file_test = open("labelled_test_data/test.txt", "w") 

# Get all test data files
for file in files:
    file_test.write( file + "\n")      
    
file_test.close()
