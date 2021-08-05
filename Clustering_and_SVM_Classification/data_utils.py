"""Data utility functions."""

import numpy as np
import csv

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
    