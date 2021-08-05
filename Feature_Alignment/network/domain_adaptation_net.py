import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
    

class Feature_extractor(nn.Module):

    def __init__(self):
        super().__init__() 
        #self.modules= list(models.resnet50(pretrained=True).children())[:-1]
        #self.feature_layers = nn.Sequential(*self.modules)
        self.feature_layers = EfficientNet.from_pretrained('efficientnet-b7', include_top = False) 
        self.drop = nn.Dropout(0.5)
       
    def forward(self, x):
        x = self.feature_layers(x)          # shape of x will be: (N,2560,1,1)   
        x = torch.flatten(x, start_dim=1) 
        #x = self.drop(x)
        return x    
    
class Class_classifier(nn.Module):

    def __init__(self, size_classes, shape_classes, hemo_dist_classes, inclusion_classes):
        super().__init__()
        
        in_features_b7 = 2560         # features from feature extractor  
        self.size = nn.Sequential(
                       nn.BatchNorm1d(in_features_b7),
                       nn.Linear(in_features_b7, 512),
                       nn.ReLU(),
                       nn.BatchNorm1d(512),
                       nn.Linear(512, 128),
                       nn.ReLU(),
                       nn.BatchNorm1d(num_features=128),
                       nn.Dropout(0.4),
                       nn.Linear(128, size_classes)                      
                       )
        
        self.shape = nn.Sequential(
                       nn.Linear(in_features_b7, shape_classes),
                       )

        self.hemo_dist = nn.Sequential(
                       nn.BatchNorm1d(in_features_b7),
                       nn.Linear(in_features_b7, 512),
                       nn.ReLU(),
                       nn.BatchNorm1d(512),
                       nn.Linear(512, 128),
                       nn.ReLU(),
                       nn.BatchNorm1d(num_features=128),
                       nn.Dropout(0.4),
                       nn.Linear(128, hemo_dist_classes),
                       )
            
        self.inclusion = nn.Sequential(
                       nn.BatchNorm1d(in_features_b7),
                       nn.Linear(in_features_b7, in_features_b7),
                       nn.ReLU(),
                       nn.BatchNorm1d(in_features_b7),
                       nn.Linear(in_features_b7, 512),
                       nn.ReLU(),
                       nn.BatchNorm1d(512),
                       nn.Linear(512, 128),
                       nn.ReLU(),
                       nn.BatchNorm1d(num_features=128),
                       nn.Dropout(0.4),
                       nn.Linear(128, inclusion_classes),
                       nn.Sigmoid()
                       )

    def forward(self, x):
          
        return {
             'size': self.size(x),
             'shape': self.shape(x),
             'hemo_dist': self.hemo_dist(x),
             'inclusion': self.inclusion(x)
             }





