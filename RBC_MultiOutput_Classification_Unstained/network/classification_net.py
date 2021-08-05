from efficientnet_pytorch import EfficientNet
import torch
import torchvision.models as models
from torch import nn


class MultiOutputModel(nn.Module):
    def __init__(self, size_classes, shape_classes, hemo_dist_classes, inclusion_classes):
        super().__init__()
        
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7', include_top = False) #(_dropout, _fc, _swish are not considered)
        self.drop = nn.Dropout(0.5)
        self.efficient_net._conv_stem = nn.Conv2d(6, 64, kernel_size = 3, stride = 2, padding = 1) # change i/p channels to 6 for merged type input
        
    
        in_features_b7 = 2560    
        self.size = nn.Sequential(
                       nn.BatchNorm1d(in_features_b7),
                       nn.Linear(in_features_b7, 512),
                       nn.ReLU(),
                       nn.BatchNorm1d(512),
                       nn.Linear(512, 128),
                       nn.ReLU(),
                       nn.BatchNorm1d(num_features=128),
                       nn.Dropout(0.4),
                       nn.Linear(128, size_classes),
                       #self.efficient_net._swish
                       )
        
        self.shape = nn.Sequential(
                       nn.BatchNorm1d(in_features_b7),
                       nn.Linear(in_features_b7, 512),
                       nn.ReLU(),
                       nn.BatchNorm1d(512),
                       nn.Linear(512, 128),
                       nn.ReLU(),
                       nn.BatchNorm1d(num_features=128),
                       nn.Dropout(0.4),
                       nn.Linear(128, shape_classes),
                       #self.efficient_net._swish
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
                       #self.efficient_net._swish
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
                       #self.efficient_net._swish,
                       nn.Sigmoid()
                       )  
        
        
    def forward(self, x):
        x = self.efficient_net(x)          # shape of x will be: (N,2560,1,1)                   
        x = torch.flatten(x, start_dim=1)
        x = self.drop(x)
        
        return {
             'size': self.size(x),
             'shape': self.shape(x),
             'hemo_dist': self.hemo_dist(x),
             'inclusion': self.inclusion(x)
             }
    
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)  
        
                               
                       
                               
           

