from efficientnet_pytorch import EfficientNet
import torch
import torchvision.models as models
from torch import nn


class PreClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7', include_top = False) #(_dropout, _fc, _swish are not considered)
        self.drop = nn.Dropout(0.5)
        self.efficient_net._conv_stem = nn.Conv2d(6, 64, kernel_size = 3, stride = 2, padding = 1)   # i/p channels = 6, merged type input
       
        in_features_b7 = 2560    
        self.output = nn.Sequential(
                       nn.BatchNorm1d(in_features_b7),
                       nn.Linear(in_features_b7, 512),
                       nn.ReLU(),
                       nn.BatchNorm1d(512),
                       nn.Linear(512, 128),
                       nn.ReLU(),
                       nn.BatchNorm1d(num_features=128),
                       nn.Dropout(0.4),
                       nn.Linear(128, num_classes),
                       nn.Sigmoid()
                       )    
        
        
    def forward(self, x):
        x = self.efficient_net(x)          # shape of x will be: (N,2560,1,1)                   
        x = torch.flatten(x, start_dim=1)
        x = self.drop(x)
        
        return self.output(x)
    
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
        
                               
                       
                               
           

