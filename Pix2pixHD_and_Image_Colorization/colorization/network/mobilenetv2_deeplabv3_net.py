import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

import network.layers as layers
import network.config as config

# create model
class MobileNetv2_DeepLabv3(nn.Module):
    """
    A Convolutional Neural Network with MobileNet v2 backbone and DeepLab v3 head
    
    """

    """######################"""
    """# Model Construction #"""
    """######################"""

    def __init__(self):
        #super(MobileNetv2_DeepLabv3, self).__init__()
        super().__init__()
        self.params = config.Params()

        # build network
        block = []

        # conv layer 1
        block.append(nn.Sequential(nn.Conv2d(1, self.params.c[0], 3, stride=self.params.s[0], padding=1, bias=False),
                                   nn.BatchNorm2d(self.params.c[0]),
                                   # nn.Dropout2d(self.params.dropout_prob, inplace=True),
                                   nn.ReLU6()))

        # conv layer 2-7
        for i in range(6):
            block.extend(layers.get_inverted_residual_block_arr(self.params.c[i], self.params.c[i+1],
                                                                t=self.params.t[i+1], s=self.params.s[i+1],
                                                                n=self.params.n[i+1]))

        # dilated conv layer 1-4
        # first dilation=rate, follows dilation=multi_grid*rate
        rate = self.params.down_sample_rate // self.params.output_stride
        block.append(layers.InvertedResidual(self.params.c[6], self.params.c[6],
                                             t=self.params.t[6], s=1, dilation=rate))
        for i in range(3):
            block.append(layers.InvertedResidual(self.params.c[6], self.params.c[6],
                                                 t=self.params.t[6], s=1, dilation=rate*self.params.multi_grid[i]))

        # ASPP layer
        block.append(layers.ASPP_plus(self.params))

        # final conv layer
        block.append(nn.Conv2d(256, self.params.num_class, 1))

        # bilinear upsample
        block.append(nn.Upsample(scale_factor=self.params.output_stride, mode='bilinear', align_corners=False))

        self.network = nn.Sequential(*block)
        # print(self.network)
        self.initialize()
        
    def initialize(self):
        """
        Initializes the model parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
    def forward(self,x):
        return self.network(x)
        
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

  