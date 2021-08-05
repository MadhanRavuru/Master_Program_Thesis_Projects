import torch
import torch.nn as nn
from network.layers  import unetConv2, unetUp
from network.utils import init_weights, count_param

class UNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        #super(UNet, self).__init__()
        super().__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
        
        self.output = nn.Conv2d(filters[0], n_classes, 1)
        self.zeropad = nn.ZeroPad2d(4)
        
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        inputs = self.zeropad(inputs)        # 3*128*128
        conv1 = self.conv1(inputs)           # 32*128*128   same padding
        maxpool1 = self.maxpool(conv1)       # 32*64*64
        
        conv2 = self.conv2(maxpool1)         # 64*64*64
        maxpool2 = self.maxpool(conv2)       # 64*32*32

        conv3 = self.conv3(maxpool2)         # 128*32*32
        maxpool3 = self.maxpool(conv3)       # 128*16*16

        conv4 = self.conv4(maxpool3)         # 256*16*16
        maxpool4 = self.maxpool(conv4)       # 256*8*8
        
        center = self.center(maxpool4)       # 512*8*8
        
        up4 = self.up_concat4(center,conv4)  # 256*16*16
        up3 = self.up_concat3(up4,conv3)     # 128*32*32
        up2 = self.up_concat2(up3,conv2)     # 64*64*64
        up1 = self.up_concat1(up2,conv1)     # 32*128*128
        
        output = self.output(up1)           # n_classes*128*128
        output = output[:,:,4:124,4:124]    # cropping to (n_classes,120,120)
        return output

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
    
   

      
