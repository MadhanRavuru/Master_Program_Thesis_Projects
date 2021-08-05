import torch
import torch.nn as nn
from network.layers  import MultiResBlock, ResPath, Up_conv, Conv2_bn
from network.utils import init_weights, count_param

class MultiResUNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=6, alpha = 1.67, is_deconv=True):
        #super(UNet, self).__init__()
        super().__init__()
        self.in_channels = in_channels
        self.alpha = alpha
        self.is_deconv = is_deconv
        
        U_filters = [32, 64, 128, 256, 512]
        W_filters = [self.alpha * f for f in U_filters]
       
        W_filters = [int(W*0.167) + int(W*0.333) + int(W*0.5) for W in W_filters]  

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.mresblock1 = MultiResBlock(self.in_channels, U_filters[0])  #(in_size, out_size to be calculated from)
        self.mresblock2 = MultiResBlock(W_filters[0], U_filters[1])
        self.mresblock3 = MultiResBlock(W_filters[1], U_filters[2])
        self.mresblock4 = MultiResBlock(W_filters[2], U_filters[3])
        self.mresblock5 = MultiResBlock(W_filters[3], U_filters[4])
        
        # residual paths
        self.respath1 = ResPath(W_filters[0], U_filters[0], 4)
        self.respath2 = ResPath(W_filters[1], U_filters[1], 3)
        self.respath3 = ResPath(W_filters[2], U_filters[2], 2)
        self.respath4 = ResPath(W_filters[3], U_filters[3], 1)
        
        # upsampling
        self.up_concat4 = Up_conv(W_filters[4], U_filters[3], self.is_deconv)
        self.up_concat3 = Up_conv(W_filters[3], U_filters[2], self.is_deconv)
        self.up_concat2 = Up_conv(W_filters[2], U_filters[1], self.is_deconv)
        self.up_concat1 = Up_conv(W_filters[1], U_filters[0], self.is_deconv)
        
        self.mresblock6 = MultiResBlock(U_filters[4], U_filters[3])    #(in_size, out_size to be calculated from)
        self.mresblock7 = MultiResBlock(U_filters[3], U_filters[2])
        self.mresblock8 = MultiResBlock(U_filters[2], U_filters[1])
        self.mresblock9 = MultiResBlock(U_filters[1], U_filters[0])
        
        self.conv = Conv2_bn(W_filters[0], n_classes, False, 1, 1, 0)
        self.zeropad = nn.ZeroPad2d(4)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):         
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        inputs = self.zeropad(inputs)                  # 3*128*128
        mresblock1 = self.mresblock1(inputs)           # 51*128*128   same padding
        maxpool1 = self.maxpool(mresblock1)            # 51*64*64
        respath1 = self.respath1(mresblock1)           # 32*128*128
        
        mresblock2 = self.mresblock2(maxpool1)         # 105*64*64
        maxpool2 = self.maxpool(mresblock2)            # 105*32*32
        respath2 = self.respath2(mresblock2)           # 64*64*64
        
        mresblock3 = self.mresblock3(maxpool2)         # 212*32*32
        maxpool3 = self.maxpool(mresblock3)            # 212*16*16
        respath3 = self.respath3(mresblock3)           # 128*32*32
        
        mresblock4 = self.mresblock4(maxpool3)         # 426*16*16
        maxpool4 = self.maxpool(mresblock4)            # 426*8*8
        respath4 = self.respath4(mresblock4)           # 256*16*16
        
        mresblock5 = self.mresblock5(maxpool4)         # 853*8*8
        
        up4 = self.up_concat4(mresblock5,respath4)     # 512*16*16
        mresblock6 = self.mresblock6(up4)              # 426*16*16
        
        up3 = self.up_concat3(mresblock6,respath3)     # 256*32*32
        mresblock7 = self.mresblock7(up3)              # 212*32*32
        
        up2 = self.up_concat2(mresblock7,respath2)     # 128*64*64
        mresblock8 = self.mresblock8(up2)              # 105*64*64
        
        up1 = self.up_concat1(mresblock8,respath1)     # 64*128*128
        mresblock9 = self.mresblock9(up1)              # 51*128*128
        
        output = self.conv(mresblock9)                 # n_classes*128*128
        output = output[:,:,4:124,4:124]               # cropping to (n_classes,120,120)
        
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
    
   

      
