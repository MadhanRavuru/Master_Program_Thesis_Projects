import torch
import torch.nn as nn
from network.utils import init_weights

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size+(n_concat-2)*out_size, out_size, False)
        if is_deconv:                                    
                self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
                self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                        nn.Conv2d(in_size, out_size, 1))           
           
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)

#############################################################
######################  MultiResUNet ########################
#############################################################
class Conv2_bn(nn.Module):
    def __init__(self, in_size, out_size, is_relu, ks=3, stride=1, padding=1):
        super(Conv2_bn, self).__init__()
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_relu:
            self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                 nn.BatchNorm2d(out_size),
                                 nn.ReLU(inplace=True),)
    
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                 nn.BatchNorm2d(out_size),)
             

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
    
        x = self.conv(x)

        return x

class Up_conv(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_odd=False):
        super(Up_conv, self).__init__()
        
        if is_deconv:
         #   if is_odd:
         #       self.up = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0),
         #                              nn.ZeroPad2d((1,0,1,0)))
         #   else:
            self.up = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0))
        else:
        #    if is_odd:
        #         self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
        #                                 nn.Conv2d(in_size, out_size, 1),
        #                                 nn.ZeroPad2d((1,0,1,0)))
        #    else:
            self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(in_size, out_size, 1))
           
        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return outputs0
        

class MultiResBlock(nn.Module):
    def __init__(self, in_size, U_filter, alpha = 1.67):
        super().__init__()
        
        self.W = alpha * U_filter
        
        self.shortcut = Conv2_bn(in_size, int(self.W*0.167) + int(self.W*0.333) + int(self.W*0.5), False, 1, 1, 0)
        self.conv3x3 = Conv2_bn(in_size, int(self.W*0.167), True)
        self.conv5x5 = Conv2_bn(int(self.W*0.167), int(self.W*0.333), True)
        self.conv7x7 = Conv2_bn(int(self.W*0.333), int(self.W*0.5), True)
        
        self.batchnorm = nn.BatchNorm2d(int(self.W*0.167) + int(self.W*0.333) + int(self.W*0.5))
        
            
    def forward(self, inp):    
        
        shortcut = inp
        
        shortcut = self.shortcut(shortcut)
        
        conv3x3 = self.conv3x3(inp)
        conv5x5 = self.conv5x5(conv3x3)
        conv7x7 = self.conv7x7(conv5x5)
        out = torch.cat([conv3x3, conv5x5, conv7x7], 1)
        out = self.batchnorm(out)
        
        out = shortcut.add(out)
        out = nn.ReLU()(out)
        out = self.batchnorm(out)
        
        return out

class ResPath(nn.Module):
    def __init__(self, in_size, out_size, length): 
        super().__init__()
        
        self.length = length
        for i in range(1, self.length+1):
            shortcut = Conv2_bn(in_size, out_size, False, 1, 1, 0)
            conv3x3 = Conv2_bn(in_size, out_size, True)
            setattr(self, 'shortcut%d'%i, shortcut)
            setattr(self, 'conv3x3_%d'%i, conv3x3)
            in_size = out_size
        
        self.batchnorm = nn.BatchNorm2d(out_size)
    
    def forward(self, inputs): 
        x = inputs
        for i in range(1, self.length+1):
            shortcut = getattr(self, 'shortcut%d'%i)
            conv3x3 = getattr(self, 'conv3x3_%d'%i)
            
            sc = shortcut(x)
            conv = conv3x3(x)
        
            out = sc.add(conv)
            out = nn.ReLU()(out)
            out = self.batchnorm(out)
            x = out
            
        return out