import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from backbones.layers import init_layer, distLinear, Linear_fw, Conv2d_fw, BatchNorm2d_fw, Flatten
from backbones.backbone_template import BackboneTemplate

    
# Simple Conv Block
class ConvBlock(nn.Module):
    def __init__(self, indim, outdim, device, pool = True, padding = 1, maml=False):
        super(ConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.maml = maml
        if self.maml: print("Maml backbone")
        if self.maml:
            self.C      = Conv2d_fw(indim, outdim, 3, padding = padding).to(device)
            self.BN     = BatchNorm2d_fw(outdim, device).to(device)
        else:
            self.C      = nn.Conv2d(indim, outdim, 3, padding= padding).to(device)
            self.BN     = nn.BatchNorm2d(outdim).to(device)
        self.relu   = nn.ReLU(inplace=True).to(device)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool   = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers).to(device)

    def forward(self,x):
        out = self.trunk(x)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, device, flatten = True, maml=False, outdim = 64):
        super(ConvNet,self).__init__()
        self.num_layers = depth
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else outdim
            B = ConvBlock(indim, outdim, device, pool = ( i < 4 ), maml=maml) #only pooling for fist 4 layers
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 5*5*outdim
        self.layer_channels = [outdim] * depth

    def forward(self,x):
        out = self.trunk(x)
        return out
    

class ConvNetNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth, device, flatten, outdim):
        super(ConvNetNopool,self).__init__()
        self.num_layers = depth
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else outdim
            outdim = outdim
            B = ConvBlock(indim, outdim, device, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1) #only first two layer has pooling and no padding
            trunk.append(B)
            
        if flatten:
            self.final_feat_dim = [outdim * 19 * 19]
            trunk.append(Flatten().to(device))
        else:
            self.final_feat_dim = [outdim,19,19]
        
        self.trunk = nn.Sequential(*trunk)

    def forward(self,x):
        out = self.trunk(x)
        return out


def Conv4(device, flatten = True,  maml=False, outdim=64):
    return ConvNet(4, device, flatten=flatten, maml=maml, outdim=outdim)

def Conv6(device, flatten = True,  maml=False, outdim=64):
    return ConvNet(6, device, flatten=flatten, maml=maml, outdim=outdim)

def Conv4NP(device,flatten = False, maml=None, outdim=64):
    return ConvNetNopool(4, device, flatten=flatten, outdim=outdim)

def Conv6NP(device,flatten = False, maml=None, outdim=64):
    return ConvNetNopool(6, device, flatten=flatten, outdim=outdim)

