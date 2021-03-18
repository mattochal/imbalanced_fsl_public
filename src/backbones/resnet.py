import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from backbones.layers import init_layer, distLinear, Linear_fw, Conv2d_fw, BatchNorm2d_fw, Flatten
from backbones.backbone_template import BackboneTemplate


# Simple ResNet Block
class SimpleBlock(nn.Module):
    def __init__(self, indim, outdim, half_res, device, maml=False):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.maml = maml
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim, device)
            self.C2 = Conv2d_fw(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = BatchNorm2d_fw(outdim, device)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim,kernel_size=3, padding=1,bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim, device)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out



# Bottleneck block
class BottleneckBlock(nn.Module):
    def __init__(self, indim, outdim, half_res, device, maml=False):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim/4)
        self.indim = indim
        self.outdim = outdim
        self.maml = maml
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1,  bias=False).to(device)
            self.BN1 = BatchNorm2d_fw(bottleneckdim, device).to(device)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1).to(device)
            self.BN2 = BatchNorm2d_fw(bottleneckdim, device).to(device)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False).to(device)
            self.BN3 = BatchNorm2d_fw(outdim, device).to(device)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1,  bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1,padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res


        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim!=outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False).to(device)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False).to(device)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)


    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out
    

class ResNet(BackboneTemplate):
    def __init__(self, block,list_of_num_layers, list_of_out_dims, device, flatten = True, maml=False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        self.maml = maml
        self.num_layers = 4
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
            bn1 = BatchNorm2d_fw(64, device).to(device)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
            bn1 = nn.BatchNorm2d(64).to(device)

        relu = nn.ReLU().to(device)
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1).to(device)

        init_layer(conv1)
        init_layer(bn1)
        
        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i>=3) and (j==0)
                B = block(indim, list_of_out_dims[i], half_res, device, maml=maml).to(device)
                trunk.append(B)
                indim = list_of_out_dims[i]
        
        if flatten:
            avgpool = nn.AvgPool2d(7).to(device)
            trunk.append(avgpool)
            trunk.append(Flatten().to(device))
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [ indim, 7, 7]

        self.trunk = nn.Sequential(*trunk).to(device)

    def forward(self,x):
        out = self.trunk(x)
        return out


def ResNet10(device, flatten = True, maml=False, outdim=None):
    return ResNet(SimpleBlock, [1,1,1,1],[64,128,256,512], device, flatten, maml=maml)

def ResNet18(device, flatten = True, maml=False, outdim=None):
    return ResNet(SimpleBlock, [2,2,2,2],[64,128,256,512], device, flatten, maml=maml)

def ResNet34(device, flatten = True, maml=False, outdim=None):
    return ResNet(SimpleBlock, [3,4,6,3],[64,128,256,512], device, flatten, maml=maml)

def ResNet50(device, flatten = True, maml=False, outdim=None):
    return ResNet(BottleneckBlock, [3,4,6,3], [256,512,1024,2048], device, flatten, maml=maml)

def ResNet101(device, flatten = True, maml=False, outdim=None):
    return ResNet(BottleneckBlock, [3,4,23,3],[256,512,1024,2048], device, flatten, maml=maml)

