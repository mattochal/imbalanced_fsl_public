import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
        
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

        
class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0)  

        if outdim <=200:
            self.scale_factor = 2;
        else:
            self.scale_factor = 10; 
    
    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor* (cos_dist) 

        return scores
    

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear): # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out
    
    
class Conv2d_fw(nn.Conv2d): # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.weight.fast = None
        if bias:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out

    
class BatchNorm2d_fw(nn.BatchNorm2d): #used in MAML to forward input with fast weight
    def __init__(self, num_features, device):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None
        self.device = device

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).to(self.device)
        running_var = torch.ones(x.data.size()[1]).to(self.device)
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training = True, momentum = 1)
            #batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training = True, momentum = 1)
        return out

    
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


class ConvNet(nn.Module):
    def __init__(self, depth, device, flatten = True, maml=False, outdim = 64):
        super(ConvNet,self).__init__()
        self.num_layers = depth
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else outdim
            B = ConvBlock(indim, outdim, device, pool = ( i <4 ), maml=maml) #only pooling for fist 4 layers
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

# class ConvNetS(nn.Module): #For omniglot, only 1 input channel, output dim is 64
#     def __init__(self, depth, flatten = True,  maml=False):
#         super(ConvNetS,self).__init__()
#         self.num_layers = depth
#         trunk = []
#         for i in range(depth):
#             indim = 1 if i == 0 else 64
#             outdim = 64
#             B = ConvBlock(indim, outdim, device, pool = ( i <4 ), maml=maml) #only pooling for fist 4 layers
#             trunk.append(B)

#         if flatten:
#             trunk.append(Flatten())

#         #trunk.append(nn.BatchNorm1d(64))    #TODO remove
#         #trunk.append(nn.ReLU(inplace=True)) #TODO remove
#         #trunk.append(nn.Linear(64, 64))     #TODO remove
#         self.trunk = nn.Sequential(*trunk)
#         self.final_feat_dim = 64

#     def forward(self,x):
#         out = x[:,0:1,:,:] #only use the first dimension
#         out = self.trunk(out)
#         #out = torch.tanh(out) #TODO remove
#         return out

# class ConvNetSNopool(nn.Module): #Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
#     def __init__(self, depth):
#         super(ConvNetSNopool,self).__init__()
#         self.num_layers = depth
#         trunk = []
#         for i in range(depth):
#             indim = 1 if i == 0 else 64
#             outdim = 64
#             B = ConvBlock(indim, outdim, device, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1) #only first two layer has pooling and no padding
#             trunk.append(B)

#         self.trunk = nn.Sequential(*trunk)
#         self.final_feat_dim = [64,5,5]

#     def forward(self,x):
#         out = x[:,0:1,:,:] #only use the first dimension
#         out = self.trunk(out)
#         return out

class ResNet(nn.Module):
    def __init__(self,block,list_of_num_layers, list_of_out_dims, device, flatten = True, maml=False):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet,self).__init__()
        self.maml = maml
        self.num_layers = 4
        assert len(list_of_num_layers)==4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False).to(device)
            bn1 = BatchNorm2d_fw(64, device).to(device)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                               bias=False).to(device)
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

def Conv4(device, flatten = True,  maml=False, outdim=64):
    return ConvNet(4, device, flatten=flatten, maml=maml, outdim=outdim)

def Conv6(device, flatten = True,  maml=False, outdim=64):
    return ConvNet(6, device, flatten=flatten, maml=maml, outdim=outdim)

def Conv4NP(device,flatten = False, maml=None, outdim=64):
    return ConvNetNopool(4, device, flatten=flatten, outdim=outdim)

def Conv6NP(device,flatten = False, maml=None, outdim=64):
    return ConvNetNopool(6, device, flatten=flatten, outdim=outdim)

# def Conv4S(device, flatten = True, maml=False, outdim=None):
#     return ConvNetS(4, device, flatten=flatten, maml=maml)

# def Conv4SNP(device,flatten = True, maml=None, outdim=None):
#     return ConvNetSNopool(4, flatten=flatten, device)

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
