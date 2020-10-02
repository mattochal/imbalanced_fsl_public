# Code modified from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py 

from strategies.strategy_template import StrategyTemplate
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(StrategyTemplate):
    
    def get_parser(parser = None):
        if parser is None: parser = argparse.ArgumentParser()
        parser.add_argument('--gamma', type=int, default=2,
                            help='Adjusts the rate at which easy examples are downweighted. At gamma=0, equivalent to CE')
        parser.add_argument('--alpha', type=bool, default=False,
                            help='If False, the weighting factor alpha=1 for all classes, otherwise alpha=<inverse support class freq>')
        parser.add_argument('--size_average', type=bool, default=False,
                            help='If true, averages the loss, else sums the loss')
        return parser
    
    def __init__(self, args, device, seed):
        super(FocalLoss, self).__init__(args, device, seed)
        self.focal_loss = FocalLossFunc(args, device)
        self.inv_clas_freq = []
        
    def update_support_set(self, support):
        support = super(FocalLoss, self).update_support_set(support)
        x, y = support
        uniq, counts = torch.unique(y, return_counts=True)
        self.inv_clas_freq = 1 / counts.float().to(self.device)
        return support
    
    def apply_inner_loss(self, loss_fn, *args):
        if type(loss_fn) == nn.CrossEntropyLoss:
            return self.focal_loss.forward(*args, weights=self.inv_clas_freq)
        else:
            raise Exception("Focal Loss not compatible with {}".format(type(loss_fn)))
    
    def apply_outer_loss(self, loss_fn, *args):
        if type(loss_fn) == nn.CrossEntropyLoss:
            return self.focal_loss.forward(*args, weights=self.inv_clas_freq)
        else:
            raise Exception("Focal Loss not compatible with {}".format(type(loss_fn)))
    
    
class FocalLossFunc(nn.Module):
    
    def __init__(self, args, device):
        super(FocalLossFunc, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self, input, target, weights=None):
        gamma  = self.args.gamma
        is_alpha  = self.args.alpha
        size_average = self.args.size_average
        
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp()) #.to(self.device)

        if is_alpha:
            if weights is None:
                uniq, count = torch.unique(target, return_counts=True)
                n = input.shape[1]
                weights = torch.zeros((n,)).type_as(input.data)
                weights[uniq] = count.sum() / (n * count.type_as(input.data))
                at = weights.gather(0,target.data.view(-1))
                logpt = logpt * Variable(at) #.to(self.device)
            at = weights.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at) #.to(self.device)
            
        loss = -1 * (1-pt)**gamma * logpt
        if size_average: return loss.mean()
        else: return loss.sum()