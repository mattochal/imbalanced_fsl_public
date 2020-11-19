from strategies.strategy_template import StrategyTemplate
import argparse
import numpy as np
import torch
import torch
import torch.nn as nn


class CBLoss(StrategyTemplate):
    
    def get_parser(parser=None):
        if parser is None: parser = argparse.ArgumentParser()
        parser.add_argument('--beta', type=float, default=2,
                            help='Class balance loss constant between [0,1)')
        return parser
    
    def __init__(self, args, device, seed):
        super(CBLoss, self).__init__(args, device, seed)
        self.inv_clas_freq = []
        self.beta = args.beta
        
    def update_support_set(self, support):
        support = super(CBLoss, self).update_support_set(support)
        x, y = support
        uniq, counts = torch.unique(y, return_counts=True)
        n = counts.float().to(self.device)
        b = self.beta
        self.weights = (1-b)/(1-b**n)
        return support

    def apply_inner_loss(self, loss_fn, *args): 
        weighted_loss_fn = type(loss_fn)(weight=self.weights)
        return weighted_loss_fn(*args)
    
    def apply_outer_loss(self, loss_fn, *args): 
        weighted_loss_fn = type(loss_fn)(weight=self.weights)
        return weighted_loss_fn(*args)
