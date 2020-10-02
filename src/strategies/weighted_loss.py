from strategies.strategy_template import StrategyTemplate
import argparse
import numpy as np
import torch
import torch
import torch.nn as nn


class WeightedLoss(StrategyTemplate):
    
    def get_parser(parser=None):
        if parser is None: parser = argparse.ArgumentParser()
        return parser
    
    def __init__(self, args, device, seed):
        super(WeightedLoss, self).__init__(args, device, seed)
        self.inv_clas_freq = []
        
    def update_support_set(self, support):
        support = super(WeightedLoss, self).update_support_set(support)
        x, y = support
        uniq, counts = torch.unique(y, return_counts=True)
        self.inv_clas_freq = 1 / counts.float().to(self.device)
        return support

    def apply_inner_loss(self, loss_fn, *args): 
        weighted_loss_fn = type(loss_fn)(weight=self.inv_clas_freq)
        return weighted_loss_fn(*args)
    
    def apply_outer_loss(self, loss_fn, *args): 
        weighted_loss_fn = type(loss_fn)(weight=self.inv_clas_freq)
        return weighted_loss_fn(*args)
