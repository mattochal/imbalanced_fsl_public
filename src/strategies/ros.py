from strategies.strategy_template import StrategyTemplate
import argparse
import numpy as np
import torch

class ROS(StrategyTemplate):
    
    def get_parser(parser = None):
        if parser is None: parser = argparse.ArgumentParser()
        return parser
    
    def __init__(self, args, device, seed):
        super(ROS, self).__init__(args, device, seed)
        self.rnd = np.random.RandomState(seed)
        
    def update_support_set(self, support):
        super(ROS, self).update_support_set(support)
        x, y = self.oversample(support)
        return x, y
    
    def oversample(self, support):
        x, y = support
        device = x.device
        
        uniq, count = torch.unique(y, return_counts=True)
        max_count = count.max().cpu().numpy()
        new_idx = []
        
        for i, cls in enumerate(uniq):
            clss_idx = torch.where(y == cls)[0].cpu().numpy()
            resampled = self.rnd.choice(clss_idx, max_count - len(clss_idx))
            new_idx.extend(clss_idx)
            new_idx.extend(resampled)
            
        if len(new_idx) == 0:
            return x, y
        
        new_idx = torch.Tensor(new_idx).long().to(device)
        new_x = x[new_idx]
        new_y = y[new_idx]
        return new_x, new_y
