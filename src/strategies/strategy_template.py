import argparse
import torch

class StrategyTemplate():

    def get_parser(parser = None):
        """
        Parser for arguments
        """
        if parser is None: parser = argparse.ArgumentParser()
        return parser
    
    def __init__(self, args, device, seed):
        self.args = args
        self.device = device
        self.seed = seed
        self.support_memory = None
        
    def reset(self):
        self.support_memory = None
        
    def update_support_set(self, support_set):  
        """
        Can be used by eg. ROS, RUS
        This rather questionable design was also designed to be compatible with Continual Few-Shot Learning
        """
        if self.support_memory is None:
            self.support_memory = support_set
        
        else:
            new_support_x, new_support_y = support_set
            support_x, support_y = self.support_memory
            support_x = torch.cat((support_x, new_support_x),0)
            support_y = torch.cat((support_y, new_support_y),0)
            self.support_memory = support_x, support_y
            
        return self.support_memory
    
    def update_support_features(self, features):
        """
        Can be used by eg. SMOTE
        """
        z, y = features
        return z, y
    
    def apply_inner_loss(self, loss_fn, *args):
        """
        Apply model's inner loop loss by default
        """
        return loss_fn(*args)
    
    def apply_outer_loss(self, loss_fn, *args):
        """
        Apply model's outer loop loss by default
        """
        return loss_fn(*args)