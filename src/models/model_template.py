import torch.nn as nn
import torch
import time
import sys
import argparse

class ModelTemplate(nn.Module):
            
    @staticmethod
    def get_parser(parser=None):
        """ 
        Return model subparser for model dependent hyperparameters / arguments found in self.args variable
        """
        if parser is None: parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=-1, help='seed, if -1 set in code')
        parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        parser.add_argument('--lr_decay', type=float, default=1.0,help='learning rate decay')
        parser.add_argument('--lr_decay_step', type=float, default=1, help='learning rate decay step size')
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super().__init__()
        self.backbone = backbone
        self.strategy = strategy
        self.args = args
        self.device = device
        self.mode = None
        self.epoch = -1
    
    def setup_model(self):
        """
        Initialises additional model parameters, setup of optimiser and lr_scheduler, etc
        """
        self.loss_fn = None
        self.optimizer = torch.optim.Adam(self.backbone.parameters(), lr=self.args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.lr_decay_step, gamma=self.args.lr_decay)
        
    def meta_train(self, task, ptracker):
        """
        Outer loop used for training
        """
        self.mode='train'
        self.train()
        
        self.net_reset()
        total_losses = []
        for support_set, target_set in task: # could be a continual learning task 
            self.net_train(support_set)
            loss = self.net_eval(target_set, ptracker)
            total_losses.append(loss)
        
        # Optimise meta-learner
        self.optimizer.zero_grad()
        loss = torch.stack(total_losses).sum(0)
        loss.backward()
        self.optimizer.step()
        self.total_losses = []
            
    def meta_test(self, task, ptracker):
        """
        Outer loop used for testing
        """
        self.mode='test'
        self.eval()
        self.meta_eval(task, ptracker)
        
    def meta_val(self, task, ptracker):
        """
        Outer loop used for validation
        """
        self.mode='val'
        self.eval()
        self.meta_eval(task, ptracker)
        
    def meta_eval(self, task, ptracker):
        """
        Outer loop used for validation and testing if they are the same
        """
        with torch.no_grad():
            self.net_reset()
            for support_set, target_set in task:
                self.net_train(support_set)
                self.net_eval(target_set, ptracker)
            self.net_post()
    
    def net_reset(self):
        """
        Used mainly by demo app.
        Network preparation set before inner loop / adaptation step
        """
        self.strategy.reset()
        
    def net_train(self, support_set):
        """
        Used mainly by demo app.
        Network innerloop / adaptation step on the support set
        """
        # don't forget to get the full support set!
        support_set = self.strategy.update_support_set(support_set)
        # ... do adaptation here 
    
    def net_eval(self, target_set, ptracker):
        """
        Used mainly by demo app.
        Network evaluation on target set after the inner loop / adaptation process
        """
        if len(target_set) == 0: return torch.tensor(0.).to(self.device)
        
        targets_x, targets_y = target_set
        pred_y = self.backbone(targets_x)
        loss = self.strategy.apply_outer_loss(self.loss_fn, pred_y, targets_y)
        
        # don't forget to store performance
        ptracker.add_task_performance(  
            pred_y.detach().cpu().numpy(),
            targets_y.detach().cpu().numpy(),
            loss.detach().cpu().numpy())
        
        return loss
    
    def net_post(self):
        """
        Used mainly by demo app.
        Used to tidy up after a task.
        eg. net_reset() could store pretrained weights temporarily, net_post() could then restore them
        """
        pass
    
    def set_mode(self, mode):
        """
        Sets the mode of algorithm ie. mode \in {'train', 'val', 'test'}
        """
        self.mode = mode
        if mode == 'train':
            self.train()
        else:
            self.eval()
        
    def set_epoch(self, epoch):
        """
        Sets the current epoch number for the meta-learning algorithm
        Updates the learning rate scheduler
        """
        self.epoch = epoch
        
    def next_epoch(self):
        self.lr_scheduler.step()
        
    def get_summary_str(self):
        """
        Returns a short string for diagonosing model, displayed in tqdm description
        """
        summary = ""
        for i, param_group in enumerate(self.optimizer.param_groups):
            summary += "lr{}={:.3E} ".format(i, param_group['lr'])
        return summary

    