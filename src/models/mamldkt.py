import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.model_template import ModelTemplate
from backbones.layers import Linear_fw
import copy
import argparse


class MamlDKT(ModelTemplate): # TODO
    
    @staticmethod
    def get_parser(parser=None):
        """
        returns a parser for the given model. Can also return a subparser
        """
        if parser is None: parser = argparse.ArgumentParser()
        parser = ModelTemplate.get_parser(parser)
        parser.add_argument('--num_inner_loop_steps', type=int, default=5)
        parser.add_argument('--inner_loop_lr', type=float, default=0.01)
        parser.add_argument('--approx', type=bool, default=True)
        parser.add_argument('--batch_size', type=int, default=4,
                           help='number of tasks before the outerloop update, eg. update meta learner every 4th task')
        parser.add_argument('--output_dim', type=dict, default={"train":-1, "val":-1, "test":-1},
                           help='output dimention for the classifer, if -1 set in code')
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super().__init__(backbone, strategy, args, device)
        self.approx = args.approx
        self.inner_loop_lr = args.inner_loop_lr
        self.num_steps = args.num_inner_loop_steps
        self.output_dim = args.output_dim
        self.batch_size = args.batch_size
        self.batch_count = 0
        self.batch_losses = []
        self.fast_parameters = []
        assert self.output_dim.train == self.output_dim.test, 'maml training output dim must mimic the testing scenario'
        
    def setup_model(self):
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = self.setup_classifier(self.output_dim.train)
        all_params = list(self.backbone.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                step_size=self.args.lr_decay_step, gamma=self.args.lr_decay)
        self.optimizer.zero_grad()
        self.optimizer.step()
        
    def setup_classifier(self, output_dim):
        classifier = Linear_fw(self.backbone.final_feat_dim, output_dim).to(self.device)
        classifier.bias.data.fill_(0)
        return classifier
        
    def meta_train(self, task, ptracker): # single iter of meta training (outer) loop 
        self.mode='train'
        self.train()
        self.net_reset()
        self.batch_count += 1
        
        total_losses = []
        for support_set, target_set in task:
            self.net_train(support_set)
            loss = self.net_eval(target_set, ptracker)
            total_losses.append(loss)
        
        loss = torch.stack(total_losses).sum(0)
        self.batch_losses.append(loss)
        
        if self.batch_count % self.batch_size == 0:
            self.optimizer.zero_grad()
            loss = torch.stack(self.batch_losses).sum(0)
            loss.backward()
            self.optimizer.step()
            self.batch_losses = []
        
    def meta_eval(self, task, ptracker):  # single iter of evaluation of task 
        self.net_reset()
        for support_set, target_set in task:
            self.net_train(support_set)
            self.net_eval(target_set, ptracker)
     
    def net_reset(self):  
        self.strategy.reset()
        self.fast_parameters = self.get_inner_loop_params()
        for weight in self.parameters():  # reset fast parameters
            weight.fast = None
    
    def net_train(self, support_set): # inner loop       
        self.zero_grad()
        
        (support_x, support_y) = self.strategy.update_support_set(support_set)
        
        # Using gaussian processes to minimise this loop
        # TODO
        
        for n_step in range(self.num_steps):
            support_h  = self.backbone.forward(support_x)
            support_h, support_y = self.strategy.update_support_features((support_h, support_y))
            scores  = self.classifier.forward(support_h)
            set_loss = self.strategy.apply_inner_loss(self.loss_fn, scores, support_y)
            
            grad = torch.autograd.grad(
                set_loss, 
                self.fast_parameters, 
                create_graph=True) # build full graph support gradient of gradient
            
            if self.approx:
                grad = [ g.detach() for g in grad ] #do not calculate gradient of gradient if using first order approximation
            
            self.fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    weight.fast = weight - self.inner_loop_lr * grad[k] # create weight.fast 
                else:
                    weight.fast = weight.fast - self.inner_loop_lr * grad[k] # update weight.fast
                self.fast_parameters.append(weight.fast) # gradients are based on newest weights, but the graph will retain the link to old weight.fasts
                
    def net_eval(self, target_set, ptracker):
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        
        targets_x, targets_y = target_set
        targets_h  = self.backbone.forward(targets_x)
        scores  = self.classifier.forward(targets_h)
        
        loss = self.strategy.apply_outer_loss(self.loss_fn, scores, targets_y)
        
        _, pred_y = torch.max(scores, axis=1)
        
        ptracker.add_task_performance(
            pred_y.detach().cpu().numpy(),
            targets_y.detach().cpu().numpy(),
            loss.detach().cpu().numpy())
        
        return loss
    
    def get_inner_loop_params(self):
        return list(self.backbone.parameters()) + list(self.classifier.parameters())
            