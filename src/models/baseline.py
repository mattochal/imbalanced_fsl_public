from models.model_template import ModelTemplate

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import argparse


class Baseline(ModelTemplate):
    
    @staticmethod
    def get_parser(parser = None):
        """
        returns a parser for the given model. Can also return a subparser
        """
        parser = ModelTemplate.get_parser(parser)
        parser.add_argument('--finetune_batch_size', type=int, default=4,
                           help='batch size used in the inner loop for finetunning')
        parser.add_argument('--finetune_iter', type=int, default=100,
                           help='number of finetuning iterations')
        parser.add_argument('--output_dim', type=dict, default={"train":-1, "test":-1},
                           help='output dimention for the classifer, if -1 set in code')
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super().__init__(backbone, strategy, args, device)
        self.output_dim = args.output_dim
        self.finetune_iter = args.finetune_iter
        self.finetune_batch_size = args.finetune_batch_size
       
    def setup_model(self):
        self.train_classifier = self.setup_classifier(self.output_dim.train)
        self.test_classifier = self.setup_classifier(self.output_dim.test)
        self.reset_test_classifier()
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        
        all_params = list(self.backbone.parameters()) + list(self.train_classifier.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                            step_size=self.args.lr_decay_step, 
                                                            gamma=self.args.lr_decay)
        
    def meta_train(self, task, ptracker):
        self.mode='train'
        self.train()
        
        for batch, _ in task:
            x, y = batch
            self.optimizer.zero_grad()
            z = self.backbone.forward(x)
            scores = self.train_classifier.forward(z)
            loss = self.loss_fn(scores, y)
            loss.backward()
            self.optimizer.step()
            
            _, preds = torch.max(scores, axis=1)
            
            ptracker.add_task_performance(
                preds.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                loss.detach().cpu().numpy())
            
    def meta_val(self, task, ptracker):
        """
        Validates just like in conventional machine learning, on different samples from same set of train classes
        Note: there's no validation loop used in the original implementation
        """
        self.mode='val'
        self.eval()
        with torch.no_grad():
            for batch, _ in task:
                x, y = batch
                self.optimizer.zero_grad()
                z = self.backbone.forward(x)
                scores = self.train_classifier.forward(z)
                loss = self.loss_fn(scores, y)
                
                _, preds = torch.max(scores, axis=1) 
                ptracker.add_task_performance(
                    preds.detach().cpu().numpy(),
                    y.detach().cpu().numpy(),
                    loss.detach().cpu().numpy())
        
        
    def meta_test(self, task, ptracker):
        """
        Test loop for FSL task
        """
        self.mode='test'
        self.eval()
        self.net_reset()
        for support_set, target_set in task:
            self.net_train(support_set)
            self.net_eval(target_set, ptracker)
        self.net_post()
        
    def net_reset(self):
        """
        Prep network for FSL task
        """
        freeze_model(self.backbone)
        self.strategy.reset()
        
    def net_train(self, support_set):
        x, y = self.strategy.update_support_set(support_set)
        z = self.backbone.forward(x)
        z, y = self.strategy.update_support_features((z, y))
        n = len(x)
        
        self.test_classifier.train()
        
        optimizer = torch.optim.SGD(
            self.test_classifier.parameters(), 
            lr=0.01, 
            momentum=0.9, 
            dampening=0.9, 
            weight_decay=0.001
        )
        
        for epoch in range(self.finetune_iter):
            rand_id = torch.from_numpy(np.random.permutation(n)).to(self.device)
            
            for i in range(0, n, self.finetune_batch_size):
                selected_id = rand_id[i: min(i+self.finetune_batch_size, n)]
                z_batch = z[selected_id]
                y_batch = y[selected_id]
                
                optimizer.zero_grad()
                self.test_classifier.zero_grad()
                scores = self.test_classifier.forward(z_batch)
                loss = self.strategy.apply_inner_loss(self.loss_fn, scores, y_batch)
                loss.backward()
                optimizer.step()
    
    def net_eval(self, target_set, ptracker):
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        self.test_classifier.eval()
        
        x, y = target_set
        z = self.backbone.forward(x)
        scores = self.test_classifier.forward(z)
        loss = self.strategy.apply_outer_loss(self.loss_fn, scores, y)
        
        _, preds = torch.max(scores, axis=1)
        
        ptracker.add_task_performance(
            preds.detach().cpu().numpy(), 
            y.detach().cpu().numpy(),
            loss.detach().cpu().numpy())
        
    def net_post(self):
        unfreeze_model(self.backbone)
        self.reset_test_classifier()
        
    def setup_classifier(self, output_dim):
        """
        Setups a regular classifer
        """
        classifier = nn.Linear(self.backbone.final_feat_dim, output_dim).to(self.device)
        classifier.bias.data.fill_(0)
        return classifier
    
    def reset_test_classifier(self):
        stdv = 1. / math.sqrt(self.test_classifier.weight.size(1))
        self.test_classifier.weight.data.uniform_(-stdv, stdv)
        self.test_classifier.bias.data.fill_(0)        

def freeze_model(model):
    for params in model.parameters():
        params.requires_grad = False
        
def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True
        