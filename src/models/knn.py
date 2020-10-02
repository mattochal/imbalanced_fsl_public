from models.model_template import ModelTemplate
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math
import argparse


def euclidean_dist( x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class KNN(ModelTemplate):
    
    @staticmethod
    def get_parser(parser = None):
        """
        returns a parser for the given model. Can also return a subparser
        """
        parser = ModelTemplate.get_parser(parser)
        parser.add_argument('--k', type=int, default=1,
                           help='number of neighbours used')
        parser.add_argument('--output_dim', type=dict, default={"train":-1, "test":-1},
                           help='output dimention for the classifer, if -1 set in code')
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super(KNN, self).__init__(backbone, strategy, args, device)
        self.output_dim = args.output_dim
        self.k = args.k
       
    def setup_model(self):
        self.train_classifier =  nn.Linear(self.backbone.final_feat_dim, self.output_dim['train']).to(self.device)
        self.train_classifier.bias.data.fill_(0)
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
        
    def net_train(self, support_set):
        x, y = self.strategy.update_support_set(support_set)
        z = self.backbone.forward(x)
        self.support_memory = self.strategy.update_support_features((z, y))
    
    def net_eval(self, target_set, ptracker):
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        
        target_x, target_y = target_set
        target_z = self.backbone.forward(target_x)
        
        support_z, support_y = self.support_memory
        
        dist = euclidean_dist(target_z, support_z)
        knn = dist.topk(self.k, largest=False)
        preds = torch.mode(support_y[knn.indices], 1).values
        
        ptracker.add_task_performance(
            preds.detach().cpu().numpy(), 
            target_y.detach().cpu().numpy(),
            0.0)