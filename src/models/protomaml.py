import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.maml import Maml
import copy
import argparse


class ProtoMaml(Maml):
    
    def __init__(self, backbone, strategy, args, device):
        super().__init__(backbone, strategy, args, device)
        
    def init_classifier(self, supports_x, supports_y):
        """
        Initializes the fast weights of FC layer with prototype embeddings
        """
        supports_h = self.backbone(supports_x)
        proto_h, proto_y = self.calc_prototypes(supports_h, supports_y)
        proto_h = torch.stack(proto_h, 0)[proto_y]
#         import pdb; pdb.set_trace()
        proto_h = F.normalize(proto_h, p=2, dim=1)
        self.classifier.weight.data = 2 * nn.Parameter(proto_h, requires_grad=True).to(self.device)
        self.classifier.bias.data = - nn.Parameter(torch.square(proto_h.norm(p=2,dim=1)), requires_grad=True).to(self.device)
    
    def calc_prototypes(self, h, y):
        """
        Computes prototypes
        """
        unique_labels = torch.unique(y)
        proto_h = []
        for label in unique_labels:
            proto_h.append(h[y==label].mean(0))
        return proto_h, unique_labels
    
    def net_train(self, support_set): # inner loop      
        self.zero_grad()
        
        (support_x, support_y) = self.strategy.update_support_set(support_set)
        self.init_classifier(support_x, support_y)  # difference with MAML
        
        for n_step in range(self.num_steps):
            support_h = self.backbone.forward(support_x)
            support_h, support_y = self.strategy.update_support_features((support_h, support_y))
            
            scores  = self.classifier.forward(support_h)
            set_loss = self.strategy.apply_inner_loss(self.loss_fn, scores, support_y)
            
            # build full graph support gradient of gradient
            grad = torch.autograd.grad(
                set_loss, 
                self.fast_parameters, 
                create_graph=True)
            
            if self.approx:
                grad = [ g.detach() for g in grad ] #do not calculate gradient of gradient if using first order approximation
            
            self.fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    weight.fast = weight - self.inner_loop_lr * grad[k] # create weight.fast 
                else:
                    weight.fast = weight.fast - self.inner_loop_lr * grad[k] # update weight.fast
                self.fast_parameters.append(weight.fast) # gradients are based on newest weights, but the graph will retain the link to old weight.fast