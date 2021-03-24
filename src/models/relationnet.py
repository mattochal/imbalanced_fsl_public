from models.protonet import ProtoNet
from backbones.layers import init_layer

import torch.nn as nn
import torch
import torch.nn.functional as F
import pprint
from torch.autograd import Variable
import utils.utils as uu
import numpy as np
import argparse


class RelationNet(ProtoNet):
    
    @staticmethod
    def get_parser(parser=None):
        if parser is None: parser = argparse.ArgumentParser(description='RelationNet')
        parser = ProtoNet.get_parser(parser)
        parser.add_argument('--loss_type', type=str, choices=['mse', 'softmax'], default='mse')
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super().__init__(backbone, strategy, args, device)
        self.loss_type = self.args.loss_type
        
    def setup_model(self):
        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == 'softmax':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise Exception("Invalid loss type: {}".format(self.loss_type))
            
        self.relation_module = RelationModule(self.backbone.final_feat_dim, 8, self.loss_type).to(self.device)
        all_params = list(self.backbone.parameters()) + list(self.relation_module.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                step_size=self.args.lr_decay_step, gamma=self.args.lr_decay)
        
    def meta_train(self, task, ptracker): # single iter of meta training (outer) loop 
        self.mode='train'
        self.train()
        
        self.net_reset()
        total_losses = []
        for support_set, target_set in task:
            self.net_train(support_set)  # in protonet
            loss = self.net_eval(target_set, ptracker)
            total_losses.append(loss)
        
        self.optimizer.zero_grad()
        loss = torch.sum(torch.stack(total_losses))
        loss.backward()
        self.optimizer.step()
            
    def net_eval(self, target_set, ptracker):
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        
        targets_x, targets_y = target_set
        targets_h = self.backbone(targets_x)
        proto_h, proto_y = self.get_prototypes()
        
        relation_pairs = self.construct_pairs(proto_h, targets_h)
        
        n_way = len(proto_y)
        scores = self.relation_module(relation_pairs)
        scores = scores.view(-1, n_way)
        
        if self.loss_type == 'mse':
            targets_y_onehot = Variable(uu.onehot(targets_y).float().to(self.device))
            loss = self.strategy.apply_outer_loss(self.loss_fn, scores, targets_y_onehot)
        else:
            loss = self.strategy.apply_outer_loss(self.loss_fn, scores, targets_y)
        
        _, pred_y = torch.max(scores, 1)
        
        ptracker.add_task_performance(
            pred_y.detach().cpu().numpy(),
            targets_y.detach().cpu().numpy(),
            loss.detach().cpu().numpy())
        
        return loss
    
    def construct_pairs(self, proto_h, targets_h):
        proto_h  =  proto_h.view(-1, *self.backbone.final_feat_dim)
        targets_h = targets_h.view(-1, *self.backbone.final_feat_dim)
        
        n_proto  = len(proto_h) # n_way
        n_targets = len(targets_h) # n_query * n_way
        
        proto_h_ext  =  proto_h.unsqueeze(0).repeat(n_targets,1,1,1,1)
        targets_h_ext = targets_h.unsqueeze(0).repeat(n_proto,1,1,1,1)
        
        targets_h_ext = torch.transpose(targets_h_ext,0,1)
        
        extend_final_feat_dim = self.backbone.final_feat_dim.copy()
        extend_final_feat_dim[0] *= 2
        relation_pairs = torch.cat((proto_h_ext, targets_h_ext),2).view(-1, *extend_final_feat_dim)
        return relation_pairs
    
    
class RelationConvBlock(nn.Module):
    def __init__(self, indim, outdim, padding = 0):
        super(RelationConvBlock, self).__init__()
        self.indim  = indim
        self.outdim = outdim
        self.C      = nn.Conv2d(indim, outdim, 3, padding = padding )
        self.BN     = nn.BatchNorm2d(outdim, momentum=1, affine=True)
        self.relu   = nn.ReLU()
        self.pool   = nn.MaxPool2d(2)

        self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self,x):
        out = self.trunk(x)
        return out

    
class RelationModule(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, feat_dim, hidden_size, loss_type='mse'):        
        super(RelationModule, self).__init__()
        self.loss_type = loss_type
        padding = 1 if ( feat_dim[1] <10 ) and ( feat_dim[2] <10 ) else 0 # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling

        self.layer1 = RelationConvBlock(feat_dim[0]*2, feat_dim[0], padding = padding )
        self.layer2 = RelationConvBlock(feat_dim[0], feat_dim[0], padding = padding )

        shrink_s = lambda s: int((int((s- 2 + 2*padding)/2)-2 + 2*padding)/2)

        self.fc1 = nn.Linear( feat_dim[0]* shrink_s(feat_dim[1]) * shrink_s(feat_dim[2]), hidden_size )
        self.fc2 = nn.Linear( hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        
        if self.loss_type == 'mse':
            out = torch.sigmoid(self.fc2(out))
        elif self.loss_type == 'softmax':
            out = self.fc2(out)
            
        return out