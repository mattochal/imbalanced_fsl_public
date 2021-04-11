from models.model_template import ModelTemplate

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import argparse
import math
import tqdm
import gc


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

def unnormalised(batch, train_mean=None):
    return batch

def l2_normalised(batch, train_mean=None):
    return batch / torch.norm(batch, p=2, dim=1, keepdim=True) # normalise each feature

def centered_l2_normalised(batch, train_mean=None):
    if train_mean is None:
        train_mean = batch.mean(0)
        
    batch = batch - train_mean
    return batch / torch.norm(batch, p=2, dim=1, keepdim=True) # normalise each feature

FEATURE_TRANSFORMS = {
    "UN": unnormalised,
    "L2N": l2_normalised,
    "CL2N": centered_l2_normalised,
}

def freeze_model(model):
    for params in model.parameters():
        params.requires_grad = False
        
def unfreeze_model(model):
    for params in model.parameters():
        params.requires_grad = True
        

class SimpleShot(ModelTemplate):
    
    @staticmethod
    def get_parser(parser=None):
        if parser is None: parser = argparse.ArgumentParser(description='SimpleShot')
        """
        returns a parser for the given model. Can also return a subparser
        """
        parser = ModelTemplate.get_parser(parser)
        parser.add_argument('--feat_trans_name', type=str, choices=FEATURE_TRANSFORMS.keys(), default='CL2N',
                           help='feature transformations')
        parser.add_argument('--train_feat_trans', type=bool, default=False,
                            help='Applies feature transform during training')
        parser.add_argument('--approx_train_mean', type=bool, default=True,
                            help='Approximates the train mean using only a fraction of the dataset')
        parser.add_argument('--output_dim', type=dict, default={"train":-1, "val":-1, "test":-1},
                           help='output dimention for the classifer, if -1 set in code')
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super().__init__(backbone, strategy, args, device)
        self.output_dim = args.output_dim
        self.feat_trans_name = args.feat_trans_name
        self.train_feat_trans = args.train_feat_trans
        self.approx_train_mean = args.approx_train_mean
        self.train_mean = None
        self.update_train_mean = True
        
    def setup_model(self):
        self.classifier = nn.Linear(self.backbone.final_feat_dim, self.output_dim['train']).to(self.device)
        self.feat_transform = FEATURE_TRANSFORMS[self.feat_trans_name]
        
        all_params = list(self.backbone.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.args.lr)
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.args.lr_decay_step, 
            gamma=self.args.lr_decay)
        
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.optimizer.step()
        
    def meta_train(self, task, ptracker): # simple batch task, the same as baseline/++
        self.update_train_mean = True
        self.mode='train'
        self.train()
        
        for support_set, _ in task:
            x, y = support_set
            self.optimizer.zero_grad()
            z = self.backbone.forward(x)
            z = self.feat_transform(z) if self.train_feat_trans else z
            scores = self.classifier.forward(z)
            loss = self.loss_fn(scores, y)
            loss.backward()
            self.optimizer.step()
            
            _, preds = torch.max(scores, axis=1) 
            ptracker.add_task_performance(
                preds.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                loss.detach().cpu().numpy())
        
    def meta_eval(self, task, ptracker):  # FSL task
        self.net_reset()
        for support_set, target_set in task:
            self.net_train(support_set)
            self.net_eval(target_set, ptracker)
        
    def net_reset(self):
        self.strategy.reset()
        self.proto_memory = dict()
        
    def set_train_mean(self, dataset, istqdm=False):
        if not self.update_train_mean: return 
        self.eval()
        
        length = len(dataset)
        if self.approx_train_mean:
            length = int(length * 0.05)
        
        with torch.no_grad():
            pbar = tqdm.tqdm(initial=0, total=length, disable=(not istqdm))
            pbar.set_description("Calculating train mean")
            
            if not istqdm:
                print("Calculating train mean")

            batch_size = 64
            batch = []
            train_mean = None
            old_i = 0
            perm = np.random.permutation(len(dataset))
            
            for i in range(0, length):
                image = dataset.get_untransformed_image(perm[i])
                batch.append(image)
                
                if (i % batch_size == batch_size-1) or (i+1 == length):
                    pbar.update(batch_size)
                    batch = torch.stack(batch).to(self.device)
                    batch_h = self.backbone(batch)
                    if train_mean is None:
                        train_mean = batch_h.mean(0)
                    else:
                        train_mean = (old_i / i) * train_mean + ((i-old_i) / i) * batch_h.mean(0)
                    old_i = i
                    batch = []
        
        self.train_mean = train_mean
        self.update_train_mean = False
        pbar.close()
        
    def net_train(self, support_set):
        assert self.train_mean is not None, 'call set_train_mean before meta_eval()'
        supports_x, supports_y = self.strategy.update_support_set(support_set)
        supports_h = self.backbone(supports_x)
        supports_h, supports_y = self.strategy.update_support_features((supports_h, supports_y))
        supports_h = self.feat_transform(supports_h, self.train_mean)
        new_proto_h, new_proto_y = self.calc_prototypes(supports_h, supports_y)
        self.update_memory(new_proto_h, new_proto_y)

    def net_eval(self, target_set, ptracker):
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        
        targets_x, targets_y = target_set
        targets_h = self.backbone(targets_x)
        targets_h = self.feat_transform(targets_h, self.train_mean)
        proto_h, proto_y = self.get_prototypes()
        dist = euclidean_dist(targets_h, proto_h)
        scores = -dist
        targets_y = targets_y
        loss = self.strategy.apply_outer_loss(self.loss_fn,scores, targets_y)
        
        _, pred_y = torch.max(scores, 1)
        
        ptracker.add_task_performance(
            pred_y.detach().cpu().numpy(),
            targets_y.detach().cpu().numpy(),
            loss.detach().cpu().numpy())
        
        return loss
    
    def calc_prototypes(self, h, y):
        """
        Computes a prototypes for using array of mapped images
        :param h: tensors of mapped images (in embedding space)
        :param y: labels for the tensors
        :returns: the prototypes and their labels as tensor
        """
        unique_labels = torch.unique(y)
        proto_h = []
        for label in unique_labels:
            proto_h.append(h[y==label].mean(0))
        return proto_h, unique_labels
    
    def update_memory(self, proto_h, proto_y):
        labels = proto_y.detach().cpu().numpy()
        for i, label in enumerate(labels):
            self.proto_memory[int(label)] = proto_h[i]
    
    def get_prototypes(self, y=None):
        if y is None:
            y = np.arange(len(self.proto_memory.keys()))
        else:
            y = y.detach().cpu().numpy()
        
        proto_h = []
        for l in y:
            h = self.proto_memory[int(l)]
            proto_h.append(h)
            
        proto_h = torch.stack(proto_h, 0)
        return proto_h, y
