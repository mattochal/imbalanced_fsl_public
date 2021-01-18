from models.protonet import ProtoNet
import torch.nn as nn
import models.backbones as backbones 
import torch
import torch.nn.functional as F
import pprint
from torch.autograd import Variable
import utils.utils as uu
import numpy as np
import argparse
from models.model_template import ModelTemplate
import utils.utils as uu
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import gpytorch
import random
import argparse


class RelationDKT(ModelTemplate):
    
    KERNEL_TYPES = ['bncossim', 'linear', 'rbf', 'matern', 'poli1', 'poli2', 'cossim', 'bncossim']
    
    @staticmethod
    def get_parser(parser = None):
        """
        returns a parser for the given model. Can also return a subparser
        """
        parser.add_argument('--kernel_type', type=str, choices=RelationDKT.KERNEL_TYPES, default='matern',
                           help='kernel type')
        parser.add_argument('--laplace', type=bool, default=False,
                           help='use laplace approximation during evaluation')
        parser.add_argument('--output_dim', type=dict, default={"train":-1, "val":-1, "test":-1},
                           help='output dimention for the classifer, if -1 set in code')
        parser.add_argument('--gpmodel_lr', type=float, default=0.0001)
        parser.add_argument('--reduce_pair_features', type=bool, default=True)
        parser.add_argument('--normalize_feature', type=bool, default=True)
        parser.add_argument('--sigmoid_relation', type=bool, default=True)
        parser.add_argument('--normalize_relation', type=bool, default=True)
        parser = ProtoNet.get_parser(parser)
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super().__init__(backbone, strategy, args, device)
        self.reduce_pair_features = self.args.reduce_pair_features
        self.kernel_type=self.args.kernel_type
        self.laplace=self.args.laplace
        self.output_dim = self.args.output_dim
        self.normalize_feature = self.args.normalize_feature
        self.normalize_relation = self.args.normalize_relation
        self.sigmoid_relation = self.args.sigmoid_relation
        
    def setup_model(self):
        self.relation_module = RelationModule(self.backbone.final_feat_dim, 8, sigmoid=self.sigmoid_relation,
                                             normalize=self.normalize_relation).to(self.device)
        
        if self.kernel_type=="bncossim":
            latent_size = np.prod(self.backbone.final_feat_dim)
            self.backbone.trunk.add_module("bn_out", nn.BatchNorm1d(latent_size).to(self.device))
        
        train_x = torch.ones(100, 64).to(self.device)
        train_y = torch.ones(100).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.gpmodel = ExactGPLayer(train_x=train_x, train_y=train_y, likelihood=self.likelihood, 
                                    kernel=self.kernel_type).to(self.device)
        self.loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gpmodel).to(self.device)
        
        self.optimizer = torch.optim.Adam([
             {'params': self.backbone.parameters(), 'lr': self.args.lr},
             {'params': self.relation_module.parameters(), 'lr': self.args.lr},
             {'params': self.gpmodel.parameters(), 'lr': self.args.gpmodel_lr},
        ])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                            step_size=self.args.lr_decay_step, 
                                                            gamma=self.args.lr_decay)
    
#     def construct_pairs(self, proto_h, targets_h):
#         n_proto  = len(proto_h) # n_way
#         n_targets = len(targets_h) # n_query * n_way
        
#         proto_h_ext  =  proto_h.unsqueeze(0).repeat(n_targets,1,1)
#         targets_h_ext = targets_h.unsqueeze(0).repeat(n_proto,1,1)
        
#         targets_h_ext = torch.transpose(targets_h_ext,0,1)
        
#         extend_final_feat_dim = self.backbone.final_feat_dim
#         extend_final_feat_dim *= 2
#         relation_pairs = torch.cat((proto_h_ext, targets_h_ext),2).view(-1, extend_final_feat_dim)
#         if self.normalize: relation_pairs = F.normalize(relation_pairs, p=2, dim=1)
#         return relation_pairs
    
    
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
    
    
    def calc_prototypes(self, h, y):
        """
        Computes prototypes
        """
        unique_labels = torch.unique(y)
        proto_h = []
        for label in unique_labels:
            proto_h.append(h[y==label].mean(0))
        return proto_h, unique_labels
    
    def forward(self, x):
        h = self.backbone.forward(x)
        if self.normalize_feature: h = F.normalize(h, p=2, dim=1)
        return h
    
    def meta_train(self, task, ptracker):
        self.mode='train'
        self.train()
        self.net_reset()
        total_losses = []
        
        for support_set, target_set in task:
            self.backbone.train()           
            self.gpmodel.train()
            self.likelihood.train()
            
            support_set = self.strategy.update_support_set(support_set)
            support_x, support_y = support_set
            target_x, target_y = target_set
            support_n = len(support_y)
            
            # Combine target and support set
            if len(target_x) > 0:
                all_x = torch.cat((support_x, target_x), dim=0)
                all_y = torch.cat((support_y, target_y), dim=0)
            else:
                all_x = support_x
                all_y = support_y
            
            all_h = self.forward(all_x)
            all_y_onehots = uu.onehot(all_y, fill_with=-1, dim=self.output_dim[self.mode])
            
            # Construct prototypes
            proto_h, proto_y = self.calc_prototypes(all_h, all_y)
            proto_h = torch.stack(proto_h)
            
            # Construct pairs
            pairs_h = self.construct_pairs(proto_h, all_h)
            pairs_h = self.relation_module(pairs_h) if self.reduce_pair_features else pairs_h
            pairs_y_onehot = all_y_onehots.view(-1)
            
            self.gpmodel.set_train_data(inputs=pairs_h, targets=pairs_y_onehot, strict=False)
            output = self.gpmodel(*self.gpmodel.train_inputs)
            loss = -self.loss_fn(output, self.gpmodel.train_targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if len(target_x) > 0:
                with torch.no_grad():
                    self.gpmodel.eval()
                    self.likelihood.eval()
                    self.backbone.eval()

                    support_h = self.forward(support_x)
                    target_h = self.forward(target_x)
                    support_h, support_y = self.strategy.update_support_features((support_h, support_y))

                    # Construct prototypes from supports 
                    s_proto_h, s_proto_y = self.calc_prototypes(support_h, support_y)
                    s_proto_h = torch.stack(s_proto_h)
                    
                    # Construct pairs between prototypes from supports and supports
                    ss_pairs_h = self.construct_pairs(s_proto_h, support_h)
                    ss_pairs_h = self.relation_module(ss_pairs_h) if self.reduce_pair_features else ss_pairs_h
                    ss_pairs_y_onehot = uu.onehot(support_y, fill_with=-1, dim=self.output_dim[self.mode]).view(-1)
        
                    # Set train data
                    self.gpmodel.set_train_data(inputs=ss_pairs_h, targets=ss_pairs_y_onehot, strict=False)
                
                    # Construct pairs between prototypes from supports and targets
                    st_pairs_h = self.construct_pairs(s_proto_h, target_h)
                    st_pairs_h = self.relation_module(st_pairs_h) if self.reduce_pair_features else st_pairs_h
                    st_pairs_y_onehot = uu.onehot(target_y, fill_with=-1, dim=self.output_dim[self.mode]).view(-1)
                    
                    # Eval performance
                    output = self.gpmodel(st_pairs_h)
                    t_loss = -self.loss_fn(output, st_pairs_y_onehot)
                    prediction = torch.sigmoid(self.likelihood(output).mean)
                    pred_y = prediction.view(-1, self.output_dim[self.mode]).argmax(1)
#                     print(prediction.view(-1, proto_n)[:3], pred_y[:3])
#                     import pdb; pdb.set_trace()
                    
                    ptracker.add_task_performance(
                        pred_y.detach().cpu().numpy(),
                        target_y.detach().cpu().numpy(),
                        t_loss.detach().cpu().numpy())
    
    
    def net_train(self, support_set):
        self.gpmodel.eval()
        self.likelihood.eval()
        self.backbone.eval()

        support_x, support_y = support_set
        support_h = self.forward(support_x)
        support_h, support_y = self.strategy.update_support_features((support_h, support_y))

        # Construct prototypes from supports 
        s_proto_h, s_proto_y = self.calc_prototypes(support_h, support_y)
        s_proto_h = torch.stack(s_proto_h)

        # Construct pairs between prototypes from supports and supports
        ss_pairs_h = self.construct_pairs(s_proto_h, support_h)
        ss_pairs_h = self.relation_module(ss_pairs_h) if self.reduce_pair_features else ss_pairs_h
        ss_pairs_y_onehot = uu.onehot(support_y, fill_with=-1, dim=self.output_dim[self.mode]).view(-1)

        # Set train data
        self.gpmodel.set_train_data(inputs=ss_pairs_h, targets=ss_pairs_y_onehot, strict=False)
        self.s_proto_h = s_proto_h
    
    
    def net_eval(self, target_set, ptracker):
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        
        with torch.no_grad():
            self.gpmodel.eval()
            self.likelihood.eval()
            self.backbone.eval()

            target_x, target_y = target_set
            target_h = self.forward(target_x)

            # Construct pairs between prototypes from supports and targets
            st_pairs_h = self.construct_pairs(self.s_proto_h, target_h)
            st_pairs_h = self.relation_module(st_pairs_h) if self.reduce_pair_features else st_pairs_h
            st_pairs_y_onehot = uu.onehot(target_y, fill_with=-1, dim=self.output_dim[self.mode]).view(-1)
            
            # Eval performance
            output = self.gpmodel(st_pairs_h)
            t_loss = -self.loss_fn(output, st_pairs_y_onehot)
            prediction = torch.sigmoid(self.likelihood(output).mean)
            pred_y = prediction.view(-1, self.output_dim[self.mode]).argmax(1)

            ptracker.add_task_performance(
                pred_y.detach().cpu().numpy(),
                target_y.detach().cpu().numpy(),
                t_loss.detach().cpu().numpy())
    
    
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
            backbones.init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self,x):
        out = self.trunk(x)
        return out

    
class RelationModule(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, feat_dim, hidden_size, sigmoid=False, normalize=False):        
        super(RelationModule, self).__init__()
        self.sigmoid = sigmoid
        self.normalize = normalize
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
        
        if self.sigmoid:
            out = torch.sigmoid(self.fc2(out))
        else:
            out = self.fc2(out)
        
        if self.normalize:
            out = F.normalize(out, p=2, dim=1)
            
        return out
    
    
class ExactGPLayer(gpytorch.models.ExactGP):
    '''
    Parameters learned by the model:
        likelihood.noise_covar.raw_noise
        covar_module.raw_outputscale
        covar_module.base_kernel.raw_lengthscale
    '''
    def __init__(self, train_x, train_y, likelihood, kernel='linear'):
        #Set the likelihood noise and enable/disable learning
        likelihood.noise_covar.raw_noise.requires_grad = False
        likelihood.noise_covar.noise = torch.tensor(0.2)
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        ## Linear kernel
        if (kernel=='linear'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        
        ## RBF kernel
        elif (kernel=='rbf' or kernel=='RBF'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        ## Matern kernel
        elif (kernel=='matern'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        
        ## Polynomial (p=1)
        elif (kernel=='poli1'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=1))
        
        ## Polynomial (p=2) 
        elif (kernel=='poli2'):       
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2))
        
        ## Cosine distance and BatchNorm Cosine distance
        elif (kernel=='cossim' or kernel=='bncossim'):
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            self.covar_module.base_kernel.variance = 1.0
            self.covar_module.base_kernel.raw_variance.requires_grad = False            
        
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")
            
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x) 

