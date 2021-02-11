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


class DKT(ModelTemplate):
    
    KERNEL_TYPES = ['bncossim', 'linear', 'rbf', 'matern', 'poli1', 'poli2', 'cossim', 'bncossim']
    
    @staticmethod
    def get_parser(parser = None):
        """
        returns a parser for the given model. Can also return a subparser
        """
        parser = ModelTemplate.get_parser(parser)
        parser.add_argument('--kernel_type', type=str, choices=DKT.KERNEL_TYPES, default='bncossim',
                           help='kernel type')
        parser.add_argument('--laplace', type=bool, default=False,
                           help='use laplace approximation during evaluation')
        parser.add_argument('--output_dim', type=dict, default={"train":-1, "val":-1, "test":-1},
                           help='output dimention for the classifer, if -1 set in code')
        parser.add_argument('--gpmodel_lr', '--gp_lr', type=float, default=0.0001)
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super(DKT, self).__init__(backbone, strategy, args, device)
        self.kernel_type=self.args.kernel_type
        self.laplace=self.args.laplace
        self.output_dim = self.args.output_dim
        self.normalize = (self.kernel_type in ['cossim', 'bncossim'])
    
    def setup_model(self):
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
                {'params': self.gpmodel.parameters(), 'lr': self.args.gpmodel_lr}
        ])
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                            step_size=self.args.lr_decay_step, 
                                                            gamma=self.args.lr_decay)
    
    def meta_train(self, task, ptracker):
        """
        Trained by feeding both the query set and the support set into the model 
        """
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
            all_h, all_y = self.strategy.update_support_features((all_h, all_y))
            all_y_onehots = uu.onehot(all_y, fill_with=-1, dim=self.output_dim[self.mode])
            
            total_losses =[]
            for idx in range(self.output_dim[self.mode]):
                self.gpmodel.set_train_data(inputs=all_h, targets=all_y_onehots[:, idx], strict=False)
                output = self.gpmodel(*self.gpmodel.train_inputs)
                loss = -self.loss_fn(output, self.gpmodel.train_targets)
                total_losses.append(loss)
            
            self.optimizer.zero_grad()
            loss = torch.stack(total_losses).sum()
            loss.backward()
            self.optimizer.step()
            
            if len(target_x) > 0:
                with torch.no_grad():
                    self.gpmodel.eval()
                    self.likelihood.eval()
                    self.backbone.eval()
                    
                    target_h = self.forward(target_x)
                    
                    predictions_list = list()
                    total_losses = list()
                    for idx in range(self.output_dim[self.mode]):
                        self.gpmodel.set_train_data(
                            inputs=all_h[:support_n], 
                            targets=all_y_onehots[:support_n, idx],
                            strict=False)
                        output = self.gpmodel(all_h[support_n:])
                        total_losses.append(self.loss_fn(output, all_y_onehots[support_n:, idx]))
                        prediction = self.likelihood(output)
                        predictions_list.append(torch.sigmoid(prediction.mean))
                        
                    predictions_list = torch.stack(predictions_list).T
                    loss = -torch.stack(total_losses).sum()
                    
                    pred_y = predictions_list.argmax(1)

                    ptracker.add_task_performance(
                        pred_y.detach().cpu().numpy(),
                        target_y.detach().cpu().numpy(),
                        loss.detach().cpu().numpy())
            
        
    def forward(self, x):
        h = self.backbone.forward(x)
        if self.normalize: h = F.normalize(h, p=2, dim=1)
        return h
        
    def net_train(self, support_set):
        self.gpmodel.train()
        self.likelihood.train()
        self.backbone.eval()
        
        support_set = self.strategy.update_support_set(support_set)
        support_x, support_y = support_set
        support_h = self.forward(support_x).detach()
        support_h, support_y = self.strategy.update_support_features((support_h, support_y))
        
        self.support_y_onehots = uu.onehot(support_y, fill_with=-1, dim=self.output_dim[self.mode])
        self.support_h = support_h
    
    def net_eval(self, target_set, ptracker):
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        
        target_x, target_y = target_set
        target_y_onehots = uu.onehot(target_y, fill_with=-1, dim=self.output_dim[self.mode])
        
        with torch.no_grad():
            self.gpmodel.eval()
            self.likelihood.eval()
            self.backbone.eval()

            target_h = self.forward(target_x).detach()
            
            total_losses =[]
            predictions_list = list()
            for idx in range(self.output_dim[self.mode]):
                self.gpmodel.set_train_data(
                    inputs=self.support_h, 
                    targets=self.support_y_onehots[:, idx], 
                    strict=False)
                output = self.gpmodel(target_h)
                prediction = self.likelihood(output)
                predictions_list.append(torch.sigmoid(prediction.mean))
                loss = -self.loss_fn(output, target_y_onehots[:, idx])
                total_losses.append(loss)
                
            pred_y = torch.stack(predictions_list).argmax(0)
            loss = torch.stack(total_losses).sum(0)
            
            ptracker.add_task_performance(
                pred_y.detach().cpu().numpy(),
                target_y.detach().cpu().numpy(),
                loss.detach().cpu().numpy())
    
    
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
        likelihood.noise_covar.noise = torch.tensor(0.1)
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
    
