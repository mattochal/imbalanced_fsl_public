import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.model_template import ModelTemplate
from backbones.layers import Linear_fw
import utils.utils as utils
import copy
import argparse


class BayesianTAML(ModelTemplate):
    
    @staticmethod
    def get_parser(parser=None):
        """
        returns a parser for the given model. Can also return a subparser
        """
        if parser is None: parser = argparse.ArgumentParser()
        parser = ModelTemplate.get_parser(parser)
        parser.add_argument('--num_inner_loop_steps', type=dict, default={"train":5, "val":10, "test":10})
        parser.add_argument('--inner_loop_lr', type=float, default=0.01)
        parser.add_argument('--approx', type=utils.str2bool, default=False)
        parser.add_argument('--approx_until', type=int, default=0,
                           help='approx until the specified epoch to expediate training')
        parser.add_argument('--batch_size', type=int, default=4,
                           help='number of tasks before the outerloop update, eg. update meta learner every 4th task')
        parser.add_argument('--output_dim', type=dict, default={"train":-1, "val":-1, "test":-1},
                           help='output dimention for the classifer, if -1 set in code')
        parser.add_argument('--omega_on', type=utils.str2bool, default=True)
        parser.add_argument('--gamma_on', type=utils.str2bool, default=True)
        parser.add_argument('--alpha_on', type=utils.str2bool, default=True)
        parser.add_argument('--z_on', type=utils.str2bool, default=True)
        parser.add_argument('--with_sampling', type=utils.str2bool, default=True)
        parser.add_argument('--num_draws', type=dict, default={"train":1, "val":1, "test":10})
        parser.add_argument('--max_shot', type=dict, default=-1, help='Max shot (if -1, set in code)')
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super(BayesianTAML, self).__init__(backbone, strategy, args, device)
        self.approx = args.approx
        self.approx_until = args.approx_until
        self.inner_loop_lr = args.inner_loop_lr
        self.num_steps = args.num_inner_loop_steps
        self.with_sampling = args.with_sampling
        self.output_dim = args.output_dim
        self.batch_size = args.batch_size
        self.num_draws = args.num_draws
        self.omega_on = args.omega_on
        self.gamma_on = args.gamma_on
        self.alpha_on = args.alpha_on
        self.z_on = args.z_on
        self.batch_count = 0
        self.batch_losses = []
        self.fast_parameters = []
        assert self.output_dim.train == self.output_dim.test, 'maml training output dim must mimic the testing scenario'
        
    def setup_model(self):
        self.loss_fn = nn.CrossEntropyLoss(reduce=False)
        self.classifier = self.setup_classifier(self.output_dim.train)
        self.inference_network = InferenceNetwork(self.backbone.num_layers, self.backbone.layer_channels, 
                                                  self.args, self.device).to(self.device)
        self.all_params = list(self.backbone.parameters()) + \
                          list(self.classifier.parameters()) + \
                          list(self.inference_network.parameters())
        if self.alpha_on:
            self.alpha = self.get_alpha(self.get_inner_loop_named_params())
            self.all_params += list(self.alpha.values())
            
        self.optimizer = torch.optim.Adam(self.all_params, lr=self.args.lr)
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
        self.inference_network.train()
        self.net_reset()
        self.zero_grad()
        self.batch_count += 1
        self.approx = self.approx if self.epoch < self.approx_until else False
        
        num_draws = self.num_draws[self.mode]
        
        total_losses = []
        for support_set, target_set in task:
            support_set = self.strategy.update_support_set(support_set)
            support_x, support_y = support_set
            targets_x, targets_y = target_set
            kl_losses = 0.
            sample_scores = 0.
            losses = 0.
            kl_scaling = 1./(len(support_y)+len(targets_y))
            for n_sample in range(num_draws):
                self.net_reset()
                kl_loss = self.net_train(support_set)
                loss, scores = self.net_eval(target_set, ptracker)
                sample_scores += torch.softmax(scores, -1)
                losses += loss
                kl_losses += kl_loss * kl_scaling
            
            losses /= num_draws
            kl_losses /= num_draws
            sample_scores /= num_draws
            
            total_losses.append(losses+0.1*kl_losses)
            
            _, pred_y = torch.max(sample_scores, axis=1)
            
            ptracker.add_task_performance(
                pred_y.detach().cpu().numpy(),
                targets_y.detach().cpu().numpy(),
                losses.detach().cpu().numpy())
        
        loss = torch.stack(total_losses).sum(0)
        self.batch_losses.append(loss)
        
        if self.batch_count % self.batch_size == 0:
            self.optimizer.zero_grad()
            loss = torch.stack(self.batch_losses).sum(0)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.all_params, 3)
            self.optimizer.step()
            self.batch_losses = []
        
    def meta_eval(self, task, ptracker):  # single iter of evaluation of task 
        num_draws = self.num_draws[self.mode]
        self.eval()
        self.inference_network.eval()
        self.net_reset()
        
        for support_set, target_set in task:
            support_set = self.strategy.update_support_set(support_set)
            targets_x, targets_y = target_set
            sample_scores = 0.
            losses = 0.
            
            for n_sample in range(num_draws):
                self.net_reset()
                self.net_train(support_set)
                loss,scores = self.net_eval(target_set, ptracker)
                sample_scores += torch.softmax(scores, -1).detach().cpu()
                losses += loss.detach().cpu()
                del loss
                del scores
    
            losses /= num_draws
            sample_scores /= num_draws
            _, pred_y = torch.max(sample_scores, axis=1)
            
            ptracker.add_task_performance(
                pred_y.numpy(),
                targets_y.detach().cpu().numpy(),
                losses.numpy())
     
    def net_reset(self):  
        self.strategy.reset()
        self.fast_parameters = self.get_inner_loop_params()
        for weight in self.fast_parameters:  # reset fast parameters
            weight.fast = None
    
    def net_train(self, support_set): # inner loop  
        (support_x, support_y) = support_set
        uniq_y = support_y.unique()
        
        with_sampling = (self.with_sampling and self.mode != 'val')
        omega, gamma, z, kl = self.inference_network((support_x, support_y), with_sampling=with_sampling)
        
        if self.z_on:
            for i, named_weight in enumerate(self.backbone.named_parameters()):
                name, weight = named_weight
                layer_id = [int(s) for s in name.split('.') if s.isdigit()][0]
                if "C.weight" in name:
                    weight = weight * (1 + z['w'][layer_id].view(-1,1,1,1))
                elif "C.bias" in name:
                    weight = weight + z['b'][layer_id]
        
        for n_step in range(self.num_steps[self.mode]):
            support_h  = self.backbone.forward(support_x)
            scores  = self.classifier.forward(support_h)
            losses = self.loss_fn(scores, support_y)
            
            if self.omega_on:
                set_loss = 0.  # inner loss
                scaling = len(uniq_y) / len(support_y)  # number of classes per sample 
                for i_class in uniq_y:  # per class loss
                    class_loss = losses[support_y==i_class].sum() * scaling * omega[i_class]
                    set_loss += class_loss
                set_loss = set_loss
            else:
                set_loss = torch.mean(losses)
            
            grad = torch.autograd.grad(
                set_loss, 
                self.fast_parameters,
                create_graph=True)

            if self.approx:
                grad = [ g.detach() for g in grad ]
            
            # grad step
            self.fast_parameters = []
            for w, named_weight in enumerate(self.get_inner_loop_named_params()):
                name, weight = named_weight
                layer_id = [int(s) for s in name.split('.') if s.isdigit()]
                layer_id = layer_id[0] if len(layer_id) > 0 else -1
                if self.alpha_on:
                    if "trunk" in name: # backbone
                        lr = self.alpha[name]
                    else: # classifier
                        if "weight" in name:
                            lr = self.alpha[name].repeat(self.output_dim[self.mode], 1)
                        else: # bias
                            lr = self.alpha[name].repeat(self.output_dim[self.mode])
                else:
                    lr = self.inner_loop_lr
                g = gamma[layer_id] if self.gamma_on else 1
                if weight.fast is None:
                    weight.fast = weight - g * lr * grad[w] # create weight.fast 
                else:
                    weight.fast = weight.fast - g * lr * grad[w] # update weight.fast
                self.fast_parameters.append(weight.fast)
        
        return kl
    
    def net_eval(self, target_set, ptracker):
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        
        targets_x, targets_y = target_set
        targets_h  = self.backbone.forward(targets_x)
        scores  = self.classifier.forward(targets_h)
        
        loss = self.loss_fn(scores, targets_y)
        loss = torch.mean(loss)
        
        return loss, scores
    
    def get_inner_loop_params(self):
        return [p[1] for p in self.get_inner_loop_named_params()]
    
    def get_inner_loop_named_params(self):
        named_params = list()
        for named_param in self.backbone.named_parameters():
            if '.BN.' not in named_param[0]:
                named_params.append(named_param)
        return named_params + list(self.classifier.named_parameters())
    
    def get_alpha(self, named_weights): # learning rates
        alpha = {}
        for named_weight in named_weights:
            name, weight = named_weight
            if "trunk" in name: # backbone 
                new_weight = torch.ones(weight.shape, requires_grad=True, device=self.device) * self.inner_loop_lr # initialise
            else: # classifier
                new_shape = (1, *weight.shape[1:])
                new_weight = torch.ones(new_shape, requires_grad=True, device=self.device) * self.inner_loop_lr # initialise
            alpha[name] = nn.Parameter(0.01*new_weight)
        return alpha
    
    
    def load_state_dict(self, state_dict):
        for k,v in list(state_dict.items()): 
            if k.startswith('alpha.'):
                self.alpha[k.replace('alpha.', '')] = v
                del state_dict[k]
        return super(BayesianTAML, self).load_state_dict(state_dict)
    
    
    def state_dict(self):
        state_dict = super(BayesianTAML, self).state_dict()
        if self.alpha_on:
            state_dict.update({'alpha.'+k:v for k,v in self.alpha.items()})
        return state_dict
    
            
class InferenceNetwork(nn.Module):
    
    def __init__(self, backbone_num_layers, backbone_layer_channels, args, device):
        super(InferenceNetwork, self).__init__()
        self.backbone_num_layers = backbone_num_layers
        self.backbone_layer_channels = backbone_layer_channels
        self.omega_on = args.omega_on
        self.gamma_on = args.gamma_on
        self.z_on = args.z_on
        self.device = device
        self.num_channel = 3
        self.max_shot = args.max_shot
        self.output_dim = args.output_dim
        
        # sample encoder (1)
        self.sample_encoder = nn.Sequential(*[
            nn.Conv2d(3, 10, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 10, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(10*21*21, 64)   # will fail if image not 84 x 84 
        ]).to(self.device)
        
        # interact1
        self.interact1 = nn.Sequential(*[
            nn.Linear(3, 4),
            nn.ReLU()
        ]).to(device)
        
        # set of (class) sets encoder (2)
        self.stats_encoder = nn.Sequential(*[
            nn.Linear(64*4, 128),  # 64*4 = 256
            nn.ReLU(),
            nn.Linear(128, 32)
        ]).to(device)
        
        # interact2
        self.interact2 = nn.Sequential(*[
            nn.Linear(3, 4),
            nn.ReLU()
        ]).to(device)
        
        # omega encoder
        self.o_encoder = nn.Sequential(*[
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # std and mean
        ]).to(device)
        
        # gamma encoder
        self.g_encoder = nn.Sequential(*[
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * (self.backbone_num_layers + 1)) # std and mean, for each layer (incl. classifier)
        ]).to(device)
        
        # z encoder
        self.z_encoder = nn.Sequential(*[
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * 2 * sum(self.backbone_layer_channels) )  # std and mean, for bias and weight, for each channel
        ]).to(device)
        
        self.softplus = nn.Softplus().to(self.device)
        self.softmax = nn.Softmax(dim=0).to(self.device)

    def _stat_pool(self, x, N):
        # Compute element-wise sample mean, var., and set cardinality
        mean, var = x.mean(dim=0), x.var(dim=0, unbiased=False)
        N = N.reshape([-1]).repeat(mean.shape).to(self.device)
        return torch.stack([mean, var, N], 1)
    
    def get_posterior(self, inputs):
        (x, y) = inputs
        
        # statistics pooling 1
        x = self.sample_encoder(x)
        
        class_stats = []
        class_num = []
        for c in range(torch.max(y)+1):
            x_c = x[y==c]
            n_c = len(x_c) # num of class samples 
            n_c = (n_c-1)/(self.max_shot-1) # normalized class support set size
            
            stat_c = self._stat_pool(x_c, N=torch.tensor(n_c))
            class_stats.append(stat_c)
            class_num.append(n_c)
        
        class_stats = torch.stack(class_stats)
        class_stats = self.interact1(class_stats)
        class_stats = class_stats.view(class_stats.shape[0], -1)
        
        # statistics pooling 2
        encoded_stats = self.stats_encoder(class_stats)
        encoded_stats = self._stat_pool(encoded_stats, N=torch.mean(torch.tensor(class_num)))
        encoded_stats = self.interact2(encoded_stats)
        encoded_stats = encoded_stats.view(1, -1)
        
        # generate omega (from statistics pooling 1) for each class
        o_stats = self.o_encoder(class_stats)
        mu_omega = o_stats[:, 0].squeeze()
        sigma_omega = o_stats[:, 1].squeeze()
        q_omega = torch.distributions.Normal(mu_omega, self.softplus(sigma_omega))
        
        # generate gamma (from statistics pooling 2) for each backbone layer
        g_stats = self.g_encoder(encoded_stats)
        mu_gamma = g_stats[:,0::2].squeeze()     # even indices for mean
        sigma_gamma = g_stats[:,1::2].squeeze()  # odd indices for sigma
        q_gamma = torch.distributions.Normal(mu_gamma, self.softplus(sigma_gamma))
        
        # generate z (from statistics pooling 2) for each backbone layer channel output 
        z_stats = self.z_encoder(encoded_stats)
        mu_z = z_stats[:,0::2].squeeze()     # even indices for mean 
        sigma_z = z_stats[:,1::2].squeeze()  # odd indices for sigma 
        q_z = torch.distributions.Normal(mu_z, self.softplus(sigma_z))
        
        return q_omega, q_gamma, q_z
    
    
    def forward(self, inputs, with_sampling=False):
        
        # compute posterior
        q_omega, q_gamma, q_z = self.get_posterior(inputs)

        # compute kl
        kl_omega = torch.sum(kl_diagnormal_stdnormal(q_omega))
        kl_gamma = torch.sum(kl_diagnormal_stdnormal(q_gamma))
        kl_z     = torch.sum(kl_diagnormal_stdnormal(q_z))
        
        # sample variables from the posterior
        omega, gamma, z = None, None, None
        
        kl = 0.
        
        if self.omega_on:
            kl = kl + kl_omega
            omega = q_omega.rsample() if with_sampling else q_omega.mean
            omega = self.softmax(omega)
        
        if self.gamma_on:
            kl = kl + kl_gamma
            g_ = q_gamma.rsample() if with_sampling else q_gamma.mean
            g_ = torch.exp(g_)
            g_ = torch.split(g_, [1] * (self.backbone_num_layers+1), 0)
            gamma = []
            for l in range(self.backbone_num_layers):
                gamma.append(g_[l])
                
            l = self.backbone_num_layers
            gamma.append(g_[l]) # last gamma for classifier
        
        if self.z_on:
            kl = kl + kl_z
            z_ = q_z.rsample() if with_sampling else q_z.mean
            zw_ = z_[0::2].squeeze()     # even indices for weights 
            zb_ = z_[1::2].squeeze()     # odd indices for biases
            zw_ = torch.split(zw_, self.backbone_layer_channels)
            zb_ = torch.split(zb_, self.backbone_layer_channels)
            
            z = {'w':[], 'b':[]}
            for l in range(self.backbone_num_layers):
                z['w'].append(zw_[l])
                z['b'].append(zb_[l])
                
        return omega, gamma, z, kl
    

def kl_diagnormal_stdnormal(p):
    pshape = p.mean.shape
    device = p.mean.device
    q = torch.distributions.Normal(torch.zeros(pshape, device=device), torch.ones(pshape, device=device))
    return torch.distributions.kl.kl_divergence(p, q).to(device)
