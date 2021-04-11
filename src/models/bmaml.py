from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.model_template import ModelTemplate

BayesianMAMLModelState = namedtuple("BayesianMAMLModelState", ["X", "y"])
BayesianMAMLSVGDState = namedtuple("BayesianMAMLSVGDState", ["theta"])


def get_kernel(particle_tensor):
    """
    Compute the RBF kernel for the input particles
    Input: particles = tensor of shape (N, M)
    Output: kernel_matrix = tensor of shape (N, N)
    """
    num_particles = particle_tensor.size(0)

    pairwise_d_matrix = get_pairwise_distance_matrix(particle_tensor)

    median_dist = torch.median(
        pairwise_d_matrix
    )  # tf.reduce_mean(euclidean_dists) ** 2
    h = median_dist / np.log(num_particles)

    kernel_matrix = torch.exp(-pairwise_d_matrix / h)
    kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
    grad_kernel = -torch.matmul(kernel_matrix, particle_tensor)
    grad_kernel += particle_tensor * kernel_sum
    grad_kernel /= h
    return kernel_matrix, grad_kernel, h


def get_pairwise_distance_matrix(particle_tensor):
    """
    Input: tensors of particles
    Output: matrix of pairwise distances
    """
    num_particles = particle_tensor.shape[0]
    euclidean_dists = torch.nn.functional.pdist(
        input=particle_tensor, p=2
    )  # shape of (N)

    # initialize matrix of pairwise distances as a N x N matrix
    pairwise_d_matrix = torch.zeros(
        (num_particles, num_particles), device=particle_tensor.device
    )

    # assign upper-triangle part
    triu_indices = torch.triu_indices(row=num_particles, col=num_particles, offset=1)
    pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

    # assign lower-triangle part
    pairwise_d_matrix = torch.transpose(pairwise_d_matrix, dim0=0, dim1=1)
    pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

    return pairwise_d_matrix


def init_theta(D, n_way, num_particles):
    ret = []

    for _ in range(num_particles):
        w = torch.empty(D, n_way)
        b = torch.empty(n_way)
        log_λ = torch.empty(1)

        torch.nn.init.xavier_normal_(w)
        torch.nn.init.zeros_(b)
        torch.nn.init.ones_(log_λ)

        ret.append(
            torch.cat([w.view(-1), b.view(-1), log_λ.view(-1)], 0).view(1, -1)
        )
    return torch.cat(ret, 0)
 
            
class BayesianMAML(ModelTemplate):
    
    @staticmethod
    def get_parser(parser=None):
        """
        returns a parser for the given model. Can also return a subparser
        """
        if parser is None: parser = argparse.ArgumentParser()
        parser = ModelTemplate.get_parser(parser)
        parser.add_argument('--num_inner_loop_steps', type=int, default=1)
        parser.add_argument('--inner_loop_lr', type=float, default=0.01)
        parser.add_argument('--batch_size', type=int, default=4,
                           help='number of tasks before the outerloop update, eg. update meta learner every 4th task')
        parser.add_argument('--num_draws', type=int, default=20, 
                            help='number of particles')
        parser.add_argument('--output_dim', type=dict, default={"train":-1, "val":-1, "test":-1},
                           help='output dimention for the classifer, if -1 set in code')
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super().__init__(backbone, strategy, args, device)
        self.inner_loop_lr = args.inner_loop_lr
        self.num_inner_loop_steps = args.num_inner_loop_steps
        self.output_dim = args.output_dim
        self.n_way = self.args.output_dim.train
        self.batch_size = args.batch_size
        self.num_draws = args.num_draws
        self.batch_count = 0
        self.batch_losses = []
        self.fast_parameters = []
        assert self.output_dim.train == self.output_dim.test, 'training output dim must mimic the testing scenario'
    
    def setup_model(self):
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.theta = nn.Parameter(
            init_theta(self.backbone.final_feat_dim, self.args.output_dim.train, self.num_draws)
        ).to(self.device)
        
        all_params = list(self.backbone.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                step_size=self.args.lr_decay_step, gamma=self.args.lr_decay)
        self.optimizer.zero_grad()
        self.optimizer.step()
        
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
        self.eval()
        self.net_reset()
        for support_set, target_set in task:
            self.net_train(support_set)
            self.net_eval(target_set, ptracker)
     
    def net_reset(self):
        self.svgd_state = None
        self.model_state = None
        self.strategy.reset()
    
    def net_train(self, support_set): # inner loop
        (support_x, support_y) = self.strategy.update_support_set(support_set)        
        support_h  = self.backbone.forward(support_x)
        (support_h, support_y) = self.strategy.update_support_features((support_h, support_y))
        self.model_state = BayesianMAMLModelState(X=support_h, y=support_y)
        self.svgd_state = self.svgd_update(self.model_state)
        
    def net_eval(self, target_set, ptracker):
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        
        targets_x, targets_y = target_set
        targets_h  = self.backbone.forward(targets_x)
        
        # theta forward
        logits = self.theta_forward(self.svgd_state.theta, targets_h)
        
        # predict
        log_proba = F.log_softmax(logits, -1)
        scores = (torch.logsumexp(log_proba, 0) - np.log(log_proba.size(0))) 
        loss = self.strategy.apply_outer_loss(self.loss_fn, scores, targets_y)
        
        _, pred_y = torch.max(scores, axis=1)
        
        ptracker.add_task_performance(
            pred_y.detach().cpu().numpy(),
            targets_y.detach().cpu().numpy(),
            loss.detach().cpu().numpy())
        
        return loss
    
    def svgd_update(self, model_state):
        svgd_state = BayesianMAMLSVGDState(self.theta)
        for _ in range(self.num_inner_loop_steps):
            svgd_state = self.next_svgd_state(model_state, svgd_state, self.inner_loop_lr)
        return svgd_state
    
    def next_svgd_state(self, model_state, svgd_state, inner_lr):
        grads = []
        
        for particle_ind in range(self.num_draws):
            particle = svgd_state.theta[particle_ind]
            logits = self.theta_forward(particle.unsqueeze(0), model_state.X)[0]
            
            loss = self.strategy.apply_inner_loss(self.loss_fn,
                logits, model_state.y
            ) - self.theta_log_prior(particle)
            
            grads.append(
                torch.autograd.grad(outputs=loss, inputs=particle, create_graph=True)[0]
            )
            
        grads = torch.stack(grads)
        kernel_matrix, grad_kernel, _ = get_kernel(particle_tensor=svgd_state.theta)

        return BayesianMAMLSVGDState(
            svgd_state.theta - inner_lr * (kernel_matrix.matmul(grads) - grad_kernel)
        )
    
    def theta_forward(self, theta, X):
        W = theta[:, : -(self.n_way + 1)].view(-1, self.backbone.final_feat_dim, self.n_way)
        b = theta[:, -(self.n_way + 1) : -1]
        return X.matmul(W) + b.unsqueeze(-2)

    def theta_log_prior(self, theta_flat):
        log_λ = theta_flat[-1]

        λ = torch.exp(log_λ)
        λ_inv = torch.exp(-log_λ)

        w_prior = torch.distributions.Normal(0.0, λ_inv)

        λ_prior = torch.distributions.Gamma(1.0, 0.1)

        return w_prior.log_prob(theta_flat[:-1]).sum() + λ_prior.log_prob(λ).to(
            theta_flat.device
        )
    
