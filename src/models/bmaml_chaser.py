from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.bmaml import BayesianMAML, BayesianMAMLModelState, BayesianMAMLSVGDState


class BayesianMAMLChaser(BayesianMAML):
    
    @staticmethod
    def get_parser(parser=None):
        """
        returns a parser for the given model. Can also return a subparser
        """
        if parser is None: parser = argparse.ArgumentParser()
        parser = BayesianMAML.get_parser(parser)
        parser.add_argument('--leader_inner_loop_lr', type=float, default=0.005)
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super(BayesianMAMLChaser, self).__init__(backbone, strategy, args, device)
        self.leader_inner_lr = args.leader_inner_loop_lr

    def theta_log_prior(self, theta_flat):
        λ_inv = 1.0
        w_prior = torch.distributions.Normal(0.0, λ_inv)
        return w_prior.log_prob(theta_flat[:-1]).sum().to(theta_flat.device)
    
    def meta_train(self, task, ptracker): # single iter of meta training (outer) loop 
        self.mode='train'
        self.train()
        self.net_reset()
        self.batch_count += 1
        
        total_losses = []
        for support_set, target_set in task:
            support_set = self.strategy.update_support_set(support_set)
            support_x, support_y = support_set
            targets_x, targets_y = target_set
            support_n = len(support_y)

            # Combine target and support set
            if len(targets_x) > 0:
                all_x = torch.cat((support_x, targets_x), dim=0)
                all_y = torch.cat((support_y, targets_y), dim=0)
            else:
                all_x = support_x
                all_y = support_y
            
            all_h = self.backbone.forward(all_x)
            all_h, all_y = self.strategy.update_support_features((all_h, all_y))
            support_h = all_h[:support_n]
            
            # chaser
            model_state = BayesianMAMLModelState(X=support_h, y=support_y)
            chaser_svgd_state = self.svgd_update(model_state)

            # leader
            merged_model_state = BayesianMAMLModelState(X=all_h, y=all_y)
            leader_svgd_state = BayesianMAMLSVGDState(chaser_svgd_state.theta)
            
            for _ in range(self.num_inner_loop_steps):
                leader_svgd_state = self.next_svgd_state(
                    merged_model_state, leader_svgd_state, self.leader_inner_lr
                )

            loss = torch.pow(
                chaser_svgd_state.theta - leader_svgd_state.theta.detach(), 2
            ).sum()
            
            total_losses.append(loss)
            
            if len(targets_x) != 0:
                
                with torch.no_grad():
                    targets_h = all_h[support_n:]

                    # theta forward
                    logits = self.theta_forward(chaser_svgd_state.theta, targets_h)

                    # predict
                    log_proba = F.log_softmax(logits, -1)
                    scores = (torch.logsumexp(log_proba, 0) - np.log(log_proba.size(0))) 
                    
                    target_loss = self.strategy.apply_outer_loss(self.loss_fn, scores, targets_y)

                    _, pred_y = torch.max(scores, axis=1) 

                    ptracker.add_task_performance(
                        pred_y.detach().cpu().numpy(),
                        targets_y.detach().cpu().numpy(),
                        target_loss.detach().cpu().numpy())
        
        loss = torch.stack(total_losses).sum(0)
        self.batch_losses.append(loss)
        
        if self.batch_count % self.batch_size == 0:
            self.optimizer.zero_grad()
            loss = torch.stack(self.batch_losses).sum(0)
            loss.backward()
            self.optimizer.step()
            self.batch_losses = []
            