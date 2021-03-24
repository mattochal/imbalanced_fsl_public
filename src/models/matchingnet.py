import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import utils.utils as utils
import copy
from models.model_template import ModelTemplate
import argparse


class MatchingNet(ModelTemplate):
    
    @staticmethod
    def get_parser(parser=None):
        if parser is None: parser = argparse.ArgumentParser(description='MatchingNet')
        """
        returns a parser for the given model. Can also return a subparser
        """
        parser = ModelTemplate.get_parser(parser)
        return parser
    
    def __init__(self, backbone, strategy, args, device):
        super().__init__(backbone, strategy, args, device)
       
    def setup_model(self):
        self.loss_fn = nn.NLLLoss().to(self.device)
        feat_dim = self.backbone.final_feat_dim
        self.FCE = FullyContextualEmbedding(feat_dim, self.device).to(self.device)
        self.G_encoder = nn.LSTM(feat_dim, feat_dim, 1, batch_first=True, bidirectional=True).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device)
        all_params = list(self.backbone.parameters()) + list(self.G_encoder.parameters()) + list(self.FCE.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                step_size=self.args.lr_decay_step, gamma=self.args.lr_decay)
    
    def net_train(self, support_set):  # innerloop / adaptation step
        x, y = self.strategy.update_support_set(support_set)
        z = self.backbone.forward(x)
        z, y = self.strategy.update_support_features((z, y))
        z = z.contiguous().view(len(z), -1)
        self.G = self.encode_training_set(z)
        self.support_y_onehot = Variable(utils.onehot(y).float()).to(self.device)
    
    def net_eval(self, target_set, ptracker):  # evaluation of a target set 
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        x, y = target_set
        z = self.backbone(x)
        y_onehot = self.support_y_onehot
        logprobs = self.get_logprobs(z, self.G, y_onehot)
        loss = self.strategy.apply_outer_loss(self.loss_fn, logprobs, y)
        
        _, pred_y = torch.max(logprobs, 1)
        
        ptracker.add_task_performance(
            pred_y.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
            loss.detach().cpu().numpy())
        return loss
    
    def net_reset(self):
        self.G = None
        self.support_y_onehot = None
        self.strategy.reset()
        
    def encode_training_set(self, S, G_encoder = None):
        if G_encoder is None:
            G_encoder = self.G_encoder
        out_G = G_encoder(S.unsqueeze(0))[0]
        out_G = out_G.squeeze(0)
        G = S + out_G[:,:S.size(1)] + out_G[:,S.size(1):]
        return G
    
    def get_logprobs(self, f, G, Y_S, FCE = None):        
        if FCE is None:
            FCE = self.FCE
        F = FCE(f, G)
        F_norm = torch.norm(F,p=2, dim =1).unsqueeze(1).expand_as(F)
        F_normalized = F.div(F_norm+ 0.00001)
        
        G_norm = torch.norm(G,p=2, dim =1).unsqueeze(1).expand_as(G)
        G_normalized = G.div(G_norm+ 0.00001) 
        
        # The original paper use cosine simlarity, but here we scale it by 100 to strengthen highest probability after softmax
        scores = self.relu( F_normalized.mm(G_normalized.transpose(0,1))  ) *100
        softmax = self.softmax(scores)
        logprobs = (softmax.mm(Y_S)+1e-6).log()
        return logprobs


class FullyContextualEmbedding(nn.Module):
    def __init__(self, feat_dim, device):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(feat_dim*2, feat_dim).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)
        self.c_0 = Variable(torch.zeros(1,feat_dim)).to(device)
        self.feat_dim = feat_dim
        #self.K = K

    def forward(self, f, G):
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose(0,1)
        K = G.size(0) #Tuna to be comfirmed
        for k in range(K):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)
            r = a.mm(G)
            x = torch.cat((f, r),1)

            h, c = self.lstmcell(x, (h, c))
            h = h + f
            
        return h


