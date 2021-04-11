import torch.nn as nn

class BackboneTemplate(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_feat_dim = []
        self.layer_channels = []
        self.trunk = None
    
    def forward(self, x):
        raise NotImplementedError