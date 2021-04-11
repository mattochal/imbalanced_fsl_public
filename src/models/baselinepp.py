from backbones.layers import distLinear
from models.baseline import Baseline

class BaselinePP(Baseline):
    
    def __init__(self, backbone, strategy, args, device):
        super().__init__(backbone, strategy, args, device)

    def setup_classifier(self, output_dim):
        """
        Setups a normalised linear classifier
        """
        return distLinear(self.backbone.final_feat_dim, output_dim).to(self.device)
    
    
    def setup_classifier(self, output_dim):
        return distLinear(self.backbone.final_feat_dim, output_dim).to(self.device)

    def reset_test_classifier(self):
        self.test_classifier = self.setup_classifier(self.output_dim[self.mode])