import argparse
from tasks.task_template import TaskTemplate
import numpy as np
import time
import sys


class FSLTask(TaskTemplate):
    
    @staticmethod
    def get_parser(parser=None):
        if parser is None:
            parser = argparse.ArgumentParser("FSL Task Parser")
        parser.add_argument('--num_classes', type=int, default=5, 
                            help="Number of classes per episode (n-way).")
        parser.add_argument('--num_supports', type=int, default=5,
                            help="Number of support set supports per class (k-shot).")
        parser.add_argument('--num_targets', type=int, default=15, 
                            help="Number of target supports per class in query set.")
        parser.add_argument('--batch_size', type=int, default=1,
                            help="Number of episodes, sampled independently, in a single batch")
        return parser
    
    @staticmethod
    def get_output_dim(args, dataset):
        return args.num_classes
    
    def __init__(self, dataset, args, class_seed, sample_seed):
        """
        Few Shot Learning Task sampler for creating a single episode for a few-shot learning task
        """
        super().__init__(dataset, args, class_seed, sample_seed)
        self.num_classes = args.num_classes
        self.num_supports = args.num_supports
        self.num_targets = args.num_targets
        self.batch_size = args.batch_size
    
    def __len__(self):
        return self.batch_size
    
    def __iter__(self):
        rng = np.random.RandomState(self.sample_seed)
        
        for batch_id in range(self.batch_size):
            
            rng = np.random.RandomState(self.class_seed)
            total_classes = self.dataset.get_num_classes()
            selected_classes = rng.permutation(total_classes)[:self.num_classes]
            
            supports_x = []
            supports_y = []
            targets_x = []
            targets_y = []
            
            for episode_lbl, actual_lbl in enumerate(selected_classes):
                
                rng = np.random.RandomState(self.sample_seed)
                
                img_idxs = self.dataset.get_image_idxs_per_class(actual_lbl)
                img_idxs = rng.permutation(img_idxs)
                
                supports_x.extend( img_idxs[:self.num_supports]  ) 
                targets_x.extend(  img_idxs[self.num_supports: self.num_supports + self.num_targets] ) 
                
                supports_y.extend([episode_lbl] * self.num_supports)
                targets_y.extend([episode_lbl] * self.num_targets)
            
                    
            support_seeds = rng.randint(0, 999999999, len(supports_y))
            target_seeds = rng.randint(0, 999999999, len(targets_y))
            supports_y = list(zip(supports_y, support_seeds))
            targets_y = list(zip(targets_y, target_seeds))
            
            support_set = (supports_x, supports_y)
            target_set = (targets_x, targets_y)
            
            yield (support_set, target_set)
            
            if self.class_seed is not None:
                self.class_seed += 1
            
            