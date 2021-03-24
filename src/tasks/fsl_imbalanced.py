import argparse
from .task_template import TaskTemplate
import numpy as np
import time
from tasks.imbalance_utils import get_num_samples_per_class, IMBALANCE_DIST


class ImbalancedFSLTask(TaskTemplate):
    
    @staticmethod
    def get_parser(parser=None):
        if parser is None: parser=argparse.ArgumentParser()
        parser.add_argument('--num_classes', type=int, default=5, 
                            help="Number of classes per episode (n-way).")
        
        parser.add_argument('--min_num_supports', type=int, default=1,
                            help="Number of support set samples per class (min k-shot).")
        parser.add_argument('--max_num_supports', type=int, default=5,
                            help="Number of support set samples per class (max k-shot).")
        parser.add_argument('--num_minority', type=float, default=1,
                            help="Fraction of classes used as minority classes (used with 'step'-imbalance distribution)")
        parser.add_argument('--imbalance_distribution', type=str, choices=IMBALANCE_DIST, default='linear',
                            help="Imbalance type, specifies how to sample supports.")
        
        parser.add_argument('--min_num_targets', type=int, default=15,
                            help="Number of target set samples per class (min k-shot).")
        parser.add_argument('--max_num_targets', type=int, default=15,
                            help="Number of target set samples per class (max k-shot).")
        parser.add_argument('--num_minority_targets', type=float, default=1,
                            help="Fraction of classes used as minority classes in target set (used with 'step'-imbalance)")
        parser.add_argument('--imbalance_distribution_targets', type=str, choices=IMBALANCE_DIST, default='balanced',
                            help="Imbalance type for targets, specifies how to sample targets.")
        
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
        
        self.min_num_supports = args.min_num_supports
        self.max_num_supports = args.max_num_supports
        self.num_minority = args.num_minority
        self.imbalance_distribution = args.imbalance_distribution
        
        self.min_num_targets = args.min_num_targets
        self.max_num_targets = args.max_num_targets
        self.num_minority_targets = args.num_minority_targets
        self.imbalance_distribution_targets = args.imbalance_distribution_targets
        
        self.batch_size = args.batch_size
    
    def __len__(self):
        return self.batch_size
    
    def __iter__(self):
        rng = np.random.RandomState(self.sample_seed)
        sampling_seed = rng.randint(9999999)
        
        for batch_id in range(self.batch_size):
            
            rng = np.random.RandomState(self.class_seed)
            total_classes = self.dataset.get_num_classes()
            selected_classes = rng.permutation(total_classes)[:self.num_classes]
            
            num_supports = get_num_samples_per_class(self.imbalance_distribution, self.num_classes, self.min_num_supports, 
                                                     self.max_num_supports, self.num_minority, rng)
            num_targets = get_num_samples_per_class(self.imbalance_distribution_targets, self.num_classes, self.min_num_targets, 
                                                     self.max_num_targets, self.num_minority_targets, rng)
            
            supports_x = []
            supports_y = []
            targets_x = []
            targets_y = []
            
            for lbl, actual_lbl in enumerate(selected_classes):
                
                rng = np.random.RandomState(self.sample_seed)
                img_idxs = self.dataset.get_image_idxs_per_class(actual_lbl)
                img_idxs = rng.permutation(img_idxs)
                
                supports_x.extend( img_idxs[:num_supports[lbl]]  ) 
                targets_x.extend(  img_idxs[num_supports[lbl]: num_supports[lbl] + num_targets[lbl]] )
                
                supports_y.extend([lbl] * num_supports[lbl])
                targets_y.extend([lbl] * num_targets[lbl])
            
            support_seeds = rng.randint(0, 999999999, len(supports_y))
            target_seeds = rng.randint(0, 999999999, len(targets_y))
            supports_y = zip(supports_y, support_seeds)
            targets_y = zip(targets_y, target_seeds)
            
            support_set = (supports_x, supports_y)
            target_set = (targets_x, targets_y)
            
            yield (support_set, target_set)
            
            if self.class_seed is not None:
                self.class_seed += 1
