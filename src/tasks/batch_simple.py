import argparse
import numpy as np
from tasks.task_template import TaskTemplate
import math

class SimpleBatchTask(TaskTemplate):
        
    @staticmethod
    def get_parser(parser=None):
        if parser is None: parser=argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=32)
        return parser
    
    @staticmethod
    def get_output_dim(args, dataset):
        return dataset.get_num_classes()
        
    def __init__(self, dataset, args, class_seed, sample_seed):
        super().__init__(dataset, args, class_seed, sample_seed)
        self.batch_size = args.batch_size
    
    def __len__(self):
        return 1
    
    def __iter__(self):
        rng = np.random.RandomState(self.sample_seed)
        idx = rng.choice(np.arange(len(self.dataset)), self.batch_size, replace=False)
        seeds = rng.randint(0, 999999, (self.batch_size,))
        labels = [(self.dataset.class_name_to_id[self.dataset.inv_class_dict[i]], i) for i, seed in zip(idx, seeds) ]
        support_set = (idx, labels)
        target_set = ([], [])
        yield (support_set, target_set)
        
            