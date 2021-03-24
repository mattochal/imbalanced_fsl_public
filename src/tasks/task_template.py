import argparse

class TaskTemplate():
    
    @staticmethod
    def get_parser(parser=None):
        if parser is None: parser=argparse.ArgumentParser()
        return parser
    
    @staticmethod
    def get_output_dim(args, dataset):
        raise NotImplementedError
    
    def __init__(self, dataset, args, class_seed, sample_seed):
        self.dataset = dataset
        self.args = args
        self.class_seed = class_seed
        self.sample_seed = sample_seed
    
    def __len__(self):
        raise NotImplementedError
    
    def __iter__(self):
        raise NotImplementedError
            