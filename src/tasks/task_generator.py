import numpy as np
import random
import time
import argparse


class TaskGenerator():
    
    def __init__(self,
                 dataset,
                 task,
                 num_tasks,
                 seed,
                 epoch,
                 deterministic,
                 fix_classes,
                 mode,
                 task_args):
        """
        Task Generator creates independent tasks for the outerloop of the algorithms.
        :param dataset: Dataset object to generate tasks from
        :param num_tasks: number of tasks
        :param seed: Start seed for the generator
        :param args: Arguments object
        """
        self.task = task
        self.dataset = dataset
        self.num_tasks = num_tasks
        self.epoch_seed = seed
        self.task_args = task_args
        self.epoch = epoch
        self.mode = mode
        self.fix_classes = fix_classes
        self.deterministic = deterministic
        self.task_rng = np.random.RandomState(self.epoch_seed)

    def __len__(self):
        return self.num_tasks
        
    def get_task_sampler(self):
        """
        return appropiate_task_sampler
        """
        
        if not self.deterministic:
            class_seed= None
            sample_seed= None
        else:
            class_seed= self.epoch_seed if self.fix_classes else self.task_seed
            sample_seed= self.task_seed

        return self.task(self.dataset,
                         self.task_args,
                         class_seed= class_seed,
                         sample_seed= sample_seed)
    
    def __iter__(self):
        """
        :returns: a sampler class that samples images from the dataset for the specific task
        """
        for i in range(self.num_tasks):
            self.task_seed = self.task_rng.randint(999999999)
            yield self.get_task_sampler()
            
#             if self.mode == "train":
#                 with open("log.txt", "a") as myfile:
#                     myfile.write('{},'.format(self.seed))
  
# class AuxilaryTaskGenerator():
    
#     def __init__(self,
#                  datasets,
#                  tasks,
#                  task_freqs,
#                  num_tasks,
#                  seed,
#                  epoch,
#                  fix_classes,
#                  mode,
#                  args):
#         """
#         Task Generator creates independent tasks for the outerloop of the algorithms.
#         :param datasets: list of dataset object to generate tasks from
#         :param tasks: list of task samplers 
#         :param num_tasks: number of tasks for each task
#         :param task_freq: frequency of tasks for each task
#         :param seed: Start seed for the generator
#         :param args: Arguments object
#         """
#         self.tasks = tasks
#         self.datasets = datasets
#         self.num_tasks = num_tasks
#         self.task_cycles = task_cycles
#         self.start_seed = seed
#         self.args = args
#         self.epoch = epoch
#         self.mode = mode
#         self.fix_classes = fix_classes
#         self.start_seed = seed
#         rng = np.random.RandomState(self.start_seed)
#         self.seed = rng.randint(9999999)
#         self.iter = 0
#         self.num_tasks_in_cycle = sum(self.num_tasks)
#         self.num_task_types = len(self.tasks)
        
#     def __len__(self):
#         return num_tasks_in_cycle * self.task_cycles
        
#     def get_task_sampler(self):
#         """
#         return appropiate_task_sampler
#         """
#         i = self.iter % self.num_tasks_in_cycle
#         for j in range(self.num_task_types):
#             if sum(self.num_task[:j]) =< i < sum(self.num_task[:j+1])
#                 return self.task[j](self.dataset,
#                                     self.args,
#                                     class_seed= self.start_seed if self.fix_classes else self.seed,
#                                     sample_seed= self.seed)
            
#     def __iter__(self):
#         """
#         :returns: a sampler class that samples images from the dataset for the specific task
#         """
#         for i in range(num_tasks_in_cycle * self.task_cycles):
#             yield self.get_task_sampler()
#             self.seed += 1
#             self.iter += 1
          