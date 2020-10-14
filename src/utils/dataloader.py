import torch
import numpy as np
import os


class DataLoader():
    
    def __init__(self, dataset, task_sampler, device, epoch=None, mode=None, strategy=None):
        self.dataset = dataset
        self.strategy = strategy
        self.n_episodes = len(task_sampler)
        self.task_sampler = task_sampler
        self.device = device
        self.epoch = epoch
        self.mode = mode
        
        
    def __len__(self):
        return self.n_episodes
        
    def __iter__(self):
        
        tasks = []
        
        for subtask in self.task_sampler:
            support_set, target_set = subtask
            support_set = list(zip(*support_set))
            target_set = list(zip(*target_set))
            
            support_x = []
            support_y = []
            targets_x = []
            targets_y = []
            
            for tag in support_set:
                x, y = self.dataset[tag]
                support_x.append(x)
                support_y.append(y)
                
            for tag in target_set:
                x, y = self.dataset[tag]
                targets_x.append(x)
                targets_y.append(y)
            
            if support_x != []:
                support_x = torch.stack(support_x).float()
                support_y = torch.as_tensor(support_y).long()

                n,c,h,w = support_x.shape
                if c == 1 and self.dataset.image_channels == 3:
                    support_x = torch.squeeze(support_x, 1)
                    support_x = torch.stack((support_x,)*3, axis=1)
            else:
                support_x = torch.tensor([])
                support_y = torch.tensor([])
            
            if targets_x != []:
                targets_x = torch.stack(targets_x).float()
                targets_y = torch.as_tensor(targets_y).long()
                
                n,c,h,w = targets_x.shape
                if c == 1 and self.dataset.image_channels == 3:
                    targets_x = torch.squeeze(targets_x, 1)
                    targets_x = torch.stack((targets_x,)*3, axis=1)
            else:
                targets_x = torch.tensor([])
                targets_y = torch.tensor([])
                
            subtask = ((support_x.to(self.device), support_y.to(self.device)), 
                       (targets_x.to(self.device), targets_y.to(self.device)))
        
            yield subtask