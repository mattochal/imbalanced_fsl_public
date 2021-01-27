from datasets.dataset_utils import denorm3d, norm3d, ImageJitter
from strategies.strategy_template import StrategyTemplate
import argparse
import numpy as np
import torch
from torchvision import transforms


class FREQ_ROS_AUG(StrategyTemplate):  # random oversampling with augmentation
    
    def get_parser(parser = None):
        if parser is None: parser = argparse.ArgumentParser()
        parser.add_argument('--mono', type=bool, default=False,
                            help='If true apply monochromatic augmentation otherwise apply color augmentation')
        return parser
    
    def __init__(self, args, device, seed):
        super(FREQ_ROS_AUG, self).__init__(args, device, seed)
        self.mono = args.mono
        self.rnd = np.random.RandomState(seed)
        transform = mono_transform if self.mono else color_transform
        self.augment = transform(augment=True, normalise=True, toPIL=True)
        
    def update_support_set(self, support):
        support = super(FREQ_ROS_AUG, self).update_support_set(support)
        x, y = support
        uniq, counts = torch.unique(y, return_counts=True)
        counts = counts.float()
        self.clas_freq = (counts / counts.mean()).to(self.device)
        return self.oversample(support, self.augment)
    
    def oversample(self, supports, augment):
        x, y = supports
        device = x.device
        
        x_cpu = x.cpu()
        uniq, count = torch.unique(y, return_counts=True)
        max_count = count.max().cpu().numpy()
        new_idx = []
        
        resampled_x = []
        resampled_y = []
        
        for i, cls in enumerate(uniq):
            clss_idx = torch.where(y == cls)[0].cpu().numpy()
            n_resampled = max_count - len(clss_idx)
            resampled_idx = self.rnd.choice(clss_idx, n_resampled)
            
            for idx in resampled_idx:
                resampled_x.append(augment(x_cpu[idx].clone()))
                resampled_y.append(cls)
                
        if len(resampled_x) == 0:
            return x, y
        
        resampled_x = torch.stack(resampled_x).to(device)
        resampled_y = torch.stack(resampled_y)
        
        new_x = torch.cat((x, resampled_x))
        new_y = torch.cat((y, resampled_y))
        
        return new_x, new_y

    def apply_inner_loss(self, loss_fn, *args):
#         super(FREQ_ROS_AUG, self).apply_inner_loss(loss_fn, *args)
        weighted_loss_fn = type(loss_fn)(weight=self.clas_freq)
        return weighted_loss_fn(*args)
    
#     def apply_outer_loss(self, loss_fn, *args):
#         super(FREQ_ROS_AUG, self).apply_outer_loss(loss_fn, *args)
#         weighted_loss_fn = type(loss_fn)(weight=self.clas_freq)
#         return weighted_loss_fn(*args)
    

class FREQ_ROS_AUG2(StrategyTemplate):  # random oversampling with augmentation
    
    def get_parser(parser = None):
        if parser is None: parser = argparse.ArgumentParser()
        parser.add_argument('--mono', type=bool, default=False,
                            help='If true apply monochromatic augmentation otherwise apply color augmentation')
        return parser
    
    def __init__(self, args, device, seed):
        super(FREQ_ROS_AUG2, self).__init__(args, device, seed)
        self.mono = args.mono
        self.rnd = np.random.RandomState(seed)
        transform = mono_transform if self.mono else color_transform
        self.augment = transform(augment=True, normalise=True, toPIL=True)
        
    def update_support_set(self, support):
        support = super(FREQ_ROS_AUG2, self).update_support_set(support)
        x, y = support
        uniq, counts = torch.unique(y, return_counts=True)
        counts = counts.float()
        self.clas_freq = ( counts.mean() / counts ).to(self.device)
        return self.oversample(support, self.augment)
    
    def oversample(self, supports, augment):
        x, y = supports
        device = x.device
        
        x_cpu = x.cpu()
        uniq, count = torch.unique(y, return_counts=True)
        max_count = count.max().cpu().numpy()
        new_idx = []
        
        resampled_x = []
        resampled_y = []
        
        for i, cls in enumerate(uniq):
            clss_idx = torch.where(y == cls)[0].cpu().numpy()
            n_resampled = max_count - len(clss_idx)
            resampled_idx = self.rnd.choice(clss_idx, n_resampled)
            
            for idx in resampled_idx:
                resampled_x.append(augment(x_cpu[idx].clone()))
                resampled_y.append(cls)
                
        if len(resampled_x) == 0:
            return x, y
        
        resampled_x = torch.stack(resampled_x).to(device)
        resampled_y = torch.stack(resampled_y)
        
        new_x = torch.cat((x, resampled_x))
        new_y = torch.cat((y, resampled_y))
        
        return new_x, new_y

    def apply_inner_loss(self, loss_fn, *args):
        weighted_loss_fn = type(loss_fn)(weight=self.clas_freq)
        return weighted_loss_fn(*args)


class FREQ_ROS_AUG3(StrategyTemplate):  # random oversampling with augmentation
    
    def get_parser(parser = None):
        if parser is None: parser = argparse.ArgumentParser()
        parser.add_argument('--mono', type=bool, default=False,
                            help='If true apply monochromatic augmentation otherwise apply color augmentation')
        return parser
    
    def __init__(self, args, device, seed):
        super(FREQ_ROS_AUG3, self).__init__(args, device, seed)
        self.mono = args.mono
        self.rnd = np.random.RandomState(seed)
        transform = mono_transform if self.mono else color_transform
        self.augment = transform(augment=True, normalise=True, toPIL=True)
        
    def update_support_set(self, support):
        support = super(FREQ_ROS_AUG3, self).update_support_set(support)
        x, y = support
        uniq, counts = torch.unique(y, return_counts=True)
        counts = counts.float()
        counts += (counts.max() - counts) * 0.5 # add to counts based on the number of oversampled images
        self.clas_freq = ( counts.mean() / counts ).to(self.device)
        return self.oversample(support, self.augment)
    
    def oversample(self, supports, augment):
        x, y = supports
        device = x.device
        
        x_cpu = x.cpu()
        uniq, count = torch.unique(y, return_counts=True)
        max_count = count.max().cpu().numpy()
        new_idx = []
        
        resampled_x = []
        resampled_y = []
        
        for i, cls in enumerate(uniq):
            clss_idx = torch.where(y == cls)[0].cpu().numpy()
            n_resampled = max_count - len(clss_idx)
            resampled_idx = self.rnd.choice(clss_idx, n_resampled)
            
            for idx in resampled_idx:
                resampled_x.append(augment(x_cpu[idx].clone()))
                resampled_y.append(cls)
                
        if len(resampled_x) == 0:
            return x, y
        
        resampled_x = torch.stack(resampled_x).to(device)
        resampled_y = torch.stack(resampled_y)
        
        new_x = torch.cat((x, resampled_x))
        new_y = torch.cat((y, resampled_y))
        
        return new_x, new_y

    def apply_inner_loss(self, loss_fn, *args):
        weighted_loss_fn = type(loss_fn)(weight=self.clas_freq)
        return weighted_loss_fn(*args)
    
class FREQ_ROS_AUG4(StrategyTemplate):  # random oversampling with augmentation
    
    def get_parser(parser = None):
        if parser is None: parser = argparse.ArgumentParser()
        parser.add_argument('--mono', type=bool, default=False,
                            help='If true apply monochromatic augmentation otherwise apply color augmentation')
        return parser
    
    def __init__(self, args, device, seed):
        super(FREQ_ROS_AUG4, self).__init__(args, device, seed)
        self.mono = args.mono
        self.rnd = np.random.RandomState(seed)
        transform = mono_transform if self.mono else color_transform
        self.augment = transform(augment=True, normalise=True, toPIL=True)
        
    def update_support_set(self, support):
        support = super(FREQ_ROS_AUG4, self).update_support_set(support)
        x, y = support
        uniq, counts = torch.unique(y, return_counts=True)
        counts = counts.float()
        counts += (counts.max() - counts) * 0.25 # add to counts based on the number of oversampled images
        self.clas_freq = ( counts.mean() / counts ).to(self.device)
        return self.oversample(support, self.augment)
    
    def oversample(self, supports, augment):
        x, y = supports
        device = x.device
        
        x_cpu = x.cpu()
        uniq, count = torch.unique(y, return_counts=True)
        max_count = count.max().cpu().numpy()
        new_idx = []
        
        resampled_x = []
        resampled_y = []
        
        for i, cls in enumerate(uniq):
            clss_idx = torch.where(y == cls)[0].cpu().numpy()
            n_resampled = max_count - len(clss_idx)
            resampled_idx = self.rnd.choice(clss_idx, n_resampled)
            
            for idx in resampled_idx:
                resampled_x.append(augment(x_cpu[idx].clone()))
                resampled_y.append(cls)
                
        if len(resampled_x) == 0:
            return x, y
        
        resampled_x = torch.stack(resampled_x).to(device)
        resampled_y = torch.stack(resampled_y)
        
        new_x = torch.cat((x, resampled_x))
        new_y = torch.cat((y, resampled_y))
        
        return new_x, new_y

    def apply_inner_loss(self, loss_fn, *args):
        weighted_loss_fn = type(loss_fn)(weight=self.clas_freq)
        return weighted_loss_fn(*args)
    
class FREQ_ROS_AUG5(StrategyTemplate):  # random oversampling with augmentation
    
    def get_parser(parser = None):
        if parser is None: parser = argparse.ArgumentParser()
        parser.add_argument('--mono', type=bool, default=False,
                            help='If true apply monochromatic augmentation otherwise apply color augmentation')
        return parser
    
    def __init__(self, args, device, seed):
        super(FREQ_ROS_AUG5, self).__init__(args, device, seed)
        self.mono = args.mono
        self.rnd = np.random.RandomState(seed)
        transform = mono_transform if self.mono else color_transform
        self.augment = transform(augment=True, normalise=True, toPIL=True)
        
    def update_support_set(self, support):
        support = super(FREQ_ROS_AUG5, self).update_support_set(support)
        x, y = support
        uniq, counts = torch.unique(y, return_counts=True)
        counts = counts.float()
        counts += (counts.max() - counts) * 0.5 # add to counts based on the number of oversampled images
        self.clas_freq = ( 1.0 / counts ).to(self.device)
        return self.oversample(support, self.augment)
    
    def oversample(self, supports, augment):
        x, y = supports
        device = x.device
        
        x_cpu = x.cpu()
        uniq, count = torch.unique(y, return_counts=True)
        max_count = count.max().cpu().numpy()
        new_idx = []
        
        resampled_x = []
        resampled_y = []
        
        for i, cls in enumerate(uniq):
            clss_idx = torch.where(y == cls)[0].cpu().numpy()
            n_resampled = max_count - len(clss_idx)
            resampled_idx = self.rnd.choice(clss_idx, n_resampled)
            
            for idx in resampled_idx:
                resampled_x.append(augment(x_cpu[idx].clone()))
                resampled_y.append(cls)
                
        if len(resampled_x) == 0:
            return x, y
        
        resampled_x = torch.stack(resampled_x).to(device)
        resampled_y = torch.stack(resampled_y)
        
        new_x = torch.cat((x, resampled_x))
        new_y = torch.cat((y, resampled_y))
        
        return new_x, new_y

    def apply_inner_loss(self, loss_fn, *args):
        weighted_loss_fn = type(loss_fn)(weight=self.clas_freq)
        return weighted_loss_fn(*args)
    
def color_transform(augment, normalise, toPIL=False):
    """
    Returns the trasformation function for data augmentation of colour images
    Add/edit your own augmentation in 'basic_augmentation' variable
    """
    transform_list = []
    image_width = 84
    image_height = 84  # hard coded for now, but shouldn't be
    
    if normalise: transform_list.append(denorm3d())
    
    if toPIL: transform_list.append(transforms.ToPILImage())
    
    if augment:
        transform_list.extend([
            transforms.RandomResizedCrop((image_width, image_height), scale=(0.15, 1.1)),
            ImageJitter(dict(
                Brightness=(0.4, 1.),
                Contrast=(0.4, 1.), 
                Color=(0.4, 1.)
            )),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    else:
        transform_list.extend([
            transforms.Resize([int(image_height*1.0), int(image_width*1.0)]), 
            transforms.CenterCrop((image_height, image_width))
        ])

    transform_list.append(transforms.ToTensor())
    
    if normalise: transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        
    transform = transforms.Compose(transform_list)
    return lambda image: transform(image)


def mono_transform(augment, normalise, image_width, image_height, toPIL=False, from3dim=False):
    """
    Returns the trasformation function for data augmentation of monochrome images
    Add/edit your own augmentation in 'basic_augmentation' variable
    """
    transform_list = []
    
    if from3dim: lambda x: x[:,0:1,:,:]
    
    if toPIL: transform_list = [transforms.ToPILImage('L')]

    if augment:
        transform_list.extend([
            # transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(10, fill=(128,)),
            transforms.RandomResizedCrop((image_width, image_height), scale=(0.15, 1.1)),
            MonoImageJitter(dict(
                Brightness=(0.25, 1.), 
                Contrast=(0.25, 1.), 
                Color=(0.25, 1.)
            )),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    else:
        transform_list.extend([
            transforms.Resize([int(image_height*1.0), int(image_width*1.0)]), 
            transforms.CenterCrop((image_height, image_width))
        ])
    
    transform_list.append(transforms.ToTensor())
    
    if normalise: transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    if from3dim: lambda x: torch.stack((torch.squeeze(x, 1),)*3, axis=1)
    
    transform = transforms.Compose(transform_list)

    return lambda image: transform(image)