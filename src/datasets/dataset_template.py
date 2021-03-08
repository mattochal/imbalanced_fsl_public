import os.path as osp
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import torchvision.transforms.functional as TF
import random
from typing import Sequence
import random
import copy
import hashlib


from datasets.dataset_utils import ImageJitter, MyRotateTransform, color_transform, mono_transform, UnNormalize, Normalize, partial_shuffle, swap, load_dataset_from_pkl



class DatasetTemplate(Dataset):

    def __init__(self, image_data, class_dict, args):
        self.image_data, self.class_dict = image_data, class_dict
        self.args = args
        self.inv_class_dict = {index:class_name for class_name, indices in self.class_dict.items() for index in indices}
        self.augment = args.aug
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.image_channels = args.image_channels
        self.class_id_to_name = sorted(list(self.class_dict.keys()))
        self.class_name_to_id = {class_name:index for index, class_name in enumerate(self.class_id_to_name)}
        self.data_len =  len(self.image_data)
        self.transform = color_transform(self.augment, args.normalise, self.image_width, self.image_height)
        self.raw_transform = color_transform(False, args.normalise, self.image_width, self.image_height)
        self.hash_signature = hashlib.md5(str.encode(" ".join([
            self.inv_class_dict[i] for i in range(self.data_len)
        ])))
    
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, meta_index):
        raise NotImplementedError()
        
    def get_num_images_per_class(self, class_id):
        """
        Returns number of images contained in the class
        :param class_id: Class id/number of the class
        """
        return len(self.class_dict[self.class_id_to_name[class_id]])
    
    def get_image_idxs_per_class(self, class_id):
        """
        Returns image indexes contained in the class.
        :param class_id: Class id/number of the class
        """
        return self.class_dict[self.class_id_to_name[class_id]]
    
    def get_class_names(self):
        return self.class_id_to_name
    
    def get_class_ids(self):
        return np.arange(len(self.class_id_to_name))
    
    def get_num_classes(self):
        return len(self.class_id_to_name)
    
    def sample_reset(self):
        self.class_dict_cache = {}
    
    def sample_image_idxs(self, class_id, n, seed=0, with_replacement=True):
        """
        Samples n images from the given class, using the given seed
        """
        rng = np.random.RandomState(seed)
        
        if with_replacement:
            image_idx = rng.permutation(self.get_image_idxs_per_class(class_id))[:n]
            return image_idx
        
        else:
            if class_id not in self.class_dict_cache:
                image_idx = rng.permutation(self.get_image_idxs_per_class(class_id))
                self.class_dict_cache[class_id] = image_idx
                
            selected = self.class_dict_cache[class_id][:n]
            self.class_dict_cache[class_id] = self.class_dict_cache[class_id][n:]
            return selected
        
    def get_signature(self):
        return self.hash_signature.hexdigest()
        
            
    
class ColorDatasetInMemory(DatasetTemplate):
    
    def __init__(self, images, class_dict, args):
        """
        Constructor of DatasetInMemory for datasets that can fit in memory. Use DatasetOnDrive to load images from hard drive.
        :param images: All images in a single array or list already loaded in memory
        :param class_dict: Dictionary mapping class names to a list of indices of images belonging to the class
        """
        super().__init__(images,  class_dict,  args)
        assert self.image_channels == 3
        assert np.shape(self.image_data)[-1] == self.image_channels
        
    def __getitem__(self, meta_index):
        """
        Returns a transformed image. To call this method, use normal indexing i.e. dataset_object[index]
        :param meta_index: Meta index containing Image index
        """
        index, tag = meta_index
        label, seed = tag
        image = self.image_data[index]
        h,w,c = image.shape
        image = Image.fromarray(np.uint8(image)).convert('RGB')
        
        # random.seed(seed) # setting the seed significantly reduces data sample variance, decreasing the performance
        image = self.transform(image)
        return image, label
    
    def get_untransformed_image(self, index):
        """
        Returns an untransformed image
        :param index: Image index
        """
        image = self.image_data[index]
        image = Image.fromarray(np.uint8(image))
        return self.raw_transform(image)
        
    
class ColorDatasetOnDisk(DatasetTemplate):
    
    def __init__(self, image_paths, class_dict, args):
        """
        Constructor of DatesetOnDrive for datasets that cannot fit in memory and are to be accessed individually from hard drive.
        :param image_paths: All image directories in a single array or list, paths can be relative or absolute
        :param class_dict: Dictionary mapping class names to a list of indices of images belonging to the class
        """
        super().__init__(image_paths,  class_dict,  args)
        assert self.image_channels == 3
    
    def load_image(self, index):
        """
        Returns an untransformed image loaded directly from disk
        :param index: Image index
        """
        image_path = self.image_data[index]
        image = Image.open(image_path).convert('RGB')
        return image
    
    def __getitem__(self, meta_index):
        """
        Returns a transformed image. To call this method, use normal indexing i.e. dataset_object[index]
        :param index: Image index
        """
        index, tag = meta_index
        label, seed = tag
        image = self.load_image(index)
        h,w,c = image.shape
        assert c == self.image_channels
        # random.seed(seed) # setting the seed significantly reduces data sample variance, decreasing the performance
        image = self.transform(image)
        return image, label
    
    def get_untransformed_image(self, index):
        """
        Returns an untransformed image
        :param index: Image index
        """
        image = self.load_image(index)
        image = Image.fromarray(np.uint8(image))
        return self.raw_transform(image)

    
class MonoDatasetInMemory(DatasetTemplate):
    def __init__(self, images, class_dict, args):
        """
        Black and white dataset in memory
        """
        super().__init__(images, class_dict, args)
        self.transform = mono_transform(self.augment, False, self.image_width, self.image_height)
        self.raw_transform = mono_transform(False, False, self.image_width, self.image_height)
        
    def __getitem__(self, meta_index):
        """
        Returns a transformed image. To call this method, use normal indexing i.e. dataset_object[index]
        :param meta_index: Meta index containing Image index
        """
        index, tag = meta_index
        label, seed = tag
        image = self.image_data[index]
        
        h,w,c = image.shape
        image = Image.fromarray(np.uint8(image).squeeze()).convert('L')
        
        # random.seed(seed) # setting the seed significantly reduces data sample variance 
        image = self.transform(image)
        return image, label
    
    def get_untransformed_image(self, index):
        """
        Returns an untransformed image
        :param index: Image index
        """
        image = self.image_data[index]
        image = Image.fromarray(np.uint8(image))
        return self.raw_transform(image)
        
        
        
class MonoDatasetOnDisk(DatasetTemplate):
    
    def __init__(self, image_paths, class_dict, args):
        """
        Constructor of BWDatasetOnDisk for monochrome datasets to be loaded from the disk.
        """
        super().__init__(image_paths,  class_dict,  args)
        self.transform = mono_transform(self.augment, False, self.image_width, self.image_height)
        self.raw_transform = mono_transform(False, False, self.image_width, self.image_height)
    
    def load_image(self, index):
        """
        Returns an untransformed image loaded directly from disk
        :param index: Image index
        """
        image_path = self.image_data[index]
        image = Image.open(image_path).convert('LA')
        return image
    
    def __getitem__(self, meta_index):
        """
        Returns a transformed image. To call this method, use normal indexing i.e. dataset_object[index]
        :param index: Image index
        """
        index, tag = meta_index
        label, seed = tag
        image = self.load_image(index)
        h,w = image.size
        c = self.c
        
        # random.seed(seed) # setting the seed significantly reduces data sample variance 
        image = self.transform(image)
        return image, label
    
    def get_untransformed_image(self, index):
        """
        Returns an untransformed image
        :param index: Image index
        """
        image = self.load_image(index)
        image = Image.fromarray(np.uint8(image))
        return self.raw_transform(image)
