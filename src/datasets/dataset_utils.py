import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import fnmatch
import os
import tqdm
import concurrent
import threading
import pickle
import itertools
import copy
from collections import defaultdict
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Sequence

from tasks.imbalance_utils import get_num_samples_per_class

import torchvision.transforms.functional as TF


def denorm3d():
    return UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def norm3d():
    return Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def load_dataset_from_pkl(filepath):
    """
    Loads dataset from a prepared pickel file and returns two objects. First, image array containing all images in a list or 
    array. Second, a dictionary mapping class names to image indexes belonging to the class.
    :param filepath: Path to file
    """
    if not osp.isfile(filepath):
        raise Exception("File not found: {}".format(filepath))
    print("Loading data from: {}".format(filepath))
    data = np.load(filepath, allow_pickle=True)
    return data


def rgb_to_bw(images):
    images = np.mean(images, axis=-1)
    images = np.expand_dims(images, axis=-1)
    return images.astype(np.uint8)


def merge_extend_dict(dict1, dict2):
    for i, j in dict2.items():
        dict1[i].extend(j)
    return dict1


def join_data(data_list):
    print("Joining {} datasets".format(len(data_list)))
    data = data_list[0]
    new_data = dict(class_dict=defaultdict(list, data["class_dict"]), image_data=data["image_data"]) 
    offset = len(data["image_data"])
    
    for data in data_list[1:]:
        new_class_dict = {k:np.array(v)+offset for k, v in  data["class_dict"].items()}
        new_data["class_dict"] = merge_extend_dict(new_data["class_dict"], new_class_dict)
        new_data["image_data"] = np.concatenate((new_data["image_data"], data["image_data"]))
        offset += len(data["image_data"])
    
    return new_data
    

def load_dataset_from_from_folder(in_path, cache_path, use_cache=True, image_size=None, use_cache_if_exists=True):
    """
    Scans the given folder for images, where the parenting folder name indicates the image class.
    :param use_cache: If True, generates and/or loads cached images into memory. If False, loads image paths into memory.
    :param image_size: Optional parameter specifying the image size in format (h, w) to resize cached images
    """
    if not os.path.exists(in_path):
        raise Exception("Path not found: {}".format(in_path))
    
    if use_cache:
        data = None
        if not os.path.isfile(cache_path):
            image_paths, class_dict = scan_folder_structure(in_path)
            print("Scanning for images {}".format(in_path,''))
            image_data = load_images_from_paths_parallel(image_paths, final_image_size=image_size)
            data = {"image_data":image_data, "class_dict": class_dict}
            
            print("Saving image cache {}".format(cache_path))
            print("Image shape: {}".format(np.shape(image_data)))
            with open(cache_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            print("Cache file already exists")
            if use_cache_if_exists:
                data = load_dataset_from_pkl(cache_path)
                
        return data
    
    else:
        print("Scanning for images {}".format(in_path))
        image_paths, class_dict = scan_folder_structure(in_path)
        print("Found {} image files".format(len(image_paths)))
        print("No caching performed, set use_cache=True for computation efficiency")
        return  {"image_data": image_paths, "class_dict": class_dict}

    
def scan_folder_structure(directory):
    """
    Scans the given folder for images, where the parenting folder name indicates the image class. 
    Returns filenames and associated class dictionary that maps labels to indexes of images belonging to class.
    """
    image_paths = []
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.gif', '.png', '.JPEG')):
                image_paths.append(os.path.join(root, filename))
                
    class_dict = {}
    for i, image_path in enumerate(image_paths):
        class_label = image_path.split("/")[-2]
        if class_label not in class_dict:
            class_dict[class_label] = [i]
        else:
            class_dict[class_label].append(i)
        
    return image_paths, class_dict


def load_images_from_paths(image_paths):
    print("Loading {} images into RAM".format(len(image_paths)))
    image_data = []

    # Process the list of files, but split the work across the process pool
    with tqdm.tqdm(total=len(image_paths)) as pbar_memory_load:
        for i, image_path in enumerate(image_paths):
            image = load_image(image_path)
            image_data.append(image)
            pbar_memory_load.set_description("Getting: {}".format(os.path.basename(image_path)))
            pbar_memory_load.update(1)
                
    return image_data


def load_images_from_paths_parallel(image_paths, num_threads=16, final_image_size=None):
    print("Loading {} images into RAM".format(len(image_paths)))
    image_data = [None]*len(image_paths)

    # Process the list of files, but split the work across the process pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        with tqdm.tqdm(total=len(image_paths)) as pbar_memory_load:
            for i, (image, image_path) in zip(range(len(image_paths)), 
                                              executor.map(lambda p: load_image(*p), 
                                                           zip(image_paths, itertools.repeat(final_image_size)))):
                image_data[i] = np.asarray(image)
                assert image_path == image_paths[i]
                image_size = np.shape(image)
                assert image_size == final_image_size, "Image was not properly resized. Image of shape {}".format(image_size)
                pbar_memory_load.update(1)
                pbar_memory_load.set_description("Getting{}: {} to {}".format(
                    " and resizing" if final_image_size is not None else "", os.path.basename(image_path), final_image_size))
    return np.asarray(image_data)

    
def load_image(image_path, image_size=None):
    if len(image_size) == 3:
        h,w,c = image_size
    else:
        h,w = image_size
        c = 1
    im = Image.open(image_path)
    im.load()
    if image_size is not None:
        im = im.resize((h,w), Image.ANTIALIAS)
    if c == 3 and im.mode == 'L':
        im = im.convert('RGB')
    return im, image_path
   
    
def color_transform(augment, normalise, image_width, image_height, toPIL=False):
    """
    Returns the trasformation function for data augmentation of colour images
    Add/edit your own augmentation in 'basic_augmentation' variable
    """
    transform_list = []
    
    if toPIL:
        transform_list = [transforms.ToPILImage()]

    if augment:
        transform_list.extend([
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop((image_width, image_height), scale=(0.15, 1.1)),
            ImageJitter(dict(
                Brightness=(0.4, 1.),   # means random level of brightness (upto 0.4) applied 100% of the time 
                Contrast=(0.4, 1.), 
                Color=(0.4, 1.)
#                 Sharpness=(0.4, 1.)
            )),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    else:
        transform_list.extend([
            # leaves out things around the egdes giving an additional boost of 1-5% accuracy points
            transforms.Resize([int(image_height*1.15), int(image_width*1.15)]), 
            transforms.CenterCrop((image_height, image_width))
        ])

    transform_list.append(transforms.ToTensor())

    if normalise:
        transform_list.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        )

    transform = transforms.Compose(transform_list)

    return lambda image: transform(image)


def mono_transform(augment, normalise, image_width, image_height, toPIL=False):
    """
    Returns the trasformation function for data augmentation of monochrome images
    Add/edit your own augmentation in 'basic_augmentation' variable
    """
    transform_list = []
    
    if toPIL:
        transform_list = [transforms.ToPILImage('L')]

    if augment:
        transform_list.extend([
            # transforms.Grayscale(num_output_channels=1),
            MonoImageJitter(dict(
                Brightness=(0.25, 1.), 
                Contrast=(0.25, 1.), 
                Color=(0.25, 1.), 
                Sharpness=(0.25, 1.)
            )),
            transforms.RandomRotation(20, fill=(128,)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop((image_width, image_height), (0.25, 1.0))
        ])
    else:
        transform_list.extend([
            # leaves out things around the egdes giving an additional boost of 1-5% accuracy points
            transforms.Resize([int(image_height*1.15), int(image_width*1.15)]),
            transforms.CenterCrop((image_height, image_width))
        ])
    
    transform_list.append(transforms.ToTensor())
    
    if normalise:
        transform_list.append(
            transforms.Normalize((0.5,), (0.5,))
        )
        
    transform = transforms.Compose(transform_list)

    return lambda image: transform(image)


def prep_datasets(datasets, general_args, conventional_split=False, from_train_only=False):
    
    new_datasets = {}
    
    if (conventional_split and from_train_only):
        image_data, class_dict, args, dataset_class = datasets['train']
        new_image_data, new_class_dict = prep_data(image_data, class_dict, args, 
                                                                                       extra_samples=True)
        new_datasets['train'] = dataset_class(new_image_data, new_class_dict, args)
        print("{} dataset contains: {} images, {} classes".format('train', len(new_image_data), len(new_class_dict)))
        
        # Val from extra train samples
        datasets['val'][0] = extra_image_data
        datasets['val'][1] = extra_class_dict
        
        for split in ["val", "test"]:
            image_data, class_dict, args, dataset_class = datasets[split]
            new_image_data, new_class_dict = prep_data(image_data, class_dict, args)
            new_datasets[split] = dataset_class(new_image_data, new_class_dict, args)
            print("{} dataset contains: {} images, {} classes".format(split, len(new_image_data), len(new_class_dict)))
    
    else:
        
        if (conventional_split and not from_train_only):
            data1 = datasets['train'][:2]
            data2 = datasets['val'][:2]
            newdata1, newdata2 = merge_train_val_and_conventional_split(data1, data2)
            datasets['train'][:2] = newdata1
            datasets['val'][:2] = newdata2
        
        for split in ["train", "val", "test"]:
            image_data, class_dict, args, dataset_class = datasets[split]
            new_image_data, new_class_dict = prep_data(image_data, class_dict, args)
            new_datasets[split] = dataset_class(new_image_data, new_class_dict, args)
            print("{} dataset contains: {} images, {} classes".format(split, len(new_image_data), len(new_class_dict)))
            
    return new_datasets


def prep_data(image_data, class_dict, args, extra_samples=False):
    
    if args.imbalance_distribution is None:
        image_data = image_data
        class_dict = {class_name:np.array(indices) for class_name, indices in class_dict.items()}
        new_image_data, new_class_dict = image_data, class_dict
        return new_image_data, new_class_dict
    
    else:
        frac = 1. if args.use_classes_frac is None else args.use_classes_frac
        num_classes = min(int(len(class_dict) * frac), len(class_dict))
        
        rng = np.random.RandomState(args.seed)
        num_samples = get_num_samples_per_class(args.imbalance_distribution, num_classes, args.min_num_samples, 
                                                     args.max_num_samples, args.num_minority, rng)
        rng.shuffle(num_samples)
        
        class_labels = sorted(class_dict.keys()) # sort for determinism
        
        if num_classes < len(class_labels):
            class_labels = rng.choice(class_labels, num_classes, replace=False)
            
        class_labels = sorted(class_labels)
        
        if extra_samples:
            # Get samples which will be used for validation
            min_samples_per_class = min([ len(class_dict[label]) for label in class_labels ])
            extra_sample_per_class = max(0, min_samples_per_class - num_samples.max())
            
            new_image_data1 = []            
            new_image_data2 = []
            new_class_dict1 = {}
            new_class_dict2 = {}
            index_offset1 = 0
            index_offset2 = 0
            for l, label in enumerate(class_labels):
                class_idx = np.array(class_dict[label])
                n1, n2 = num_samples[l], extra_sample_per_class
                n = min(n1+n2, len(class_idx))
                selected_idx = rng.choice(class_idx, n, replace=False)
                new_image_data1.append(image_data[selected_idx[:n1]])
                new_image_data2.append(image_data[selected_idx[n1:]])
                new_class_dict1[label] = index_offset1 + np.arange(n1)
                new_class_dict2[label] = index_offset2 + np.arange(n2)
                index_offset1 += n1
                index_offset2 += n2
                
            new_image_data1 = np.vstack(new_image_data1)
            new_image_data2 = np.vstack(new_image_data2)
            return new_image_data1, new_class_dict1, new_image_data2, new_class_dict2
        
        else:
            new_image_data = []
            new_class_dict = {}
            index_offset = 0
            
            for l, label in enumerate(class_labels):
                class_idx = np.array(class_dict[label])
                n = min(num_samples[l], len(class_idx))
                selected_idx = rng.choice(class_idx, n, replace=False)
                new_image_data.append(image_data[selected_idx])
                new_class_dict[label] = index_offset + np.arange(n)
                index_offset += n

            new_image_data = np.vstack(new_image_data)
            return new_image_data, new_class_dict

    
def partial_shuffle(array, fraction, seed):  # shuffles elements within a fraction of the array
    if len(array) == 0 or fraction == 0.0:
        return array
    rng = np.random.RandomState(seed)
    n_to_shuffle = np.around(fraction*len(array)).astype(int)
    array_idx = np.arange(len(array))
    idx_to_shuffle = rng.choice(array_idx, n_to_shuffle, replace=False)
    
    new_array = copy.copy(array)
    
    for i in range(n_to_shuffle-1, 0, -1):
        j = rng.randint(i) # add +1 to allow swap with itself
        a = idx_to_shuffle[i]
        b = idx_to_shuffle[j]
        new_array = swap(new_array, a, b)
        
    return new_array


def swap(array, a, b):
    temp = array[a]
    array[a] = array[b]
    array[b] = temp
    return array


def get_unique_counts(y):
    unique =  torch.unique(y)
    counts = []
    for lbl in unique:
        counts.append((y==lbl).sum())
    return unique, torch.tensor(counts)


def conventional_split(image_data, class_dict, split_portions=[0.8, 0.2, 0.0], method='per_class'):
    print("Creating conventional {}/{}/{} split".format(*split_portions))
    data_split = dict(train=dict(), val=dict(), test=dict())
    image_split = dict(train=[], val=[], test=[])
    idx_counter = dict(train=0, val=0, test=0)
    
    if method == 'per_class':
        for cls, idxs in class_dict.items():
            n = len(idxs)
            idxs = np.array(idxs)
            n_train = int(n*split_portions[0])
            n_val = int(n*split_portions[1])
            n_test = int(n*split_portions[2])
            perms = np.random.permutation(n)
            
            data_split['train'][cls] = np.arange(idx_counter['train'], idx_counter['train']+n_train)
            data_split['val'][cls] = np.arange(idx_counter['val'], idx_counter['val']+n_val)
            data_split['test'][cls] = np.arange(idx_counter['test'], idx_counter['test']+n_test)
            
            idx_counter['train'] += n_train
            idx_counter['val'] += n_val
            idx_counter['test'] += n_test
            
            image_split['train'].append(image_data[idxs[perms[:n_train]]])
            image_split['val'].append(image_data[idxs[perms[n_train:n_train+n_val]]])
            image_split['test'].append(image_data[idxs[perms[n_train+n_val:]]])
            
    else:
        # there could be other ways to do this eg. without taking classes into consideration but then there might some imbalance 
        raise Exception('Unimplemented Error')
        
    train = dict(image_data=np.concatenate(image_split['train']), class_dict=data_split['train'])
    val = dict(image_data=np.concatenate(image_split['val']), class_dict=data_split['val'])
    test = dict(image_data=np.concatenate(image_split['test']), class_dict=data_split['test'])
    
    return train, val, test

def merge_train_val_and_conventional_split(data1, data2):
    """
    Merges the train and the validation splits into a single set of classes.
    Creates a conventional split of 75%/25% samples for train/val
    """
    image_data1, class_dict1 = data1
    image_data2, class_dict2 = data2
    
    # joining is slow but works
    data = join_data([
        dict(image_data=image_data1, class_dict=class_dict1),
        dict(image_data=image_data2, class_dict=class_dict2),
    ])
    
    # then split
    train, val, _ = conventional_split(data['image_data'], data['class_dict'])
    data1 = (train['image_data'], train['class_dict'])
    data2 = (val['image_data'], val['class_dict'])
    
    return data1, data2


class ImageJitter(object):
    transformtypedict=dict(
        Brightness=ImageEnhance.Brightness, 
        Contrast=ImageEnhance.Contrast, 
        Sharpness=ImageEnhance.Sharpness, 
        Color=ImageEnhance.Color
    )

    def __init__(self, transformdict):
        self.transforms = [(ImageJitter.transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms)) # pick level of jitter for each transform
        probtensor = torch.rand(len(self.transforms)) # pick probabilty for type of jitter

        for i, (transformer, value) in enumerate(self.transforms):
            alpha, thresh = value
            if probtensor[i] < thresh:
                r = alpha*(randtensor[i]*2.0 -1.0) + 1
                out = transformer(out).enhance(r).convert('RGB')
            
        return out
    
    
class MonoImageJitter(object):
    
    transformtypedict=dict(
        Brightness=ImageEnhance.Brightness, 
        Contrast=ImageEnhance.Contrast, 
        Sharpness=ImageEnhance.Sharpness, 
        Color=ImageEnhance.Color
    )

    def __init__(self, transformdict):
        self.transforms = [(ImageJitter.transformtypedict[k], transformdict[k]) for k in transformdict]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms)) # pick level of jitter for each transform
        probtensor = torch.rand(len(self.transforms)) # pick probabilty for type of jitter
        
        for i, (transformer, value) in enumerate(self.transforms):
            alpha, thresh = value
            if probtensor[i] < thresh:
                r = alpha*(randtensor[i]*2.0 -1.0) + 1
                out = transformer(out).enhance(r).convert('L')
            
        return out


class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
    
    
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor
