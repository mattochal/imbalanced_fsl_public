import os
os.environ['PYTHONUNBUFFERED'] = '1'

import shutil
import time
import json
import torch
import numpy as np
import random
import argparse
import sys
import json
import pprint
import re
import warnings
import collections.abc
import copy
from collections import defaultdict
from sklearn.metrics import average_precision_score
from json import JSONEncoder
from torch.nn.modules.module import _addindent
import torch
import numpy as np

# Performance Tracker
from utils.ptracker import PerformanceTracker
from utils.parser_utils import *
from utils.bunch import bunch

# Backbones
from backbones.conv import Conv4, Conv6, Conv4NP, Conv6NP
from backbones.resnet import ResNet10, ResNet18, ResNet34, ResNet50, ResNet101

# Imbalance Distributions
from tasks.imbalance_utils import IMBALANCE_DIST

# Tasks
from tasks.fsl import FSLTask
from tasks.fsl_imbalanced import ImbalancedFSLTask
from tasks.batch_simple import SimpleBatchTask

# Datasets
from datasets.mini import get_MiniImageNet
from datasets.cub import get_CUB200
from datasets.imgnt import get_ImageNet
from datasets.mini_to_cub import get_MiniImageNet_to_CUB200
from datasets.custom import get_custom_dataset_from_folders
from datasets.dataset_utils import prep_datasets

# FSL models
from models.protonet import ProtoNet
from models.relationnet import RelationNet
from models.matchingnet import MatchingNet
from models.baseline import Baseline
from models.baselinepp import BaselinePP
from models.maml import Maml
from models.dkt import DKT
from models.simpleshot import SimpleShot
from models.protomaml import ProtoMaml
from models.bmaml import BayesianMAML
from models.bmaml_chaser import BayesianMAMLChaser
from models.btaml import BayesianTAML
from models.knn import KNN
from models.protodkt import ProtoDKT
from models.relationdkt import RelationDKT

# Imbalance Strategies
from strategies.ros import ROS
from strategies.ros_aug import ROS_AUG
from strategies.focal_loss import FocalLoss
from strategies.weighted_loss import WeightedLoss
from strategies.cb_loss import CBLoss
from strategies.strategy_template import StrategyTemplate

TASKS = {
    "fsl"           : FSLTask,
    "fsl_imbalanced": ImbalancedFSLTask
}

DATASETS = {
    "mini"       : get_MiniImageNet,
    "cub"        : get_CUB200,
    "imgnt"      : get_ImageNet,
    "mini_to_cub": get_MiniImageNet_to_CUB200,
    "custom"     : get_custom_dataset_from_folders
}

MODELS = {
    "protonet"    : ProtoNet,
    "relationnet" : RelationNet,  
    "matchingnet" : MatchingNet,  
    "baseline"    : Baseline,        
    "baselinepp"  : BaselinePP,    
    "maml"        : Maml,                       
    "gpshot"      : DKT,                   
    "dkt"         : DKT,
    "protodkt"    : ProtoDKT,
    "relationdkt" : RelationDKT,
    "simpleshot"  : SimpleShot,
    "protomaml"   : ProtoMaml,
    "bmaml"       : BayesianMAML,
    "bmaml_chaser": BayesianMAMLChaser,
    "btaml"       : BayesianTAML,
    "knn"         : KNN
}

BACKBONES = {
    "Conv4"    : Conv4,
    "Conv6"    : Conv6,
    "ResNet10" : ResNet10,
    "ResNet18" : ResNet18,
    "ResNet34" : ResNet34,
    "ResNet50" : ResNet50,
    "ResNet101": ResNet101
}

STRATEGIES = {
    "ros"           : ROS,
    "ros_aug"       : ROS_AUG,
    "weighted_loss" : WeightedLoss,
    "focal_loss"    : FocalLoss,
    "cb_loss"       : CBLoss,
    None           : StrategyTemplate
}

def get_base_parser(*args, **kwargs):
    """
    Main parser
    """
    parser=argparse.ArgumentParser(*args, **kwargs)
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
    
#     parser.add_argument('--help_all', '--help_model', '--help_dataset', '--help_strategy', '--help_task', '--help_ptracker',
#                         action=PrintHelpAction, nargs=0,
#                         help="Print help for given model, dataset, task, strategy args")
    
    parser.add_argument('--task', type=str, default='fsl', choices=TASKS.keys(),
                        help='Task name')
    parser.add_argument('--dataset', type=str, default='mini', choices=DATASETS.keys(),
                        help='Dataset name')
    parser.add_argument('--model', type=str, default='protonet', choices=MODELS.keys(),
                        help='FSL method name')
    parser.add_argument('--backbone', type=str, default='Conv4', choices=BACKBONES.keys(),
                        help='Backbone neural network name')
    parser.add_argument('--strategy', type=str, default=None, choices=STRATEGIES.keys(),
                       help='Imbalance strategy. If None, no imbalance strategy is used')
    
    parser.add_argument('--gpu', default='0',
                        help='gpu number or "cpu"')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=True,
                        help='If true, the tasks are generated deteministically based on the given seed')
    
    parser.add_argument('--results_folder', type=str, default="../experiments/dummy/", #  default="../../experiments/", 
                        help='parent folder where all experiments are saved')
    parser.add_argument('--experiment_name', type=str, default="default")
    parser.add_argument('--experiment_folder', type=str, default=None,
                        help='experiment folder used to save checkpoints and results')
    parser.add_argument('--clean_folder', type=str2bool, nargs='?', const=True, default=False,
                        help='Clears the experiment folder if it exisits')
    parser.add_argument('--storage_friendly', type=str2bool, nargs='?', const=True, default=True,
                        help='Deletes previously saved models systematically, only keeps best, latest model')
    parser.add_argument('--data_path', type=str, default="data/", 
                        help='Data folder with datasets in named subdirectories.')
    
    parser.add_argument('--continue_from', type=str, default=None, 
                        help="Continue from a checkpoint file, epoch, or 'latest', 'best', or 'from_scratch'/None.")
    parser.add_argument('--load_backbone_only', type=str2bool, nargs='?', const=True, default=False,
                        help="Loads the backbone only from 'continue_from'")
    parser.add_argument('--dummy_run', type=str2bool, nargs='?', const=True, default=False, 
                        help='A dry run of the settings with a 1 epoch and validation, a reduced number of tasks, no saving')
    parser.add_argument('--conventional_split', type=str2bool, nargs='?', const=True, default=None,
                        help='Joins classes in meta-training and meta-validation datests. '
                             'Then conventional 80%%-20%% split for train-val datasets. '
                             'If None, will be split automatically based on model.')
    parser.add_argument('--conventional_split_from_train_only', type=str2bool, nargs='?', const=True, default=False,
                        help='Performs conventional 80%%-20%% data split from the train dataset only,'
                             ' without joining with the validation split. Working only when meta-dataset reduced, see'
                             ' data.dataset_utils.prep_datasets() for details.')
    parser.add_argument('--backbone_channel_dim', type=int, default=64,
                       help='Number of channels of the backbone model.')
    parser.add_argument('--tqdm', type=str2bool, nargs='?', const=True, default=False,
                       help="Enable/Disable tqdm, especially useful when running experiment and redirecting to files")
    
    group = parser.add_argument_group('TASK SAMPLING OPTIONS')
    group.add_argument('--num_epochs', type=int, default=100, 
                       help="If none, then will stop training after achieving a stopping criterion, see ExperimentBuilder")
    group.add_argument('--num_tasks_per_epoch', type=int, default=500)
    group.add_argument('--num_tasks_per_validation', type=int, default=200, 
                        help="Number of tasks to evaluate on after every epoch.")
    group.add_argument('--num_tasks_per_testing', type=int, default=600, 
                        help="Number of tasks to evaluate on after meta-training.")
    group.add_argument('--evaluate_on_test_set_only', '--test', type=str2bool, nargs='?', const=True, default=False, 
                        help="If present, no (further) training is performed and only the test dataset is evaluated.")
    group.add_argument('--val_or_test',  type=str, choices=["test","val"], default="val", 
                        help="Dataset to perform validation on. Default val")
    group.add_argument('--no_val_loop',  type=str2bool, nargs='?', const=True, default=False, 
                        help="No validation loop. Default=False, meaning assume there is a validation loop.")
    group.add_argument('--test_performance_tag', type=str, default="test", 
                        help='The tag name for the performance file evaluated on test set, eg "test" in epoch-###_test.json')
    
    group = parser.add_argument_group('VISUALISATION OPTIONS')
    group.add_argument('--fix_class_distribution', type=str2bool, nargs='?', const=True, default=False, 
                        help='If present, will fix the class distribution such that the model will be evaluated and tested '
                        'on the same set of classes between tasks.')
    group.add_argument('--count_samples_stats', type=str2bool, nargs='?', const=True, default=False,
                       help='If true, counts the images and stores the distribution stats of images shown during the run')
    return parser


def get_dataset_parser(dataset_name, parser=None):
    """
    Generic dataset parser
    """
    if parser is None: parser = argparse.ArgumentParser()

    if dataset_name not in DATASETS:
        raise Exception("Dataset not found: {}".format(dataset_name))
        
    parser.add_argument('--dataset_version', type=str, default=None)  
    parser.add_argument('--data_path',  type=str, default=None,
                        help="Path to folder with all datasets.")
    parser.add_argument('--aug', type=str2bool, nargs='?', const=True, default=False, 
                       help='Boolean for data augmentation. '
                            'Use train.aug/test.aug/val.aug to turn on/off augmentation for different stages')
    parser.add_argument('--normalise', type=str2bool, nargs='?', const=True, default=True, 
                        help='Set true to normalise colour')
    parser.add_argument('--use_cache', type=str2bool, nargs='?', const=True, default=True, 
                       help='Loads and caches a dataset stored in subfolder structure for faster subsequent loading.')
    parser.add_argument('--image_width', type=int, default=84,
                        help='image width')
    parser.add_argument('--image_height', type=int, default=84,
                        help='image height')
    parser.add_argument('--image_channels', type=int, default=3,
                        help='image channels')
    parser.add_argument('--min_num_samples', type=int, default=None,
                        help="Minimum number of samples per class")
    parser.add_argument('--max_num_samples', type=int, default=None,
                        help="Max number of samples per class")
    parser.add_argument('--num_minority', type=float, default=None,
                        help="Fraction of classes used as minority classes (used with 'step'-imbalance distribution)")
    parser.add_argument('--imbalance_distribution', type=str, choices=IMBALANCE_DIST, default=None,
                        help="Imbalance type, specifies how to sample images from larger meta-training dataset.")
    parser.add_argument('--use_classes_frac', type=float, default=None,
                        help="Selects a random subset of classes of dataset (expressed as a fraction in range [0,1]). "
                        "Set 'imbalance_distribution' to 'balanced' or other to make this work.")
    parser.add_argument('--seed', type=int, default=-1,
                        help='seed for dataset manipulations, if -1 uses the main program seed')
    
    return parser


def get_args(sysargv=None, json_args=None):
    """
    Gets parameters passed from the stdin or stdin_list or arg_dict, and loads parameters from a file if available
    Arguments passed through sys.argv take priority over config file args and default parser args
    Arguments passed through config file take priority over default parser args
    """
#     pprint.pprint(sysargv)
#     pprint.pprint(json_args), 
#     import pdb; pdb.set_trace()
    
    if sysargv is None:
        sysargv = sys.argv
    
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--args_file', type=str, default=None)
    
    base_parser = get_base_parser(description="FSL Pytorch Framework", add_help=False, parents=[config_parser])
    
    # Step 1. Get config file
    config_args, remaining_argv = config_parser.parse_known_args(sysargv)
    config_args = vars(config_args)
            
    if json_args is None and config_args['args_file'] not in [None,'None','none','']:
        json_args = load_json(config_args['args_file'])
        json_args = from_syntactic_sugar(json_args)
        json_args['args_file'] = config_args['args_file']
    
    elif json_args is None:
        json_args = {}
    
    # Step 2. Update base args defaults using json
    default_args = vars(base_parser.parse_args([]))
    default_args, excluded_args = update_dict_exclusive(default_args, json_args)
    base_parser.set_defaults(**default_args)
    
    # Step 3. Update base args using command line args
    base_args, remaining_argv = base_parser.parse_known_args(remaining_argv)
    base_args = vars(base_args)
    
    # Step 4. Initilize nested parsers
    model= base_args['model']
    strategy= base_args['strategy']
    dataset= base_args['dataset']
    task= base_args['task']
    
    model_parser = MODELS[model].get_parser(argparse.ArgumentParser())
    strategy_parser = STRATEGIES[strategy].get_parser(argparse.ArgumentParser())
    data_parser = get_dataset_parser(dataset)
    task_parsers = {_phase:_class.get_parser() for _phase,_class in get_task_classes(task, model).items()}
    ptracker_parser = PerformanceTracker.get_parser()
    
    nested_parsers = argparse.ArgumentParser(description="Nested Parser", parents=[base_parser], add_help=False)
    nested_parsers.add_argument('--model_args', **OnePhaseDict.TYPE(), subparser=model_parser,
                               help='FSL method settings as a json parsable string (one phase args)')
    nested_parsers.add_argument('--strategy_args', **OnePhaseDict.TYPE(), subparser=strategy_parser,
                               help='Imbalance strategy settings as a json parsable string (one phase args)')
    nested_parsers.add_argument('--dataset_args', **ThreePhaseDict.TYPE(), subparser=data_parser,
                               help='Dataset settings as a json parsable string (three phase args)')
    nested_parsers.add_argument('--ptracker_args', **ThreePhaseDict.TYPE(), subparser=ptracker_parser,
                               help='Task arguments as a json parsable string (three phase args)')
    nested_parsers.add_argument('--task_args', **ThreePhaseDict.TYPE(), subparser=task_parsers,
                               help='Task settings as a json parsable string (three phase args)')
    
    # Step 5. Translate and expand nested args in excluded args
    is_three_phase = {a.dest:type(a) is ThreePhaseDict for a in nested_parsers._actions}
    for k in excluded_args.keys():
        if k in is_three_phase and is_three_phase[k]:
            excluded_args[k] = expand_three_phase(excluded_args[k])
    
    # Step 6. Updated nested args defaults using base_args
    default_args = vars(nested_parsers.parse_args([]))
    default_args, _excluded = update_dict_exclusive(default_args, base_args)
    assert _excluded == {}, "This should be empty. Something must have gone wrong."
    default_args, excluded_args = update_dict_exclusive(default_args, excluded_args)
    nested_parsers.set_defaults(**default_args)
    
    # Step 7. Update nested args using command line args
    nested_args, remaining_argv = nested_parsers.parse_known_args(remaining_argv)
    nested_args = vars(nested_args)
    
    # Step 8. Delete excluded args left over by nestedparsers
    for k in list(nested_args.keys()):
        if isinstance(nested_args[k], abc.Mapping) and \
           '__excluded' in nested_args[k]:
            if k not in excluded_args:
                excluded_args[k] = {}
            excluded_args[k] = update_dict(excluded_args[k], nested_args[k]['__excluded'])
            if excluded_args[k] == {}: del excluded_args[k]
            del nested_args[k]['__excluded']
       
    # Dry run settings for testing purposes
    if 'dummy_run' in nested_args and nested_args['dummy_run']:
        nested_args.update(dict(
            num_epochs=1,
            num_tasks_per_epoch=3,
            num_tasks_per_validation=3,
            num_tasks_per_testing=3
        ))
    
    return nested_args, excluded_args, nested_parsers

        
def set_torch_seed(seed):
    """
    Sets the torch seed
    """
    torch.manual_seed(seed=seed)
    
    
def set_gpu(x):
    """
    Sets the cpu or gpu device
    """
    if x == 'cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(x))
    print('Using device:', device)
    return device

    
def get_data(args):
    """
    Loads and returns dataset objects for each experiment stage (train, val, test).
    """
    print("Getting data: {}".format(args.dataset))
    
    for phase in args.dataset_args.keys():
        if 'seed' not in args.dataset_args[phase] or args.dataset_args[phase]['seed'] < 0:
            args.dataset_args[phase]['seed'] = args.seed
        if 'data_path' not in args.dataset_args[phase] or args.dataset_args[phase]['data_path'] is None:
            args.dataset_args[phase]['data_path'] = args.data_path
    
    if (args.conventional_split is None and args.model in ["baseline", "baselinepp", "knn"]):
        args.conventional_split = True
    
    datasets = DATASETS[args.dataset](args.dataset_args)
    datasets = prep_datasets(datasets, args,
                             conventional_split=args.conventional_split,
                             from_train_only=args.conventional_split_from_train_only)
    
    return datasets

    
def get_task_classes(task_name, model_name):
    """
    Returns task sampler classes for each stage of the experiment
    """
    if task_name not in TASKS:
        raise Exception("Task {} not found. See utils.get_tasks() or utils.TASKS".format(task_name))
        
    train_task = TASKS[task_name]
    val_task = TASKS[task_name]
    test_task = TASKS[task_name]
    
    # Special samplers for baseline and simpleshot
    if model_name in ["baseline", "baselinepp", "knn"]:
        if task_name in ["fsl", "fsl_imbalanced"]:
            train_task = SimpleBatchTask
            val_task = SimpleBatchTask
            
    if model_name in ["simpleshot"]:
        if task_name in ["fsl", "fsl_imbalanced"]:
            train_task = SimpleBatchTask
    
    task_classes = dict(
        train= train_task, 
        val= val_task, 
        test= test_task 
    )
    
    return task_classes
    
    
def get_tasks(args):
    """
    Returns task sampler classes for each stage of the experiment
    Sorts out task_args
    """
    tasks = get_task_classes(args.task, args.model)
    return tasks


def get_backbone(args, device):
    """
    Returns the backbone model
    """
    
    if args.backbone not in BACKBONES:
        raise Exception("Backbone not found: {}".format(args.backbone))
    
    if args.model in ["relationnet", "relationdkt"]:
        if args.backbone == "Conv4":
            return Conv4NP(device, outdim=args.backbone_channel_dim)
        if args.backbone == "Conv6":
            return Conv6NP(device, outdim=args.backbone_channel_dim)
    
    if args.model in ["maml", "protomaml", "btaml"]:
        return BACKBONES[args.backbone](device, maml=True, outdim=args.backbone_channel_dim)
        
    return BACKBONES[args.backbone](device, outdim=args.backbone_channel_dim)


def get_model(backbone, tasks, datasets, stategy, args, device):
    """
    Returns FSL model and sorts out model_args
    """
    print("Getting model: {}".format(args.model))
    
    if args.model not in MODELS:
        raise Exception("Model {} does not exist".format(args.model))
    
    if 'seed' in args.model_args and args.model_args.seed == -1:
        args.model_args.seed = args.seed
    
    if args.model in ["baseline", "baselinepp", "maml", "gpshot", "dkt", 
                      "relationdkt", "protomaml", "knn", "simpleshot",
                     "bmaml", "bmaml_chaser", "btaml", "btaml_star", "protodkt"]:
        output_dims = dict()
        for s in ["train", "val", "test"]:
            output_dims[s] = tasks[s].get_output_dim(args.task_args[s], datasets[s])
        args.model_args['output_dim'] = bunch.bunchify(output_dims)
        
    if args.model in ['btaml']:
        if args.task in ['fsl']:
            maxshot = max([args.task_args[setname].num_supports for setname in ['train', 'test', 'val']])
        if args.task in ['fsl_imbalanced']:
            maxshot = max([args.task_args[setname].max_num_supports for setname in ['train', 'test', 'val']])
        args.model_args['max_shot'] = maxshot
    
    model = MODELS[args.model](backbone, stategy, args.model_args, device)
    model.setup_model()
    return model.to(device)
                  
       
def get_strategy(args, device):
    print("Getting strategy: {}".format(args.strategy))
    
    if args.strategy not in STRATEGIES:
        raise Exception("Ups. Imbalance strategy {} does not exist!".format(args.strategy))
    
    strategy = STRATEGIES[args.strategy](args.strategy_args, device, args.seed)
    return strategy

def compress_args(args, parser=None):
    compressed_args = copy.deepcopy(args)
    
    if parser is None:
        _,_,parser = get_args([], args)
    
    is_three_phase = {a.dest:type(a) is ThreePhaseDict for a in parser._actions}
    for k in compressed_args.keys():
        if k in is_three_phase and is_three_phase[k]:
            compressed_args[k] = compress_three_phase(compressed_args[k])
    
    return compressed_args
    

def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr 
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr +=  ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr


def onehot(a, dim=None, fill_with=0, dtype=torch.float32):
    """
    Converts a vector into onehot encoding
    """
    if dim is None: dim = a.max()+1
        
    if -1 in a: # i.e. there is a distractor / out of distribution class
        a_tmp = a.clone()
        a_tmp = a_tmp[a != -1]
        b = torch.ones((*a.shape, dim), dtype=dtype) * fill_with
        b[a != -1, a_tmp] = 1
        
    else: 
        b = torch.ones((*a.shape, dim), dtype=dtype) * fill_with
        b[np.arange(len(a)),a] = 1
        
    cuda_check = a.is_cuda
    if cuda_check:
        b = b.to(a.get_device())
    return b


def find(pattern, path):
    """
    Generic file finder method 
    """
    import os, fnmatch
    
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
