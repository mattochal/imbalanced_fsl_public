import os
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

# Backbones
import models.backbones as backbones

# Imbalance Distributions
from tasks.imbalance_utils import IMBALANCE_DIST

# Tasks
from tasks.fsl import FSLTask
from tasks.fsl_imbalanced import ImbalancedFSLTask
from tasks.batch_simple import SimpleBatchTask

# Datasets
from datasets.mini import get_MiniImageNet
from datasets.cub import get_CUB200
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
from models.gpshot import GPShot
from models.simpleshot import SimpleShot
from models.protomaml import ProtoMaml
from models.bmaml import BayesianMAML
from models.bmaml_chaser import BayesianMAMLChaser
from models.btaml import BayesianTAML
from models.knn import KNN

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
    "gpshot"      : GPShot,
    "simpleshot"  : SimpleShot,
    "protomaml"   : ProtoMaml,
    "bmaml"       : BayesianMAML,
    "bmaml_chaser": BayesianMAMLChaser,
    "btaml"       : BayesianTAML,
    "knn"         : KNN,
}

BACKBONES = {
    "Conv4"    : backbones.Conv4,
    "Conv4S"   : backbones.Conv4S,
    "Conv6"    : backbones.Conv6,
    "ResNet10" : backbones.ResNet10,
    "ResNet18" : backbones.ResNet18,
    "ResNet34" : backbones.ResNet34,
    "ResNet50" : backbones.ResNet50,
    "ResNet101": backbones.ResNet101
}

STRATEGIES = {
    "ros"          : ROS,
    "ros_aug"      : ROS_AUG,
    "weighted_loss": WeightedLoss,
    "focal_loss"   : FocalLoss,
    "cb_loss"      : CBLoss,
    None           : StrategyTemplate
}

def get_main_parser(optional=None):
    """
    Main parser
    """
    parser=argparse.ArgumentParser(
        description="FSL Pytorch Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument('--help_all', '--help_model', '--help_dataset', '--help_strategy', '--help_task', '--help_ptracker',
                        action=PrintHelpAction, nargs=0,
                        help="Print help for given model, dataset, task, strategy args")
    
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
    
    parser.add_argument('--model_args', type=json.loads, default=dict(),
                        help='FSL method settings as a json parsable string')
    parser.add_argument('--dataset_args', type=json.loads, default=dict(),
                        help='Dataset settings as a json parsable string')
    parser.add_argument('--task_args', type=json.loads, default=dict(),
                        help='Task settings as a json parsable string')
    parser.add_argument('--strategy_args',  type=json.loads, default=dict(),
                       help='Imbalance strategy settings as a json parsable string')
    
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
    
    parser.add_argument('--args_file', type=str, default="None", 
                        help="file path to json configuration file")
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
    parser.add_argument('--disable_tqdm', type=str2bool, nargs='?', const=True, default=True,
                       help="Disable tqdm, especially useful when running experiment and redirecting to files")
    
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
    group.add_argument('--test_performance_tag', type=str, default="test", 
                        help='The tag name for the performance file evaluated on test set, eg "test" in epoch-###_test.json')
    
    group = parser.add_argument_group('VISUALISATION OPTIONS')
    group.add_argument('--fix_class_distribution', type=str2bool, nargs='?', const=True, default=False, 
                        help='If present, will fix the class distribution such that the model will be evaluated and tested '
                        'on the same set of classes between tasks.')
    group.add_argument('--count_samples_stats', type=str2bool, nargs='?', const=True, default=False,
                       help='If true, counts the images and stores the distribution stats of images shown during the run')
    
    group = parser.add_argument_group('PERFORMANCE TRACKING')
    group.add_argument('--ptracker_args', type=json.loads, default=dict(),
                        help='Task arguments as a json parsable string')
    return parser


def get_dataset_parser(dataset_name, parser=None):
    """
    Generic dataset parser
    """
    if parser is None: parser = argparse.ArgumentParser()

    if dataset_name not in DATASETS:
        raise Exception("Dataset not found: {}".format(dataset_name))
        
    parser.add_argument('--dataset_version', type=str, default=None)  
    parser.add_argument('--data_path',  type=str, default="../data/", # default="../../data/", 
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
    parser.add_argument('--min_num_samples', type=int, default=1,
                        help="Minimum number of samples per class")
    parser.add_argument('--max_num_samples', type=int, default=10,
                        help="Max number of samples per class")
    parser.add_argument('--num_minority', type=float, default=1,
                        help="Fraction of classes used as minority classes (used with 'step'-imbalance distribution)")
    parser.add_argument('--imbalance_distribution', type=str, choices=IMBALANCE_DIST, default=None,
                        help="Imbalance type, specifies how to sample images from larger meta-training dataset.")
    parser.add_argument('--use_classes_frac', type=float, default=None,
                        help="Selects a random subset of classes of dataset (expressed as a fraction in range [0,1]). "
                        "Set 'imbalance_distribution' to 'balanced' or other to make this work.")
    parser.add_argument('--seed', type=int, default=-1,
                        help='seed for dataset manipulations, if -1 uses the main program seed')
    
    return parser

    
def get_raw_args(parser, stdin_list=None, args_dict=dict()):
    """
    Gets parameters passed from the stdin or stdin_list or arg_dict, and loads parameters from a file if available
    Arguments passed through arg_dict take priority over stdin or stdin_list, and args_file
    Arguments passed through stdin or stdin_list take priority over args_
    """
    args = parser.parse_args(stdin_list) # default args (or generated by parsing stdin_list)
    args_vars = vars(args)
    args_vars.update(args_dict)
    
    stdin_list = sys.argv if stdin_list is None else stdin_list
    stdin_list_keys = [arg[2:] for arg in stdin_list if arg.startswith("--") \
                       and arg not in ["--dataset_args", "--model_args", "--task_args", "--ptracker_args", "--strategy_args"]]
    
    # Update args from file if available
    if args.args_file not in ["None", None, ""]: 
        
        # Stores expandable args (those generated by nested parsers) in a separete variable to sort out later
        args_vars["_dataset_args"] = args.dataset_args
        args_vars["_model_args"] = args.model_args
        args_vars["_task_args"] = args.task_args
        args_vars["_ptracker_args"] = args.ptracker_args
        args_vars["_strategy_args"] = args.strategy_args
    
        args_json = extract_args_from_file(args.args_file, to_ignore=stdin_list_keys)
        args_vars.update(args_json)
        
    args = sortout_ptracker_args(args)
       
    # Dummy run settings for testing purposes
    if 'dummy_run' in args_vars and args_vars['dummy_run']:
        args_vars.update(dict(
            num_epochs=1,
            num_tasks_per_epoch=3,
            num_tasks_per_validation=3,
            num_tasks_per_testing=3
        ))
        
    args = Bunch(args_vars)
    
    os.environ['PYTHONUNBUFFERED'] = '1'  # good formatting 
    
    print(" -------------- GIVEN ARGS -------------")
    pprint.pprint(args.__dict__, indent=2)
    print(" ---------------------------------------")
    return args
    
    
def extract_args_from_file(json_file, to_ignore=[]):
    """
    Extracts arguments from a json file. 
    Omits to_ignore list of hyperparameters so that the ones specified from the console take priority.
    """
    print("Loading args from json file")
    summary_filename = json_file
    
    with open(summary_filename) as f:
        args_json = json.load(fp=f)
    
    print("Ignoring", to_ignore)
    for arg in to_ignore:
        if arg in args_json:
            del args_json[arg]
    
    return args_json
    
        
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
    Sorts out dataset_args
    """
    print("Getting data: {}".format(args.dataset))
    
    if args.dataset not in DATASETS:
        raise Exception("Dataset '{0}' does not exist! ".format(args.dataset))
        
    # Set dataset seed
    if 'seed' not in args['dataset_args'] or args['dataset_args']['seed'] < 0:
        args['dataset_args']['seed'] = args.seed
    
    # Update parameters
    args.dataset_args = get_full_dataset_args(
        args.dataset,
        convert_to_nested_dict(args.dataset_args)
    )
    
    # Update parameters 2nd time if _dataset_args in args
    if "_dataset_args" in args:
        args.dataset_args = get_full_dataset_args(
            args.dataset,
            convert_to_nested_dict(args._dataset_args),
            default_args = args.dataset_args
        )
        del args._dataset_args
        
    args.dataset_args = toBunch(args.dataset_args)
    datasets = DATASETS[args.dataset](args.dataset_args)
    
    if (args.conventional_split is None and \
        args.model in ["baseline", "baselinepp", "knn"]):
        args.conventional_split = True
    
    # Preps datasets if they are imbalanced, 
    # Also if specified, joins training and validation classes into one set, and performs a conventional split of images 
    datasets = prep_datasets(datasets, args, conventional_split=args.conventional_split,
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
    model_name = args["model"]
    task_name = args["task"]
    tasks = get_task_classes(task_name, model_name)
    
    # Update task_args
    args.task_args = get_full_task_args(
        tasks,
        convert_to_nested_dict(args.task_args)
    )
    
    # Update task_args 2nd time if _task_args in args
    if "_task_args" in args:
        args.task_args = get_full_task_args(
            tasks,
            convert_to_nested_dict(args._task_args),
            default_args = args.task_args
        )
        del args._task_args
    
    args.task_args = toBunch(args.task_args)
    
    return tasks


def get_backbone(args, device):
    """
    Returns the backbone model
    """
    model_name = args["model"]
    backbone_name = args["backbone"]
    
    if backbone_name not in BACKBONES:
        raise Exception("Backbone not found: {}".format(backbone_name))
    
    if "relationnet" in model_name:
        if backbone_name == "Conv4":
            return backbones.Conv4NP(device, outdim=args.backbone_channel_dim)
        if backbone_name == "Conv4S":
            return backbones.Conv4SNP(device, outdim=args.backbone_channel_dim)
        if backbone_name == "Conv6":
            return backbones.Conv6NP(device, outdim=args.backbone_channel_dim)
    
    if model_name in ["maml", "protomaml", "oml"]:
        return BACKBONES[backbone_name](device, maml=True, outdim=args.backbone_channel_dim)
    
    if model_name in ["btaml"] and backbone_name in ["Conv4", "Conv6"]:
        args.backbone_channel_dim = 32
        return BACKBONES[backbone_name](device, maml=True, outdim=args.backbone_channel_dim)
        
    return BACKBONES[backbone_name](device, outdim=args.backbone_channel_dim)


def get_model(backbone, tasks, datasets, stategy, args, device):
    """
    Returns FSL model and sorts out model_args
    """
    model_name = args["model"]
    task_name = args["task"]
    print("Getting model: {}".format(model_name))
    
    if model_name not in MODELS:
        raise Exception("Model {} does not exist".format(model_name))

    if model_name in ["baseline", "baselinepp", "maml", "gpshot", "protomaml", "knn", "simpleshot",
                     "bmaml", "bmaml_chaser", "btaml", "btaml_star"]:
        output_dims = dict()
        for s in ["train", "val", "test"]:
            output_dims[s] = tasks[s].get_output_dim(args.task_args[s], datasets[s])
        args.model_args['output_dim'] = output_dims
    
    if model_name in ['simpleshot'] and args.model_args['disable_tqdm'] is None:
        args.model_args['disable_tqdm'] = args['disable_tqdm']
        
    if model_name in ['btaml'] and args.model_args['max_shot'] == -1:
        if args.task in ['fsl']:
            maxshot = max([args.task_args[setname].num_supports for setname in ['train', 'test', 'val']])
        if args.task in ['fsl_imbalanced']:
            maxshot = max([args.task_args[setname].max_num_supports for setname in ['train', 'test', 'val']])
        args.model_args['max_shot'] = maxshot
    
    # Update model_args
    args.model_args = get_full_model_args(
        model_name,
        convert_to_nested_dict(args.model_args),
    )
    
    # Update 2nd time if _model_args in args
    if "_model_args" in args:
        args.model_args = get_full_model_args(
            model_name,
            convert_to_nested_dict(args._model_args),
            default_args= args.model_args
        )
        del args._model_args
        
    args.model_args = toBunch(args.model_args)
    model = MODELS[model_name](backbone, stategy, args.model_args, device)
    model.setup_model()
    
    return model.to(device)
    
                  
def sortout_ptracker_args(args):
                  
    # Update ptracker_args parameters
    args.ptracker_args = get_full_ptracker_args(convert_to_nested_dict(args.ptracker_args))
    
    # Update ptracker_args parameters 2nd time if _dataset_args in args
    if "_ptracker_args" in args:
        args.ptracker_args = get_full_ptracker_args(
            convert_to_nested_dict(args._ptracker_args),
            default_args = args.ptracker_args
        )
        del args._ptracker_args
        
    return args
                  
                  
def get_strategy(args, device):
    print("Getting strategy: {}".format(args.strategy))
    
    if args.strategy not in STRATEGIES:
        raise Exception("Ups. Imbalance strategy {} does not exist!".format(args.strategy))
    
    if args.strategy in ['ros_aug', 'ros_aug2']:
        args.mono = (args.dataset in ['sss', 'sss_mini'])
    
    # Update parameters
    args.strategy_args = get_full_strategy_args(
        args.strategy,
        convert_to_nested_dict(args.strategy_args)
    )
    
    # Update parameters 2nd time if _strategy_args in args
    if "_strategy_args" in args:
        args.strategy_args = get_full_strategy_args(
            args.strategy,
            convert_to_nested_dict(args._strategy_args),
            default_args = args.strategy_args
        )
        del args._strategy_args
        
    args.strategy_args = toBunch(args.strategy_args)
    strategy = STRATEGIES[args.strategy](args.strategy_args, device, args.seed)
    
    return strategy
    
def expand_and_update_args(default_args, args, suppressed=True):
    """
    Updated default_args with hyperparameters in args
    param default_args: should a dict(test=dict(), train=dict(), val=dict()) to update, containing full default arguments
    """
    
    # Place args in buckets such that they can be sorted out properly according to bucket rank
    args_in_buckets = dict(test=dict(), train=dict(), val=dict(), eval=dict(), trval=dict(),_other=dict())
    update_dict(args_in_buckets, args)
    for param in list(args_in_buckets.keys()): # any hyperparams without an unassigned bucket go into '_other'
        if param not in ["train", "test", "val", "eval", "trval", "_other"]:
            args_in_buckets["_other"][param] = args_in_buckets[param]
            del args_in_buckets[param]
    
    # Mapping between buckets and setnames eg <param> in args['eval'][<param>] goes to args['test'] and args['val'] 
    buckets_to_setnames = {
        "_other": ["train", "test", "val"], 
        "trval": ["train", "val"],
        "eval": ["test", "val"], 
        "test": ["test"],
        "val": ["val"],
        "train": ["train"], 
    }    
    # List order defines bucket rank. Gives priority to the more specific bucket, eg eval < val
    bucket_order = ['_other', 'trval', 'eval', 'test', 'val', 'train']
    
    # Goes through each bucket in turn, adding to default_args from args_in_buckets, while taking order into account
    for bucket in bucket_order:
        setnames = buckets_to_setnames[bucket]
        
        # For each hyperparam in a bucket, sort out any clashes based on bucket order
        for hyperparam in args_in_buckets[bucket]:
            
            # Detect if the hyperparam is missing from the default_args
            missing_from = [s for s in setnames if hyperparam not in default_args[s]]
            if len(missing_from)!=0 and not suppressed:
                print("#\t Hyperparameter args.{}.{} not found in default_args.{}".format(
                    bucket, hyperparam, '|'.join(missing_from)))
            
            # Detect if the hyperparam is defined in another bucket of a lower order with overlapping setnames
            bucket_clushes = []
            for other_bucket in bucket_order:
                if not suppressed and hyperparam in args_in_buckets[other_bucket] and \
                   bucket_order.index(other_bucket) < bucket_order.index(bucket) and \
                   not set(buckets_to_setnames[bucket]).isdisjoint(set(buckets_to_setnames[other_bucket])):
                    bucket_clushes.append(other_bucket)
                    
            if len(bucket_clushes) > 0:
                other_bucket = bucket_clushes[-1]  # highest order bucket
                if not suppressed:
                    print("#\t Overwriting args{}.{} = {} \t with args{}.{} = {} ".format(
                       '' if other_bucket == '_other' else ".{}".format(other_bucket), 
                        hyperparam, args_in_buckets[other_bucket][hyperparam], 
                        '' if bucket == '_other' else ".{}".format(bucket),
                        hyperparam, args_in_buckets[bucket][hyperparam]))
                
            # update parameter
            for s in setnames:
                if hyperparam in default_args[s]:
                    default_args[s][hyperparam] = args_in_buckets[bucket][hyperparam]
                elif not suppressed:
                    print("#\t Ignoring hyperparameter args.{}.{}".format(s, hyperparam))
    
    return default_args


def convert_to_nested_dict(params, upto_level=None):
    """
    Splits keys containing '.', and converts into a nested dict
    """
    combined = dict()
    for key, value in list(params.items()):
        dict_item = nested_item(key.split("."), value, upto_level)
        combined = update_dict(combined, dict_item)
    return combined

def nested_value(v, upto_level=None):
    if isinstance(v, collections.abc.Mapping):
        return convert_to_nested_dict(v, upto_level)
    else:
        return v

def nested_item(keylist, value, upto_level=None):
    """
    Recursive method for converting into a nested dict
    Splits keys containing '.', and converts into a nested dict
    """
#     print(keylist, value)
#     if value == ['accuracy', 'loss', 'per_cls_stats']:
#         import pdb; pdb.set_trace();

    if upto_level is None: upto_level = len(keylist)
    
    if len(keylist) == 0:
        return nested_value(value)
    
    if len(keylist) == 1 or upto_level <= 0:
        key = '.'.join(keylist)
        base = dict()
        base[key] = nested_value(value)
        return base
    
    else:
        key = keylist[0]
        value = nested_item(keylist[1:], value, upto_level-1)
        base = dict()
        base[key] = nested_value(value)
        return base


def update_dict(base, to_update):
    """
    Updates a nested dict
    """
    if base is None:
        return to_update
    
    for k, v in to_update.items():
        if isinstance(v, collections.abc.Mapping):
            base[k] = update_dict(base.get(k, {}), v)
        else:
            base[k] = v
    return base

def get_default_parser_args(parser):
    """
    Returns default args of a given parser
    """
    args = parser.parse_args([])
    args_vars = vars(args)
    return args_vars


# def get_full_strategy_args(strategy, args, default_args=None):
#     """
#     Gets default ptracker args and updates with given ptracker args
#     """
#     if strategy is None:
#         return dict()
    
#     if default_args is None:
#         _default_args = get_default_parser_args(STRATEGIES[strategy].get_parser())
#         default_args = dict()
#         default_args["train"] = copy.copy(_default_args)
#         default_args["val"] = copy.copy(_default_args)
#         default_args["test"] = copy.copy(_default_args)
#     return expand_and_update_args(default_args, args)

def get_full_strategy_args(strategy, args, default_args=None, suppressed=True):
    """
    Gets default model args and updates with given model args
    Note: model_args do not contain args for separate stages, this is left for model implementation
    """
    if strategy is None:
        return dict()

    if default_args is None:
        default_args = get_default_parser_args(STRATEGIES[strategy].get_parser(argparse.ArgumentParser()))
    
    # Update default_args
    for key, value in args.items():
        if key in default_args:
            default_args[key] = value # updating value
        elif not suppressed:
            print("#\t Key not found in the model_args {}".format(key))
    
    return default_args

def get_full_ptracker_args(args, default_args=None):
    """
    Gets default ptracker args and updates with given ptracker args
    """
    if default_args is None:
        _default_args = get_default_parser_args(PerformanceTracker.get_parser())
        default_args = dict()
        default_args["train"] = copy.copy(_default_args)
        default_args["val"] = copy.copy(_default_args)
        default_args["test"] = copy.copy(_default_args)
        
    return expand_and_update_args(default_args, args)

def get_full_dataset_args(dataset, args, default_args=None):
    """
    Gets default dataset args and updates with given dataset args
    """
    if default_args is None:
        _default_args = get_default_parser_args(get_dataset_parser(dataset))
        default_args = dict()
        default_args["train"] = copy.copy(_default_args)
        default_args["val"] = copy.copy(_default_args)
        default_args["test"] = copy.copy(_default_args)
    return expand_and_update_args(default_args, args)


def get_full_task_args(tasks, args, default_args=None):
    """
    Gets default task args and updates with given task args
    """
    if default_args is None:
        default_args = dict()
        default_args["train"] = get_default_parser_args(tasks["train"].get_parser(argparse.ArgumentParser()))
        default_args["val"] = get_default_parser_args(tasks["val"].get_parser(argparse.ArgumentParser()))
        default_args["test"] = get_default_parser_args(tasks["test"].get_parser(argparse.ArgumentParser()))
    return expand_and_update_args(default_args, args)


def get_full_model_args(model, args, default_args=None, suppressed=True):
    """
    Gets default model args and updates with given model args
    Note: model_args do not contain args for separate stages, this is left for model implementation
    """
    if default_args is None:
        default_args = get_default_parser_args(MODELS[model].get_parser(argparse.ArgumentParser()))
    
    # Update default_args
    for key, value in args.items():
        if key in default_args:
            default_args[key] = value # updating value
        elif not suppressed:
            print("#\t Key not found in the model_args {}".format(key))
    
    return default_args

def compress(args_dict):
    """
    Compresses an args_dict into a more compact form of dictionary
    The args_dict should contain 'train', 'test', 'val' keys 
    Returns 'train.<param>' type of notation
    """
    
    similarity_args = defaultdict(lambda : list())
    
    setnames = ["train", "test", "val"]
    if not all([(s in args_dict) for s in setnames]):
        return args_dict
    
    for i, s in enumerate(setnames):
        args = args_dict[s]
        for k,v in args.items():
            if type(v) is list:
                v = tuple(v)
            similarity_args[(k,v)].append(s)
    
    compressed_args = dict()
    for k_v_pair, setnames in similarity_args.items():
        k, v = k_v_pair
        if len(setnames) == 2 and "test" in setnames and "val" in setnames:
            s = "eval"
            if s not in compressed_args: compressed_args[s] = dict()
            compressed_args[s][k] = v
            
        elif len(setnames) == 2 and "train" in setnames and "val" in setnames:
            s = "trval"
            if s not in compressed_args: compressed_args[s] = dict()
            compressed_args[s][k] = v
        
        elif len(setnames) == 3:
            compressed_args[k] = v
            
        else:
            for s in setnames:
                if s not in compressed_args: compressed_args[s] = dict()
                compressed_args[s][k] = v
     
    return compressed_args


def compress_and_print(args):
    """
    Prints args in a more compact form
    """
    temp_args = copy.deepcopy(args)
    temp_args = toDict(temp_args)
    temp_args['model_args'] = compress(temp_args['model_args'])
    temp_args['dataset_args'] = compress(temp_args['dataset_args'])
    temp_args['task_args'] = compress(temp_args['task_args'])
    temp_args['ptracker_args'] = compress(temp_args['ptracker_args'])
    temp_args['strategy_args'] = compress(temp_args['strategy_args'])
    
    print(" ---------- FULL ARGS (COMPACT) ---------")
    pprint.pprint(temp_args, indent=2)
    print(" ----------------------------------------")

    
    
def str2bool(v):
    """
    Acceptable boolean type passed through stdin
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
"""
str2bool_args is a shortcut args list for str2bool type 
"""
str2bool_args = dict( type=str2bool, nargs='?', const=True )        


class Bunch(object):
    """
    Bunch object made for convenience allows to be accessed like as dictionary and an object 
    """
    
    def __init__(self, adict):
        super().__init__()
        update_dict(self.__dict__, adict)
    def update_dict(self, adict):
        update_dict(self.__dict__, adict)
    def __contains__(self, arg):
        return arg in self.__dict__
    def __repr__(self):
        return json.dumps(self.__dict__, indent =2, default= lambda o: o.__dict__)
    def __getitem__(self, key):
         return self.__dict__[key]
    def __setitem__(self, key, value):
         self.__dict__[key] = value
        
        
def toBunch(d, nested_only=False):
    """
    Converts a nested dictionary to a nested Bunch object
    param nested_only: only converts the nested dict, leaves the outer dict as dict
    """
    if isinstance(d, Bunch): return d
    new_d = {}
    for key, value in list(d.items()):
        if isinstance(value, collections.abc.Mapping):
            new_d[key] = toBunch(value)
        else:
            new_d[key] = value
    new_d = new_d if nested_only else Bunch(new_d)
    return new_d


def toDict(bunch):
    """
    Converts a nested bunch to a nested dict
    """
    new_d = {}
    adict = bunch.__dict__ if isinstance(bunch, Bunch) else bunch
    for key, value in list(adict.items()):
        if isinstance(value, collections.abc.Mapping) or isinstance(value, Bunch):
            new_d[key] = toDict(value)
        else:
            new_d[key] = value
    return new_d


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


class PrintHelpAction(argparse.Action):
    
    def __call__(self, parser, namespace, values, option_string=None):
        arg_vars = vars(namespace)
        model =  arg_vars["model"]
        task = arg_vars["task"]
        strategy =arg_vars["strategy"]
        dataset = arg_vars["dataset"]
        
        print_model = (option_string == '--help_all') | (option_string == '--help_model')
        print_dataset = (option_string == '--help_all') | (option_string == '--help_dataset')
        print_strategy = (option_string == '--help_all') | (option_string == '--help_strategy')
        print_task = (option_string == '--help_all') | (option_string == '--help_task')
        print_ptracker = (option_string == '--help_all') | (option_string == '--help_ptracker')
        
        if (option_string == '--help_all'):
            parser.print_help()
        
        print("\n")
        print('',"-"*87)
        print("\t\t\t\tJSON parsable ARG OPTIONS")
        print('',"-"*87)
        print('',"-"*87)
        print("|","Note: ", "\t\t\t\t\t\t\t\t\t\t|")
        print("|","\t'3-phase'", "args means that three separate copies of args are generated - one for", "|")
        print("|","\t", "each of the three different phases of the experiment: train, val, test).", "\t|")
        print("|","\t", "Use the syntactic sugar for easy differentiation between phases:", "\t\t|")
        print("|","\t\t", '\'{"{train,val,test,trval,eval}.[ARG1]":[VALUE1], ... }\'', "\t\t|")
        print('',"-"*87)
        print("\n")
        
        usage_args = '\'{"[ARG1]":[VALUE1], ... }\''
        
        if print_ptracker:
            print("PTRACKER_ARGS (3-phase):")
            helpstr = PerformanceTracker.get_parser(
                parser=argparse.ArgumentParser(usage='%(prog)s --ptracker_args {}'.format(usage_args),
                                               formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                               add_help=False)).format_help()
            print('\t' + helpstr.replace('\n', '\n\t'))
            print("\n")
            
        if print_model:
            print("MODEL_ARGS for '{}' (NOT 3-phase):".format(model))
            helpstr = MODELS[model].get_parser(
                parser=argparse.ArgumentParser(usage='%(prog)s --model {} --model_args {}'.format(model, usage_args),
                                               formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                               add_help=False)).format_help()
            print('\t' + helpstr.replace('\n', '\n\t'))
            print("\n")
        
        if print_dataset:
            print("DATASET_ARGS for '{}' (3-phase)".format(dataset))
            helpstr = get_dataset_parser(dataset, 
                               parser=argparse.ArgumentParser(
                                   usage='%(prog)s --dataset {} --dataset_args {}'.format(dataset, usage_args),
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   add_help=False)).format_help()
            print('\t' + helpstr.replace('\n', '\n\t'))
            print("\n")
        
        if print_task:
            print("TASK_ARGS for '{}' (3-phase)".format(task))
            helpstr = TASKS[task].get_parser(parser=argparse.ArgumentParser(
                usage='%(prog)s --task {} --task_args {}'.format(task, usage_args),
                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                add_help=False)).format_help()
            print('\t' + helpstr.replace('\n', '\n\t'))
            print("\n")
        
        if print_strategy:
            print("STRATEGY_ARGS for '{}' (NOT 3-phase)".format(strategy))
            if strategy is not None:
                helpstr = STRATEGIES[strategy].get_parser(parser=argparse.ArgumentParser(
                    usage='%(prog)s --strategy {} --strategy_args {}'.format(strategy, usage_args), 
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                    add_help=False)).format_help()
                print('\t' + helpstr.replace('\n', '\n\t'))
            else:
                print("\tStrategy unspecified (no strategy will be applied).")
                print("\tSelect strategy OPTIONS: {}".format(list(STRATEGIES.keys())))
            print("\n")
        
        sys.exit(0)
