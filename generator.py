import sys
sys.path.insert(0,'./src/')

from utils.utils import MODELS, TASKS, DATASETS, get_main_parser, Bunch, toBunch, get_default_parser_args, get_task_classes, convert_to_nested_dict, update_dict, get_full_task_args, get_full_model_args, get_full_dataset_args, compress, get_full_ptracker_args, get_full_strategy_args, str2bool

import itertools
import os
import copy
import json
import pdb
import argparse
from functools import partial
from pprint import pprint
from collections import defaultdict
import string



class FormatDict(dict):
    # https://ideone.com/DZJO1I
    # https://stackoverflow.com/a/11284026
    def __missing__(self, key):
        return FormatPlaceholder(key)
    
class FormatPlaceholder:
    # https://ideone.com/DZJO1I
    # https://stackoverflow.com/a/11284026
    def __init__(self, key):
        self.key = key
    def __format__(self, spec):
        result = self.key
        if spec:
            result += ':' + spec
        return '{' + result + '}'
    def __getitem__(self, index):
        self.key = '{}[{}]'.format(self.key, index)
        return self
    def __getattr__(self, attr):
        self.key = '{}.{}'.format(self.key, attr)
        return self
    
class StringTemplate(object):
    
    def __init__(self, template):
        self.template = template

    def format(self, dict):
        formatter = string.Formatter()
        mapping = FormatDict(dict)
        return formatter.vformat(self.template, (), mapping)
    
    
def get_default_config():
    main_args = get_default_parser_args(get_main_parser())
    main_args['model_args'] = None
    main_args['task_args'] = None
    main_args['dataset_args'] = None
    main_args['ptracker_args'] = None
    main_args['strategy_args'] = None
    return convert_to_nested_dict(main_args)


def flatten_dict(d, prefix=None, seperator='.', value_map=lambda x: x):
    new_d = {}
    for key, value in list(d.items()):
        new_key = key if prefix is None else prefix + seperator + key
        if type(value) == dict:
            flattened = flatten_dict(value, new_key, seperator)
            new_d.update(flattened)
        else:
            new_d[new_key] = value
    return new_d
            
    
def substitute_hyperparameters(config, hyperparameters):
#     import pdb; pdb.set_trace()
    hyperparameters = convert_to_nested_dict(hyperparameters)
    dataset_args = hyperparameters['dataset_args'] if 'dataset_args' in hyperparameters else dict() 
    task_args = hyperparameters['task_args'] if 'task_args' in hyperparameters else dict() 
    model_args = hyperparameters['model_args'] if 'model_args' in hyperparameters else dict() 
    ptracker_args = hyperparameters['ptracker_args'] if 'ptracker_args' in hyperparameters else dict() 
    strategy_args = hyperparameters['strategy_args'] if 'strategy_args' in hyperparameters else dict() 
    
    task_name = hyperparameters['task'] if 'task' in hyperparameters else config['task']
    model_name = hyperparameters['model'] if 'model' in hyperparameters else config['model']
    dataset_name = hyperparameters['dataset'] if 'dataset' in hyperparameters else config['dataset']
    strategy_name = hyperparameters['strategy'] if 'strategy' in hyperparameters else config['strategy']
    
    tasks = get_task_classes(task_name, model_name)
    
    full_task_args = get_full_task_args(tasks, task_args, 
        default_args=config['task_args'])
    
    full_model_args = get_full_model_args(model_name, model_args, 
        default_args=config['model_args'])
    
    full_dataset_args = get_full_dataset_args(dataset_name, dataset_args, 
        default_args=config['dataset_args'])
    
    full_ptracker_args = get_full_ptracker_args(ptracker_args, 
        default_args=config['ptracker_args'])
    
    full_strategy_args = get_full_strategy_args(strategy_name, strategy_args, 
        default_args=config['strategy_args'])
    
    hyperparameters['model_args'] = full_model_args
    hyperparameters['task_args'] = full_task_args
    hyperparameters['dataset_args'] = full_dataset_args
    hyperparameters['ptracker_args'] = full_ptracker_args
    hyperparameters['strategy_args'] = full_strategy_args
    
    config = update_dict(config, hyperparameters)
    return config
        
def compress_config(config):
    config['task_args'] = compress(config['task_args'])
    config['dataset_args'] = compress(config['dataset_args'])
    config['ptracker_args'] = compress(config['ptracker_args'])
    config['strategy_args'] = compress(config['strategy_args'])
    return config

def unpack(comb):
    new_comb = {}
    for key, value in comb.items():
        if type(key) is tuple and type(value) is tuple:
            unpacked_dict = {a[0]:a[1] for a in zip(key, value)}
            new_comb.update(unpacked_dict)
        else:
            new_comb[key] = value
    return new_comb
    
    
def hyperparameter_combinations(variables):
    '''
    Generates all possible combinations of variables
    :param variables: dictionary mapping variable names to a list of variable values that vary between experiments
    :returns experiments: a list of dictionaries mapping variable names to singular variable values
    '''
    # https://codereview.stackexchange.com/a/171189
    keys, values = zip(*variables.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    unpacked_combinations = [unpack(comb) for comb in combinations]
    
    return unpacked_combinations


def get_python_script(exp_kwargs={}, bash=False):
    script = 'python {}src/main.py '.format(os.path.abspath(".")+"/" if bash else "")
    for arg in exp_kwargs:
        script += '--{} {} '.format(arg, exp_kwargs[arg])
    return script


def get_bash_script(exp_kwargs={}):
    script = 'export CUDA_VISIBLE_DEVICES=$1; \n'
    exp_kwargs['gpu'] = 0
    script += get_python_script(exp_kwargs, bash=True) + ' $2'
    return script
    

def generate_experiments(experiment_name_template, variables, default_config, args,
                        config_name='config', script_name='script', log_name='log', save=True):
                
    if args.dummy_run:
        variables.update({
                    'num_epochs': [3],
                    'num_tasks_per_epoch': [3],
                    'num_tasks_per_validation': [3],
                    'num_tasks_per_testing': [3]
        })
        if ('model' in variables and variables['model'] == 'simpleshot') or \
           ('model' in default_config and default_config['model'] == 'simpleshot'):
            variables['model_args.approx_train_mean'] = [True]
    combinations = hyperparameter_combinations(variables)
    
    scripts = []
    configs = []
    script_paths = []
    config_paths = []
    
    def value_map(x):
        return int(x) if type(x) == bool else x
    
    if '{' in experiment_name_template and '}' in experiment_name_template:
        experiment_name_template = experiment_name_template.replace('.', '_') # dots not allowed in {} of str.format(...)
    
    for i_comb, hyperparameters in enumerate(combinations):
        gpu = args.gpu
        
        full_config = substitute_hyperparameters(default_config, hyperparameters)
        
        compressed_config = compress_config(copy.deepcopy(full_config))
        config = copy.deepcopy(compressed_config)
        
        
        if '{' in experiment_name_template and '}' in experiment_name_template:
            combined_config_flattened = {
                **flatten_dict(full_config, seperator='_', value_map=value_map),
                **flatten_dict(compressed_config, seperator='_', value_map=value_map)
            }
            experiment_name = experiment_name_template.format(**combined_config_flattened)
        else:
            experiment_name = experiment_name_template
        
        config['experiment_name'] = experiment_name
        config['task_args'] = flatten_dict(config['task_args'])
        config['dataset_args'] = flatten_dict(config['dataset_args'])
        config['ptracker_args'] = flatten_dict(config['ptracker_args'])
        config['strategy_args'] = flatten_dict(config['strategy_args'])
        
        config_path = os.path.join(args.results_folder, experiment_name, 'configs', '{}.json'.format(config_name))
        script_path = os.path.join(args.results_folder, experiment_name, 'scripts', '{}.sh'.format(script_name))
        output_path = os.path.join(args.results_folder, experiment_name, 'logs', '{}.txt'.format(log_name))
        
        if args.bash:
            config_path = os.path.abspath(config_path)
            script_path = os.path.abspath(script_path)
            output_path = os.path.abspath(output_path)
            script_content = get_bash_script(exp_kwargs={'args_file': config_path, 'gpu':gpu}) + '\n'
            script = 'bash {} {} '.format(script_path, gpu)
        else:
            script_content = get_python_script(exp_kwargs={'args_file': config_path, 'gpu':gpu})
            script = script_content
        
        if not args.no_log:
            script += ' &> ' + output_path
        
        if save:
            print(script)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            os.makedirs(os.path.dirname(script_path), exist_ok=True)
            
            with open(config_path,'w') as f:
                json.dump(config, f, indent=2)

            with open(script_path,'w') as f:
                f.write(script_content)
            
        scripts.append(script)
        script_paths.append(script_path)
        configs.append(config)
        config_paths.append(config_path)
    
    return zip(scripts, script_paths, configs, config_paths)        
        
        
def make_names(settings, way):
    test_names = []
    for setting in settings:
        min_k, max_k, minor, dist, t_min_k, t_max_k, t_minor, t_dist  = setting
        test_names.append('{}-{}shot_{}{}_{}-{}query_{}{}'.format(min_k, max_k, dist, 
                                                           "" if minor is None else '_{}minor'.format(int(minor*way)), 
                                                           t_min_k, t_max_k, t_dist, 
                                                           "" if t_minor is None else '_{}minor'.format(int(t_minor*way)), 
                                                          ))
    return test_names

    
def fsl_imbalanced(args, models=[], strategies=[], seeds=[], train_tasks=[], test_tasks=[], var_update={}, save=True, 
                   expfolder=''):
    
    n_way = 5
    dataset = 'mini'
    
    experiement_files = []
    for seed in seeds:
        for model in models:
            for train_task in train_tasks:
                train_name = make_names([train_task], n_way)[0]
#                 print(train_task)
#                 print(train_name)
                
                for strategy in strategies:
                    variables = {
                        'results_folder'             : [os.path.abspath(args.results_folder)],
                        'seed'                       : [seed],
                        'backbone'                   : ['Conv4'],
                        'num_epochs'                 : [200],
                        'num_tasks_per_epoch'        : [500],
                        'num_tasks_per_validation'   : [200],
                        'num_tasks_per_testing'      : [600],
                        'strategy'                   : [strategy],
                        'model'                      : [model],
                        'model_args.lr'              : [0.001], 
                        'model_args.lr_decay'        : [0.1],
                        'model_args.lr_decay_step'   : [100],
                        'task'                       : ['fsl_imbalanced'],
                        'task_args.batch_size'       : [1],
                        'task_args.num_classes'      : [n_way],
                        'dataset'                    : [dataset],
                        'dataset_args.data_path'     : [args.data_path],
                        'dataset_args.train.aug'     : [True],
                        'ptracker_args.test.metrics' : [['accuracy', 'loss', 'per_cls_stats']]
                    }
                    
                    variables[('task_args.min_num_supports', 
                               'task_args.max_num_supports',
                               'task_args.num_minority',
                               'task_args.imbalance_distribution', 
                               'task_args.min_num_targets', 
                               'task_args.max_num_targets',
                               'task_args.num_minority_targets',
                               'task_args.imbalance_distribution_targets')] = [train_task]
                    
                    if len(test_tasks) > 0:
                        variables[('task_args.test.min_num_supports', 
                                   'task_args.test.max_num_supports',
                                   'task_args.test.num_minority',
                                   'task_args.test.imbalance_distribution', 
                                   'task_args.test.min_num_targets', 
                                   'task_args.test.max_num_targets',
                                   'task_args.test.num_minority_targets',
                                   'task_args.test.imbalance_distribution_targets')] = test_tasks
                    
                    # experiment path
                    expath = expfolder + '{dataset}/{backbone}/train_on_' + train_name + '/{strategy}/{model}/'
#                             '{num_epochs}epochs_{num_tasks_per_epoch}tasks/'
                    
                    if model in ['protonet', 'protodkt']:
#                         variables[(
#                             'task_args.train.num_classes', 
#                             'task_args.train.min_num_targets',
#                             'task_args.train.max_num_targets')] = [
# #                                                                 (20, max(1,int(target_imbalance[0]/3)),
# #                                                                  int(target_imbalance[1]/3)), 
#                                                                 (5, 15, 15)
#                                                               ]
                        expath += '{task_args.train.num_classes}trainway/'

                    elif model in ['baseline', 'baselinepp', 'knn']:
                        variables['task_args.trval.batch_size'] = [128]  # train and validation batch
#                         expath += '{task_args.train.batch_size}trainbatch_' + \
#                                   '{task_args.val.batch_size}valbatch'

                    elif model in ['maml', 'protomaml']:
                        variables['model_args.batch_size'] = [4] if model == 'maml' else [1]
                        variables['model_args.inner_loop_lr'] = [0.1]
                        variables['model_args.num_inner_loop_steps'] = [10]
#                         expath += '{model_args.batch_size}trainbatch_'+ \
#                                   '{model_args.inner_loop_lr}innerlr_' + \
#                                   '{model_args.num_inner_loop_steps}innersteps'

                    elif model in ['bmaml', 'bmaml_chaser']:
                        variables['model_args.batch_size'] = [1]
                        variables['model_args.inner_loop_lr'] = [0.1]
                        variables['model_args.num_inner_loop_steps'] = [1]
                        variables['model_args.num_draws'] = [20]
#                         expath += '{model_args.batch_size}trainbatch_'+ \
#                                   '{model_args.inner_loop_lr}innerlr_' + \
#                                   '{model_args.num_inner_loop_steps}innersteps_'+\
#                                   '{model_args.num_draws}draws'

                        if model == 'bmaml_chaser':
                            variables['model_args.leader_inner_loop_lr'] = [0.5]
#                             expath += '_{model_args.leader_inner_loop_lr}leaderlr'
#                         expath += '/'
                        
                    elif model == 'btaml':
                        variables['backbone_channel_dim'] = [64]
                        variables['model_args.lr'] = [0.0001]
                        variables[('model_args.approx','model_args.approx_until')] = [(True,50),(True,25),(False,0)]
                        variables['model_args.batch_size'] = [4]
                        variables['model_args.inner_loop_lr'] = [0.1]
                        variables['model_args.num_inner_loop_steps'] = [10]
                        variables[('model_args.alpha_on', 
                                   'model_args.omega_on',
                                   'model_args.gamma_on',
                                   'model_args.z_on')] = [(True, True, True, True)]
                        expath += "{model_args.alpha_on}a_{model_args.omega_on}o_" + \
                                   "{model_args.gamma_on}g_" + \
                                   "{model_args.z_on}z_{model_args.approx_until}till/"
#                         expath += '{model_args.batch_size}trainbatch_'+ \
#                                   '{model_args.inner_loop_lr}innerlr_' + \
#                                   '{model_args.num_inner_loop_steps}innersteps/'
                    
                    elif model == 'relationnet':
                        variables['model_args.loss_type'] = ['softmax']
                        expath += '{model_args.loss_type}/'

                    elif model == 'simpleshot':
                        variables['task_args.train.batch_size'] = [128]
                        variables['model_args.feat_trans_name'] = ['CL2N']
                        variables['model_args.train_feat_trans'] = [False]
                        variables['model_args.approx_train_mean'] = [False]  # if true will speed up dataset mean calculation
#                         expath += '{task_args.train.batch_size}trainbatch_' + \
#                                   '{model_args.feat_trans_name}/'
                        
                    expath += '{seed}/'
                    variables.update(var_update)
                    
#                     print(expath)
                    
#                     print(variables)
                    
                    config = get_default_config()
                    expfiles = generate_experiments(expath, variables, config, args, save=save)
                    experiement_files.extend(expfiles)
    return experiement_files


def imbalanced_task_test(args, expfiles):
    
    n_way=5
    test_settings = [
        (5, 5,  None, 'balanced'),  # K_min, K_max, N_min, I-distribution 
        (1, 9,  None, 'random'),
#         (3, 7,  None, 'linear'),
#         (1, 9,  None, 'linear'),
        (1, 9,  0.2,  'step')       # N_min expressed as a fraction of 'n_way'
    ]
    target_imbalance = [
        (15, 15,  None, 'balanced')
    ]
    test_names = make_names(test_settings, n_way)
    
    for experiment in expfiles:
        
        default_config = get_default_config()
        script, script_path, config, config_path = experiment
        
        # substitute for backward compatibility so new code versions work on old configs
        default_config = substitute_hyperparameters(default_config, config) 
        
        assert default_config['task'] == 'fsl_imbalanced'
        
        for t, test_setting in enumerate(test_settings):
            test_name = test_names[t]
            min_k, max_k, minor, dist = test_setting
            
            variables = {
                'continue_from' :                        ['best'],
                'evaluate_on_test_set_only':             [True],
                'test_performance_tag':                  [test_name],
                'task_args.test.num_classes':            [n_way],
                'task_args.test.min_num_supports':       [min_k],
                'task_args.test.max_num_supports':       [max_k],
                'task_args.test.num_minority':           [minor],
                'task_args.test.imbalance_distribution': [dist] 
            }
            
            variables[('task_args.test.min_num_targets', 
                       'task_args.test.max_num_targets',
                       'task_args.test.num_minority_tagets',
                       'task_args.test.imbalance_distribution_targets')] = target_imbalance
            
            generate_experiments(
                default_config['experiment_name'], 
                variables, 
                default_config,
                args,
                save=True,
                config_name='config_{}'.format(test_name),
                script_name='script_{}'.format(test_name),
                log_name='log_{}'.format(test_name)
            )
    
            
def strategy_inference(args, expfiles):
    n_way=5
    test_settings = [  
        (1, 9,  None, "linear"),
        (2, 8,  None, "linear"),
        (3, 7,  None, "linear"),
        (4, 6,  None, "linear"),
        (5, 5,  None, 'linear'),  # K_min, K_max, N_min, I-distribution    
        
#         (1, 9,  None, 'random'),
#         (4, 6,  None, 'linear'),
#         (1, 9,  0.2,  'step')       # N_min expressed as a fraction of 'n_way'
    ]
    test_names = make_names(test_settings, n_way)
    
    # Inference Strategies
    strategies = [
#         'ros',
#         'ros_aug',
#         'weighted_loss',
#         'focal_loss',
        'cb_loss',
    ]
    
    for experiment in expfiles:
        for strategy in strategies:
            default_config = get_default_config()
            script, script_path, config, config_path = experiment
            
            config['strategy'] = strategy
            config['strategy_args'] = {}
            
            default_config = substitute_hyperparameters(default_config, config) 

            assert default_config['task'] == 'fsl_imbalanced'
            assert default_config['strategy'] == strategy
            
            continue_from = os.path.join(args.results_folder, default_config['experiment_name'])
            
            for t, test_setting in enumerate(test_settings):
                test_name = test_names[t]
                min_k, max_k, minor, dist = test_setting

                variables = {
                    'continue_from' :                        [continue_from],
                    'evaluate_on_test_set_only':             [True],
                    'test_performance_tag':                  [test_name],
                    'task_args.test.num_classes':            [n_way],
                    'task_args.test.num_targets':            [16],
                    'task_args.test.min_num_supports':       [min_k],
                    'task_args.test.max_num_supports':       [max_k],
                    'task_args.test.num_minority':           [minor],
                    'task_args.test.imbalance_distribution': [dist]
                }
                expath = 'strategy_inference/{strategy}/'

                if strategy == 'cb_loss':
                    variables['strategy_args.beta'] = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99]
                    expath += "{strategy_args.beta}beta/"
                
                expath += default_config['experiment_name']
                    
                generate_experiments(
                    expath,
                    variables, 
                    default_config,
                    args,
                    save=True,
                    config_name='config_{}'.format(test_name),
                    script_name='script_{}'.format(test_name),
                    log_name='log_{}'.format(test_name)
                )
            
                 
def imbalanced_dataset(args, models=[], seeds=[], save=True):
    
    # meta-training dataset imbalance settings
    imbalance_settings = [
        (300, 300, None, 'balanced'),
        (30 , 570, None, 'random'),
        (30 , 570, None, 'linear'),
        (30 , 570, 0.5,  'step'),
    ]
    
    strategies=[None]
    train_tasks=[(5, 5, None, 'balanced')]
    var_update = {'num_epochs': [200], 'num_tasks_per_epoch': [250]}
    
    is_baseline = lambda x: x in ['baseline', 'baselinepp', 'knn']
    
    experiement_files = []
    for experiment in fsl_imbalanced(args, models=models, strategies=strategies, seeds=seeds, var_update=var_update,
                          train_tasks=train_tasks, save=False):
        
        script, script_path, config, config_path = experiment
        default_config = get_default_config()
        default_config = substitute_hyperparameters(default_config, config)
        model = default_config['model']
        
        for setting in imbalance_settings:
            min_s, max_s, minor, dist  = setting
            
            variables = {
                'dataset_args.train.min_num_samples'       :[min_s],
                'dataset_args.train.max_num_samples'       :[max_s],
                'dataset_args.train.num_minority'          :[minor],
                'dataset_args.train.imbalance_distribution':[dist],
                'conventional_split'                       :[is_baseline(model)],
                'conventional_split_from_train_only'       :[is_baseline(model)]
            }
            
            expath = default_config['experiment_name']
            
            experiement_files.extend(generate_experiments(
                expath, 
                variables, 
                default_config,
                args,
                save=save
            ))
    return experiement_files

            
def cub_inference(args, expfiles, save=True):
    
    datasets = [
        'mini_to_cub'
    ]
    
    for dateset in datasets:
        
        for experiment in expfiles:
            script, script_path, config, config_path = experiment
            default_config = get_default_config()
            default_config = substitute_hyperparameters(default_config, config)
            continue_from = os.path.join(args.results_folder, default_config['experiment_name'])
            expath = 'cub_inference/{dataset}/' + default_config['experiment_name']
            
            variables = {
                'continue_from' : [continue_from],
                'dataset': [dateset],
                'evaluate_on_test_set_only': [True],
                'dataset_args.test.imbalance_distribution': [None],
            }

            generate_experiments(
                expath,
                variables, 
                default_config,
                args,
                save=True
            )
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='{gpu}', help='GPU ID')
    parser.add_argument('--dummy_run', type=str2bool, nargs='?', const=True, default=False,
                        help='Produces scripts as a "dry run" with a reduced number of tasks, '
                             'and no mobel saving (useful for debugging)')
    parser.add_argument('--data_path', type=str, default='/media/disk2/mateusz/data/pkl/',
                        help='Folder with data')
    parser.add_argument('--results_folder', type=str, default='./experiments/',
                        help='Folder for saving the experiment config/scripts/logs into')
    parser.add_argument('--imbalanced_task', type=str2bool, nargs='?', const=True, default=False,
                        help='Generate imbalanced support set experiments')
    parser.add_argument('--imbalanced_targets', type=str2bool, nargs='?', const=True, default=False,
                        help='Generate imbalanced target set experiments')
    parser.add_argument('--imbalanced_dataset', type=str2bool, nargs='?', const=True, default=False,
                        help='Generate imbalanced dataset experiments')
    parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=False,
                        help='Generate imbalanced test cases for scenarios')
    parser.add_argument('--no_log', type=str2bool, nargs='?', const=True, default=False,
                        help='Output won''t get redirected to logs')
    parser.add_argument('--inference', type=str2bool, nargs='?', const=True, default=False,
                        help='Generate inference tests i.e, eval with ROS+ for task-level; eval on CUB for dataset-level')
    parser.add_argument('--bash', type=str2bool, nargs='?', const=True, default=False,
                        help='Prints bash scripts instead of python')
    parser.add_argument('--minimal', type=str2bool, nargs='?', const=True, default=False,
                        help='Generate minimal experiments.')
    args = parser.parse_args()
    
    args.results_folder = os.path.abspath(args.results_folder)
    
    models = [
        'protonet',
        'relationnet',
#         'matchingnet',
        'gpshot',
#         'simpleshot',
        'baseline',
        'baselinepp',
#         'knn',
        'maml',
        'protomaml',
        'bmaml',
#         'bmaml_chaser',
#         'protodkt',
#         'btaml',  # -- left out due to an implementation error
    ]
    
    strategies = [
        None,
#         'ros',
#         'ros_aug',
#         'focal_loss',    # -- left for anyone to try 
#         'weighted_loss',  # -- left for anyone to try 
#         'cb_loss'
    ]
    
    seeds = [
        0,
#         1, 
#         2
    ]
    
    balanced_tasks = [
        (5, 5, None, 'balanced', 15, 15, None, 'balanced')
    ]
    
    imbalanced_tasks = [
        (1, 9, None, 'linear', 15, 15, None, 'balanced')
#         (1, 9, None, 'random')
#         (1, 9, None, 'linear'), 
#         (3, 7, None, 'linear'), 
#         (1, 9, 0.2, 'step'),
#         (1, 9, 0.8, 'step')
    ]
    
    if args.minimal:
        models = models[:2]
        strategies = strategies[:2]
        seeds = seeds[:1]
        
        
    if args.imbalanced_task:
        # Standard meta-training
        standard_expfiles = fsl_imbalanced(args, models=models, strategies=[None], seeds=seeds, train_tasks=balanced_tasks,
                                save=not (args.test or args.inference), expfolder='imbalanced_task/')
        # Random Shot meta-training
        randomshot_expfiles = fsl_imbalanced(args, models=models, strategies=strategies, seeds=seeds, 
                                             train_tasks=imbalanced_tasks,  save=not (args.test or args.inference), 
                                             expfolder='imbalanced_task/')
        
        if args.test: 
            imbalanced_task_test(args, standard_expfiles)
            imbalanced_task_test(args, randomshot_expfiles)
            
        if args.inference:
            strategy_inference(args, standard_expfiles)
            strategy_inference(args, randomshot_expfiles)
    
    if args.imbalanced_targets:
        # Standard meta-training
        train_tasks = [
            (5, 5, None, 'balanced', 5, 5, None, 'balanced'),
            (5, 5, None, 'balanced', 1, 9, None, 'linear'),
            (1, 9, None, 'linear', 5, 5, None, 'balanced'),
            (1, 9, None, 'linear', 1, 9, None, 'linear'),
        ]
        
        test_tasks = [
            (1, 9, None, 'linear', 5, 5, None, 'balanced')
        ]
        
        expfiles = fsl_imbalanced(args, models=models, strategies=[None], seeds=seeds, train_tasks=train_tasks, 
                                  test_tasks=test_tasks, save=not(args.test or args.inference), expfolder='imbalanced_targets/')
        
        if args.test:
            imbalanced_target_test(args, expfiles)
            
        if args.inference:
            print('Strategy inference for imbalnaced target set not yet implemented.')
    
    
    if args.imbalanced_dataset:
        expfiles = imbalanced_dataset(args, models=models, seeds=seeds, save=not (args.test or args.inference))
        
        if args.test:
            print('Balanced task testing is performed automatically after training. Use --inference to evaluate on CUB.')
        
        if args.inference:
            cub_inference(args,expfiles)
            
    
    if not (args.imbalanced_dataset or args.imbalanced_task or args.imbalanced_targets):
        print('Please specify --imbalanced_dataset or --imbalanced_task')
            
        
