import sys
sys.path.insert(0,'./src/')

from utils.utils import *
from utils.parser_utils import *

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

    
def substitute_hyperparams(hyperparams, config=None, suppress_warning=True):
    hyperparams = from_syntactic_sugar(copy.deepcopy(hyperparams))
    
    if config is None:
        combined_args, excluded_args, _ = get_args([], hyperparams)
    else:
        # Convert from three phase syntactic sugar format
        for three_phase_args in ['dataset_args', 'ptracker_args', 'task_args']:
            if three_phase_args in hyperparams:
                hyperparams[three_phase_args] = expand_three_phase(hyperparams[three_phase_args])
        
        combined_args, excluded_args  = update_dict_exclusive(config, hyperparams)
    
    if not suppress_warning and len(excluded_args) > 0:
        print("""excluded""")
        pprint(excluded_args)
        
    return combined_args


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
    
    assert all([isinstance(v, list) for v in values]), "All variable values should be contained in a list!" +\
                                                        " Put square parentheses, ie. '[' and ']', around the lonely value. " 
    
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    unpacked_combinations = [unpack(comb) for comb in combinations]
    
    return unpacked_combinations


def get_python_script(exp_kwargs={}):
    script = 'python src/main.py '
    for arg in exp_kwargs:
        script += '--{} {} '.format(arg, exp_kwargs[arg])
    return script


def get_bash_script(exp_kwargs={}):
    script = 'export CUDA_VISIBLE_DEVICES=$1; \n'
    exp_kwargs['gpu'] = 0
    script += get_python_script(exp_kwargs) + ' $2'
    return script
    
    
def value_map(x):
    return int(x) if type(x) == bool else x
    
def generate_experiments(name_template, 
                         variables,
                         g_args,
                         default_config=None,
                         config_name='config', 
                         script_name='script', 
                         log_name='log', 
                         save=True):
    
    global GPU_COUNTER  # used to evenly distribute jobs among the gpus (optional)
    ngpus = len(g_args.gpu)
    
    # If dry run
    if g_args.dummy_run:
        variables.update({
                    'num_epochs': [3],
                    'num_tasks_per_epoch': [3],
                    'num_tasks_per_validation': [3],
                    'num_tasks_per_testing': [3]
        })
        if ('model' in variables and variables['model'] == 'simpleshot') or \
           (default_config is not None and 'model' in default_config and default_config['model'] == 'simpleshot'):
            variables['model_args.approx_train_mean'] = [True]
    
    scripts = []
    configs = []
    script_paths = []
    config_paths = []
    
    # Iterate over variable combinations
    combinations = hyperparameter_combinations(variables)
    
    for i_comb, hyperparams in enumerate(combinations):
        
        # Generate full config
        full_config = substitute_hyperparams(hyperparams, default_config)
        
        # Generate a compressed version of config
        compressed_config = compress_args(full_config)
        
        # Flattened config for template name
        sperator = '_'
        flat_config = {
            **flatten_dict(full_config, seperator=sperator, value_map=value_map),
            **flatten_dict(compressed_config, seperator=sperator, value_map=value_map)
        }
        
        # Assign experiment_name
        experiment_name = name_template.replace('.', '_').format(**flat_config)
        full_config['experiment_name'] = compressed_config['experiment_name'] = experiment_name
        
        # Setup paths
        experiment_path = os.path.join(g_args.results_folder, experiment_name)
        config_path = os.path.join(experiment_path, 'configs', '{}.json'.format(config_name))
        script_path = os.path.join(experiment_path, 'scripts', '{}.sh'.format(script_name))
        output_path = os.path.join(experiment_path, 'logs', '{}.txt'.format(log_name))
        
        # Select gpu
        gpu = g_args.gpu[GPU_COUNTER % ngpus]
        GPU_COUNTER+=1
        
        # Run from .sh script file or directly using python
        if g_args.bash:
            config_path = os.path.abspath(config_path)
            script_path = os.path.abspath(script_path)
            output_path = os.path.abspath(output_path)
            script_content = get_bash_script(exp_kwargs={'args_file': config_path, 'gpu':gpu}) + '\n'
            script_command = 'bash {} {} '.format(script_path, gpu)
        else:
            script_content = get_python_script(exp_kwargs={'args_file': config_path, 'gpu':gpu})
            script_command = script_content
        
        # Save script content
        if save:
            if not g_args.no_log: script_command += ' &> ' + output_path
            print(script_command)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            os.makedirs(os.path.dirname(script_path), exist_ok=True)
            
            with open(config_path,'w') as f:
                json.dump(compressed_config, f, indent=2)
            
            with open(script_path,'w') as f:
                f.write(script_content)
        
        # Save scripts for reference
        scripts.append(script_command)
        script_paths.append(script_path)
        configs.append(compressed_config)
        config_paths.append(config_path)
    
    return zip(scripts, script_paths, configs, config_paths)        
        
        
def make_names(settings, way, shot_half=True, query_half=True):
    names = []
    for setting in settings:
        min_k, max_k, minor, dist, t_min_k, t_max_k, t_minor, t_dist  = setting
        name = ""
        if shot_half: 
            name += '{}-{}shot_{}{}'.format(min_k, max_k, dist, "" if minor is None else '_{}minor'.format(int(minor*way)))
        if query_half:
            name += '_{}-{}query_{}{}'.format(t_min_k, t_max_k, t_dist, 
                                                 "" if t_minor is None else '_{}minor'.format(int(t_minor*way)))
        names.append(name)
    return names

    

def fsl_imbalanced(g_args, models=[], strategies=[], seeds=[], train_tasks=[], test_tasks=[], var_update={}, save=True, 
                   expfolder='', pretrained_backbone=None, slow_learning=False, dataset = 'mini', backbone='Conv4',
                   template_prefix='{dataset}/'):
    
    n_way = 5
    
    if slow_learning:
        train_setup = {
            'num_epochs': [200],
             'model_args.lr'              : [0.001],
             'model_args.lr_decay'        : [1.0],
             'model_args.lr_decay_step'   : [200],
             'num_tasks_per_epoch'        : [2500],
            }
    else:
        train_setup = {
            'num_epochs': [200],
             'model_args.lr'              : [0.001], 
             'model_args.lr_decay'        : [0.1],
             'model_args.lr_decay_step'   : [100],
             'num_tasks_per_epoch'        : [500],
            }
        
    if pretrained_backbone is not None:
        train_setup = {
        'num_epochs': [50],
         'model_args.lr'              : [0.0001], 
         'model_args.lr_decay'        : [0.1],
         'model_args.lr_decay_step'   : [25],
        }
    
    is_baseline = lambda x: x in ['baseline', 'baselinepp', 'knn']
    
    experiement_files = []
    for seed in seeds:
        for model in models:
            for train_task in train_tasks:
                train_name = make_names([train_task], n_way)[0]
                
                for strategy in strategies:
                
                    variables = {
                        'results_folder'             : [os.path.abspath(g_args.results_folder)],
                        'seed'                       : [seed],
                        'backbone'                   : [backbone],
                        'num_tasks_per_validation'   : [200],
                        'num_tasks_per_testing'      : [600],
                        'strategy'                   : [strategy],
                        'model'                      : [model],
                        'task'                       : ['fsl_imbalanced'],
                        'task_args.batch_size'       : [1],
                        'task_args.num_classes'      : [n_way],
                        'dataset'                    : [dataset],
                        'dataset_args.data_path'     : [g_args.data_path],
                        'dataset_args.train.aug'     : [True],
                        'ptracker_args.test.metrics' : [['accuracy', 'loss', 'per_cls_stats']],
                        'tqdm'                       : [False]
                    }
                    variables.update(train_setup)
                    
                    variables[('task_args.min_num_supports', 
                               'task_args.max_num_supports',
                               'task_args.num_minority',
                               'task_args.imbalance_distribution', 
                               'task_args.min_num_targets', 
                               'task_args.max_num_targets',
                               'task_args.num_minority_targets',
                               'task_args.imbalance_distribution_targets')] = [train_task]

                    if is_baseline(model):
                        variables.update({
                            'no_val_loop'                       :[False],
                            'conventional_split'                :[True],
                            'conventional_split_from_train_only':[False],
                        })
                    
                    
                    if len(test_tasks) > 0:   # else if no test task is given, assume train task is the same as evaluation task 
                        variables[('task_args.eval.min_num_supports', 
                                   'task_args.eval.max_num_supports',
                                   'task_args.eval.num_minority',
                                   'task_args.eval.imbalance_distribution', 
                                   'task_args.eval.min_num_targets', 
                                   'task_args.eval.max_num_targets',
                                   'task_args.eval.num_minority_targets',
                                   'task_args.eval.imbalance_distribution_targets')] = test_tasks
                    
                    is_pretrained=''
                    if pretrained_backbone is not None:
                        variables['continue_from'] = [pretrained_backbone]
                        variables['load_backbone_only'] = [True]
                        is_pretrained = 'pretrained_'
                    
                    # experiment path
                    template = expfolder + template_prefix + is_pretrained + '{backbone}/train_on_' + train_name
                    template += '/{strategy}/' 
                    #           + '{num_epochs}epochs_{num_tasks_per_epoch}tasks/'

                    if slow_learning:
                        template += '{num_tasks_per_epoch}x{num_epochs}ep_{model_args.lr}lr_' +\
                                    '{model_args.lr_decay_step}step/'
                    
                    template += '{model}/'
                    
                    if model in ['protonet']:
                        variables[(
                            'task_args.train.num_classes',
                            'task_args.train.min_num_targets',
                            'task_args.train.max_num_targets')] = [
                                                                (20, 5, 5), 
                                                                (5, 15, 15)
                                                              ]
                        template += '{task_args.train.num_classes}trainway/'

                    elif model in ['baseline', 'baselinepp', 'knn']:
                        variables['task_args.trval.batch_size'] = [128]  # train and validation batch

                    elif model in ['maml', 'protomaml']:
                        variables['model_args.batch_size'] = [4] if model == 'maml' else [1]
                        variables['model_args.inner_loop_lr'] = [0.1] if model == 'maml' else [0.005]
                        variables['model_args.num_inner_loop_steps'] = [10] if model == 'maml' else [5]

                        #template += '{model_args.num_inner_loop_steps}innersteps_' +\
                        #          '{model_args.inner_loop_lr}innerlr/'

                    elif model in ['bmaml', 'bmaml_chaser']:
                        variables['model_args.batch_size'] = [1]
                        variables['model_args.inner_loop_lr'] = [0.1]
                        variables['model_args.num_inner_loop_steps'] = [1]
                        variables['model_args.num_draws'] = [20]
                        
                        if model == 'bmaml_chaser':
                            variables['model_args.leader_inner_loop_lr'] = [0.5]

                        
                    elif model == 'btaml':
                        variables['model_args.approx'] = [False]
                        variables['model_args.batch_size'] = [4]
                        variables['model_args.inner_loop_lr'] = [0.01]
                        variables['model_args.num_inner_loop_steps'] = [{"train":4, "val":10, "test":10}]
                        variables[('model_args.alpha_on', 
                                   'model_args.omega_on',
                                   'model_args.gamma_on',
                                   'model_args.z_on')] = [(True, True, True, True)]

                        #template += "{model_args.alpha_on}a_{model_args.omega_on}o_" + \
                        #            "{model_args.gamma_on}g_" + \
                        #            "{model_args.z_on}z_{model_args.approx_until}till/"
                        #template += '{model_args.batch_size}trainbatch_'+ \
                        #            '{model_args.inner_loop_lr}innerlr_' + \
                        #            '{model_args.num_inner_loop_steps}innersteps/'
                    
                    elif model == 'relationnet':
                        variables['model_args.loss_type'] = ['softmax']
                        template += '{model_args.loss_type}/'

                    elif model == 'simpleshot':
                        variables['task_args.train.batch_size'] = [128]
                        variables['model_args.feat_trans_name'] = ['CL2N']
                        variables['model_args.train_feat_trans'] = [False]
                        variables['model_args.approx_train_mean'] = [False]  # if true will speed up dataset mean calculation
                        
                    template += '{seed}/'
                    variables.update(var_update)
                    
                    expfiles = generate_experiments(template, variables, g_args, save=save)
                    experiement_files.extend(expfiles)
    return experiement_files


def imbalanced_task_test(g_args, expfiles):
    n_way=5

    test_settings = [
        # Test of 5 avr shot experiments
        (5, 5,  None, 'balanced', 15, 15, None, 'balanced'),
        (4, 6,  None, 'linear', 15, 15, None, 'balanced'),
        (1, 9,  None, 'random', 15, 15, None, 'balanced'),
        (1, 9,  0.2,  'step', 15, 15, None, 'balanced'),
        
        # (3, 7,  None, 'linear', 15, 15, None, 'balanced'),
        # (2, 8,  None, 'linear', 15, 15, None, 'balanced'),
        # (1, 9,  None, 'linear', 15, 15, None, 'balanced'),
        # (1, 21,  0.8, 'step', 15, 15, None, 'balanced'),
        # (1, 6,  0.2, 'step', 15, 15, None, 'balanced'),
        # (1, 9,  0.8, 'step', 15, 15, None, 'balanced'),
        # (3, 7,  None, 'random', 15, 15, None, 'balanced'),
        
        #### Test settings for 15 avr shot experiments 
        # (15, 15,  None, 'balanced', 15, 15, None, 'balanced'),
        # (10, 20,  None,   'linear', 15, 15, None, 'balanced'),
        # (13, 17,  None,   'linear', 15, 15, None, 'balanced'),
        # ( 5, 25,  None,   'linear', 15, 15, None, 'balanced'),
        # ( 3, 27,  None,   'linear', 15, 15, None, 'balanced'),
        # ( 1, 29,  None,   'linear', 15, 15, None, 'balanced'),
        
        #### Test settings for 25 avr shot experiments 
        # (25, 25,  None, 'balanced', 15, 15, None, 'balanced'),
        # (20, 30,  None,   'linear', 15, 15, None, 'balanced'),
        # (15, 35,  None,   'linear', 15, 15, None, 'balanced'),
        # (10, 40,  None,   'linear', 15, 15, None, 'balanced'),
        # ( 5, 45,  None,   'linear', 15, 15, None, 'balanced'),
        # ( 1, 49,  None,   'linear', 15, 15, None, 'balanced'),
    ]
    
    test_names = make_names(test_settings, n_way)
    
    for experiment in expfiles:
        
        script, script_path, config, config_path = experiment
        
        # expanded args also useful for backward compatibility. 
        config = substitute_hyperparams(config)
        
        assert config['task'] == 'fsl_imbalanced'
        
        for t, test_setting in enumerate(test_settings):
            test_name = test_names[t]
            
            variables = {
                'continue_from' :                        ['best'],
                'evaluate_on_test_set_only':             [True],
                'test_performance_tag':                  [test_name],
                'task_args.test.num_classes':            [n_way],
            }
            
            variables[(
                'task_args.test.min_num_supports',
                'task_args.test.max_num_supports',
                'task_args.test.num_minority',
                'task_args.test.imbalance_distribution',
                'task_args.test.min_num_targets', 
                'task_args.test.max_num_targets',
                'task_args.test.num_minority_targets',
                'task_args.test.imbalance_distribution_targets'
            )] = [test_setting]
            
            generate_experiments(
                config['experiment_name'], 
                variables,
                g_args,
                default_config = config,
                save=True,
                config_name='config_test_on_{}'.format(test_name),
                script_name='script_test_on_{}'.format(test_name),
                log_name='log_test_on_{}'.format(test_name)
            )

            
def strategy_inference(g_args, expfiles):
    n_way=5
    test_tasks = [  
        (5, 5,  None, 'balanced', 15, 15, None, 'balanced'),  # K_min, K_max, N_min, I-distribution 
        (4, 6,  None, 'linear', 15, 15, None, 'balanced'),
        (1, 9,  None, 'random', 15, 15, None, 'balanced'),
        (1, 9,  0.2,  'step', 15, 15, None, 'balanced')       # N_min expressed as a fraction of 'n_way'
        
        # Other, uncomment if appropiate
        # (5, 5,  None, 'balanced', 15, 15, None, 'balanced'),  # K_min, K_max, N_min, I-distribution 
        # (4, 6,  None, 'linear', 15, 15, None, 'balanced'),
        # (3, 7,  None, 'linear', 15, 15, None, 'balanced'),
        # (2, 8,  None, 'linear', 15, 15, None, 'balanced'),
        # (1, 9,  None, 'linear', 15, 15, None, 'balanced'),
        # (1, 21,  0.8, 'step', 15, 15, None, 'balanced'),
        # (1, 6,  0.2, 'step', 15, 15, None, 'balanced'),
        # (1, 9,  0.8, 'step', 15, 15, None, 'balanced'),
        # (3, 7,  None, 'random', 15, 15, None, 'balanced'),
    ]
    test_names = make_names(test_tasks, n_way, query_half=False)
    
    # Inference Strategies
    strategies = [
        'ros',
        # 'ros_aug',
        # 'focal_loss',
        # 'weighted_loss',
        # 'cb_loss',
    ]
    
    for experiment in expfiles:
        for strategy in strategies:
            script, script_path, base_config, config_path = experiment
            
            base_config['strategy'] = strategy
            base_config['strategy_args'] = {}
            
            # expanded args also useful for backward compatibility. 
            base_config = substitute_hyperparams(base_config)

            assert base_config['task'] == 'fsl_imbalanced'
            
            continue_from = os.path.join(g_args.results_folder, base_config['experiment_name'])
            template = os.path.join('inference/{strategy}/', base_config['experiment_name'])
            
            for t, setting in enumerate(test_tasks):
                config = copy.deepcopy(base_config)
                test_name = test_names[t]

                variables = {
                    'continue_from' :                        [continue_from],
                    'evaluate_on_test_set_only':             [True],
                    'test_performance_tag':                  [test_name],
                    'task_args.test.num_classes':            [n_way],
                }
                variables[(
                    'task_args.test.min_num_supports',
                    'task_args.test.max_num_supports',
                    'task_args.test.num_minority',
                    'task_args.test.imbalance_distribution',
                    'task_args.test.min_num_targets', 
                    'task_args.test.max_num_targets',
                    'task_args.test.num_minority_tagets',
                    'task_args.test.imbalance_distribution_targets'
                )] = [setting]
                
                template = 'inference/{strategy}/'

                if strategy == 'cb_loss':
                    variables['strategy_args.beta'] = [0.8]
#                     template += "{strategy_args.beta}beta/"
                
                template += config['experiment_name']
                
#                 print('# ', config['experiment_name'])
#                 print('# ', template)
#                 print('#', config['continue_from'])
                
                generate_experiments(
                    template, 
                    variables,
                    g_args,
                    default_config = config,
                    save=True,
                    config_name='config_test_on_{}'.format(test_name),
                    script_name='script_test_on_{}'.format(test_name),
                    log_name='log_test_on_{}'.format(test_name)
                )
            
                 
def imbalanced_dataset(g_args, models=[], seeds=[], save=True, backbone=None):
    # meta-training dataset imbalance settings
    imbalance_settings = [
        (300, 300, None, 'balanced'),
        (30 , 570, None, 'linear'),
        (30, 570, 0.5, 'step'),
        (25, 444, 0.34375,  'step'),
        (None, None, None, 'step-animal'),
        
        # Reduced Dataset
        # (150, 150, None, 'balanced'),
        # (30, 190, 0.25, 'step'),
        # (30, 270,  0.5, 'step'),
        # (30, 510, 0.75, 'step'),
    ]
    
    strategies=[None]
    train_tasks=[
        (5, 5, None, 'balanced', 15, 15, None, 'balanced'),
        #(1, 9, None, 'linear', 15, 15, None, 'balanced')  # uncomment for combined imbalance results
    ]
    var_update = {
        'num_epochs': [200], 
        'num_tasks_per_epoch': [500], 
        # 'dataset_args.train.use_classes_frac': [0.5]
    }
    
    is_baseline = lambda x: x in ['baseline', 'baselinepp', 'knn']
    
    experiement_files = []

    for experiment in fsl_imbalanced(g_args, models=models, strategies=strategies, 
                                     seeds=seeds, var_update=var_update,
                                     train_tasks=train_tasks, save=False, 
                                     backbone=backbone):

        for setting in imbalance_settings:
            script, script_path, config, config_path = experiment

            # expanded args also useful for backward compatibility. 
            config = substitute_hyperparams(config)
            model = config['model']
            min_s, max_s, minor, dist  = setting
            
            variables = {}
            
            if dist == 'step-animal':
                variables.update({'dataset_args.train.dataset_version' : ['step-animal']})
                dist = None
                
            variables.update({
                'dataset_args.train.min_num_samples'       :[min_s],
                'dataset_args.train.max_num_samples'       :[max_s],
                'dataset_args.train.num_minority'          :[minor],
                'dataset_args.train.imbalance_distribution':[dist],
            })
            
            if is_baseline(model):
                variables.update({
                    'no_val_loop'                       :[True],  # no validation loop for baselines
                    'conventional_split'                :[False],
                    'conventional_split_from_train_only':[False],
                })
            
            template = os.path.join('imb_mini','{dataset_args.train.min_num_samples}_'+\
                                    '{dataset_args.train.max_num_samples}_'+\
                                    '{dataset_args.train.num_minority}_'+\
                                    '{dataset_args.train.imbalance_distribution}',
                                    config['experiment_name'])
            
            experiement_files.extend(generate_experiments(
               template, 
               variables, 
               g_args,
               default_config=config,
               save=save
            ))
    return experiement_files


def tailed_dataset(g_args, models=[], seeds=[], save=True, backbone='Conv4'):
    
    # meta-training dataset imbalance settings
    datasets = [
        "imgnt"
    ]
    dataset_versions = [
        "longtail",
        "balanced",
    ]
    
    strategies=[None]
    train_tasks=[(5, 5, None, 'balanced', 15, 15, None, 'balanced')]
    var_update = {'num_epochs': [200], 'num_tasks_per_epoch': [1000]}
    
    is_baseline = lambda x: x in ['baseline', 'baselinepp', 'knn']
    
    experiement_files = []
    
    for dataset in datasets:
        for version in dataset_versions:
            for experiment in fsl_imbalanced(g_args, models=models, strategies=strategies, seeds=seeds, var_update=var_update,
                                             train_tasks=train_tasks, save=False, dataset=dataset, backbone='ResNet10', 
                                             template_prefix=''):
                script, script_path, config, config_path = experiment
                # expanded args also useful for backward compatibility.
                config = substitute_hyperparams(config)
                
                template = os.path.join('imb_{dataset}/{dataset_args.dataset_version}/', config['experiment_name'])
                
                experiement_files.extend(generate_experiments(
                    template,
                    {
                        "dataset"                            : [dataset],
                        "dataset_args.dataset_version"       : [version],
                        'dataset_args.imbalance_distribution': [None],
                        "dataset_args.seed"                  : [config['seed']],
                        'no_val_loop'                        : [is_baseline(config['model'])],
                        'conventional_split'                 : [False],
                        'conventional_split_from_train_only' : [False]
                    }, 
                    g_args,
                    config,
                    save=save
                ))
            
    return experiement_files

            
def cub_inference(g_args, expfiles, save=True):
    
    datasets = [
        ('cub_inf', 'mini_to_cub')
    ]
    
    for prefix, dateset in datasets:
        
        for experiment in expfiles:
            script, script_path, config, config_path = experiment
            config = copy.deepcopy(base_config)
            
            # expanded args also useful for backward compatibility. 
            config = substitute_hyperparams(config)
            continue_from = os.path.join(g_args.results_folder, config['experiment_name'])
            template = os.path.join(prefix,config['experiment_name'])
            
            variables = {
                'continue_from' : [continue_from],
                'dataset': [dateset],
                'evaluate_on_test_set_only': [True],
                'dataset_args.test.imbalance_distribution': [None],
            }
            
            generate_experiments(
                template,
                variables, 
                g_args,
                config,
                save=True
            )
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=['{gpu}'], type=str, nargs="+", help='GPU ID')
    parser.add_argument('--dummy_run', type=str2bool, nargs='?', const=True, default=False,
                        help='Produces scripts as a "dry run" with a reduced number of tasks, '
                             'and no mobel saving (useful for debugging)')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='Folder with data')
    parser.add_argument('--models', '--model', type=str, nargs="*", default=[],
                        help='Run selected models')
    parser.add_argument('--strategies', '--strategy', type=str, nargs="*", default=[],
                        help='Run selected strategies')
    parser.add_argument('--backbone', type=str, default='Conv4',
                        help='See ./src/utils/utils.py file for a list of valid backbones.')
    parser.add_argument('--seeds', '--seed', type=int, nargs="*", default=[],
                        help='Generate experiments using selected seed numbers')
    parser.add_argument('--results_folder', type=str, default='./experiments/',
                        help='Folder for saving the experiment config/scripts/logs into')
    parser.add_argument('--imbalanced_supports', type=str2bool, nargs='?', const=True, default=False,
                        help='Generate imbalanced support set experiments')
    parser.add_argument('--imbalanced_targets', type=str2bool, nargs='?', const=True, default=False,
                        help='Generate imbalanced target set experiments')
    parser.add_argument('--imbalanced_dataset', type=str2bool, nargs='?', const=True, default=False,
                        help='Generate imbalanced dataset experiments')
    parser.add_argument('--tailed_dataset', type=str2bool, nargs='?', const=True, default=False,
                        help='Generate imbalanced dataset experiments with a long-tail distribution')
    parser.add_argument('--dataset', type=str, default='mini')
    parser.add_argument('--slow_learning', type=str, nargs='?', const=True, default=False,
                        help='If true, runs slower learning rate.')
    parser.add_argument('--load_backbone', type=str, nargs='?', const=True, default=False,
                        help='If true, loads backbone of a Baseline++ model.')
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
    g_args = parser.parse_args()
    
    g_args.results_folder = os.path.abspath(g_args.results_folder)
    
    GPU_COUNTER = 0    

    if g_args.models is None or len(g_args.models) == 0:
        models = [
            'baseline',
            'baselinepp',
            'matchingnet',
            'relationnet',
            'protonet',
            'dkt',
            'simpleshot',
            'maml',
            'protomaml',
            'btaml',
            # 'bmaml',
            # 'bmaml_chaser',
            # 'knn',
        ]
    else:
        models = g_args.models
        
    if g_args.strategies is None or len(g_args.strategies) == 0:
        strategies = [
            None,
            'ros',
            'ros_aug',
            'focal_loss',
            'weighted_loss',
            'cb_loss'
        ]
    else:
        strategies = [ None if s in ['None','none'] else s for s in g_args.strategies]
    
    if g_args.seeds is None or len(g_args.seeds) == 0:
        seeds = [
            0,
            1, 
            2
        ]
    else:
        seeds = g_args.seeds
        
    balanced_tasks = [
        (5, 5, None, 'balanced', 15, 15, None, 'balanced'),  # Standard Meta-Training
        # (15, 15, None, 'balanced', 15, 15, None, 'balanced'),  # -- uncomment if appropiate
        # (25, 25, None, 'balanced', 15, 15, None, 'balanced'),  # -- uncomment if appropiate
    ]
    
    imbalanced_tasks = [
        (1, 9, None, 'random', 15, 15, None, 'balanced'),   # Random-Shot Meta-Training
        # (1, 29, None, 'random', 15, 15, None, 'balanced'),  # -- uncomment if appropiate
        # (1, 49, None, 'random', 15, 15, None, 'balanced'),  # -- uncomment if appropiate
    ]
    
    if g_args.minimal:
        models = models[:2]
        strategies = strategies[:2]
        seeds = seeds[:1]
    
    
    backbone = None
    if g_args.load_backbone:
        backbone_files = fsl_imbalanced(g_args, models=['baselinepp'], strategies=[None], seeds=seeds, train_tasks=[
           (5, 5, None, 'balanced', 15, 15, None, 'balanced')], save=False, expfolder='imbalanced_supports/', 
                                        backbone=g_args.backbone, dataset=g_args.dataset)
        _, _, config, _ = backbone_files[0]
        backbone = config['experiment_name']
        backbone = "/media/disk2/mateusz/repositories/imbalanced_fsl_dev/experiments/imbalanced_task/mini/Conv4/"+\
                    "train_on_5-5shot_balanced_None/baseline/0/checkpoint/epoch-199"
    
    
    if g_args.imbalanced_supports:
        # Standard meta-training
        standard_expfiles = fsl_imbalanced(g_args, models=models, strategies=[None], seeds=seeds, train_tasks=balanced_tasks,
                                save=not (g_args.test or g_args.inference), expfolder='imbalanced_supports/', 
                                           pretrained_backbone=backbone, slow_learning=g_args.slow_learning, 
                                        backbone=g_args.backbone, dataset=g_args.dataset)
        # Random Shot meta-training
        randomshot_expfiles = fsl_imbalanced(g_args, models=models, strategies=strategies, seeds=seeds,
                                             train_tasks=imbalanced_tasks, save=not (g_args.test or g_args.inference), 
                                             expfolder='imbalanced_supports/', pretrained_backbone=backbone, 
                                             slow_learning=g_args.slow_learning, backbone=g_args.backbone, 
                                             dataset=g_args.dataset)
        
        if g_args.test: 
            imbalanced_task_test(g_args, standard_expfiles)
            imbalanced_task_test(g_args, randomshot_expfiles)
            
        if g_args.inference:
            strategy_inference(g_args, standard_expfiles)
            strategy_inference(g_args, randomshot_expfiles)
    
    
    if g_args.imbalanced_dataset:
        expfiles = imbalanced_dataset(g_args, models=models, seeds=seeds, save=not (g_args.test or g_args.inference), 
                                        backbone=g_args.backbone)
        
        if g_args.test:
            print('Balanced task testing is performed automatically after training. Use --inference to evaluate on CUB.')
        
        if g_args.inference:
            cub_inference(g_args,expfiles)
            
    
    if g_args.imbalanced_targets:
        train_tasks = [
            (5, 5, None, 'balanced', 5, 5, None, 'balanced'),
            (5, 5, None, 'balanced', 1, 9, None, 'linear'),
            (1, 9, None, 'linear', 5, 5, None, 'balanced'),
            (1, 9, None, 'linear', 1, 9, None, 'linear'),
        ]
        
        test_tasks = [
            (1, 9, None, 'linear', 5, 5, None, 'balanced')
        ]
        
        expfiles = fsl_imbalanced(g_args, models=models, strategies=[None], seeds=seeds, train_tasks=train_tasks, 
                                  test_tasks=test_tasks, save=not(g_args.test or g_args.inference), 
                                  expfolder='imbalanced_targets/', pretrained_backbone=backbone, 
                                  slow_learning=g_args.slow_learning, backbone=g_args.backbone, dataset=g_args.dataset)
        
        if g_args.test:
            imbalanced_target_test(g_args, expfiles)
            
        if g_args.inference:
            print('Strategy inference for imbalnaced target set not yet implemented.')
    
    
    if g_args.tailed_dataset:
        expfiles = tailed_dataset(g_args, models=models, seeds=seeds, save=not (g_args.test or g_args.inference), 
                                        backbone=g_args.backbone)
        
        if g_args.test:
            print('Testing performed automatically after training')
        
        if g_args.inference:
            print('No inference experiments')
            
    
    if not (g_args.imbalanced_dataset or g_args.imbalanced_supports or g_args.imbalanced_targets or g_args.tailed_dataset):
        print('Please specify --imbalanced_dataset or --imbalanced_supports')

