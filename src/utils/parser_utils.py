import json
import argparse
import pprint
import copy
import collections
from collections import abc, defaultdict
import os, sys
import re
import pytest


def update_dict(base, to_update):
    """
    Updates a nested dict
    """
    if base is None:
        return to_update
    
    for k, v in to_update.items():
        if isinstance(v, abc.Mapping) and k in base:
            base[k] = update_dict(base[k], v)
        else:
            base[k] = v
    return base


def update_dict_exclusive(base, to_update):
    """
    Updates a nested dict, excluding any parameters missing from base
    """
    if base is None:
        return to_update
    
    excluded = {}
    for k, v in to_update.items():
        if isinstance(v, abc.Mapping) and k in base:
            base[k], excluded[k] = update_dict_exclusive(base[k], v)
        elif k in base:
            base[k] = v
        else:
            excluded[k] = v
    
    # filter empty
    excluded = {k:v for k,v in excluded.items() if v != {}}
    return base, excluded


def from_syntactic_sugar(params):
    """
    Splits keys containing '.', and converts into a nested dict
    """
    combined = dict()
    for key, value in list(params.items()):
        dict_item = expand_key(key.split("."), value)
        combined = update_dict(combined, dict_item)
    return combined


def expand_key(keylist, value):
    """
    Recursive method for converting into a nested dict
    Splits keys containing '.', and converts into a nested dict
    """
    
    if len(keylist) == 0:
        return expand_value(value)
    
    elif len(keylist) == 1:
        key = '.'.join(keylist)
        base = dict()
        base[key] = expand_value(value)
        return base
    
    else:
        key = keylist[0]
        value = expand_key(keylist[1:], value)
        base = dict()
        base[key] = expand_value(value)
        return base

    
def expand_value(v):
    if isinstance(v, abc.Mapping):  # if dict
        return from_syntactic_sugar(v)
    else:
        return v
    
    
def to_syntactic_sugar(args, fromlvl=None):
    """
    """
    fromlvl = 0 if fromlvl is None else fromlvl
    
    if not isinstance(args, abc.Mapping): # if not dict
        return args
    
    sugar_dict = dict()
    for key, value in list(args.items()):
        if isinstance(value, abc.Mapping): # if not dict
            value = to_syntactic_sugar(value, fromlvl-1)
            for subkey, subvalue in list(value.items()):
                if fromlvl <= 0:
                    new_key = '.'.join([str(key),str(subkey)])
                    sugar_dict[new_key] = to_syntactic_sugar(subvalue, fromlvl-2)
                else:
                    if key not in sugar_dict:
                        sugar_dict[key] = dict()
                    sugar_dict[key][subkey] = to_syntactic_sugar(subvalue, fromlvl-2)
        else:
            sugar_dict[key] = value
        
    return sugar_dict


def expand_three_phase_and_update(default_args, args, suppress_warning=True):
    """
    Updated default_args with hyperparameters in args
    param default_args: should a dict(test=dict(), train=dict(), val=dict()) to update, containing full default arguments
    """
    
#     print("default_args",default_args)
#     print("args",args)
#     import pdb; pdb.set_trace()
    
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
            if len(missing_from)!=0 and not suppress_warning:
                print("#\t Hyperparameter args.{}.{} not found in default_args.{}".format(
                    bucket, hyperparam, '|'.join(missing_from)))
            
            # Detect if the hyperparam is defined in another bucket of a lower order with overlapping setnames
            bucket_clushes = []
            for other_bucket in bucket_order:
                if not suppress_warning and hyperparam in args_in_buckets[other_bucket] and \
                   bucket_order.index(other_bucket) < bucket_order.index(bucket) and \
                   not set(buckets_to_setnames[bucket]).isdisjoint(set(buckets_to_setnames[other_bucket])):
                    bucket_clushes.append(other_bucket)
            
            if len(bucket_clushes) > 0:
                other_bucket = bucket_clushes[-1]  # get highest order bucket
                if not suppress_warning:
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
    
#     print("default_args", default_args)
#     import pdb; pdb.set_trace()
    
    return default_args

    

def expand_three_phase(args, suppress_warning=True):
    """
    Updated default_args with hyperparameters in args
    param default_args: should a dict(test=dict(), train=dict(), val=dict()) to update, containing full default arguments
    """
    
    # Place args in buckets such that they can be sorted out properly according to bucket rank
    args_in_buckets = dict(test=dict(), train=dict(), val=dict(), eval=dict(), trval=dict(),_other=dict())
    args_in_buckets = update_dict(args_in_buckets, args)
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
    
    new_args = {}
    
    # Goes through each bucket in turn, adding to new_args from args_in_buckets, while taking order into account
    for bucket in bucket_order:
        setnames = buckets_to_setnames[bucket]
        
        # For each hyperparam in a bucket, sort out any clashes based on bucket order
        for hyperparam in args_in_buckets[bucket]:
            
            # Detect if the hyperparam is defined in another bucket of a lower order with overlapping setnames
            bucket_clushes = []
            for other_bucket in bucket_order:
                if not suppress_warning and hyperparam in args_in_buckets[other_bucket] and \
                   bucket_order.index(other_bucket) < bucket_order.index(bucket) and \
                   not set(buckets_to_setnames[bucket]).isdisjoint(set(buckets_to_setnames[other_bucket])):
                    bucket_clushes.append(other_bucket)
            
            if len(bucket_clushes) > 0:
                other_bucket = bucket_clushes[-1]  # get highest order bucket
                if not suppress_warning:
                    print("#\t Overwriting args{}.{} = {} \t with args{}.{} = {} ".format(
                       '' if other_bucket == '_other' else ".{}".format(other_bucket), 
                        hyperparam, args_in_buckets[other_bucket][hyperparam], 
                        '' if bucket == '_other' else ".{}".format(bucket),
                        hyperparam, args_in_buckets[bucket][hyperparam]))
                
            # update parameter
            for s in setnames:
                if s not in new_args:
                    new_args[s] = {}
                new_args[s][hyperparam] = args_in_buckets[bucket][hyperparam]
    
    return new_args


def compress_three_phase(args_dict):
    """
    Compresses an args_dict into a more compact form of dictionary
    The args_dict should contain 'train', 'test', 'val' keys
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


def load_json(filename):
    with open(filename) as f:
        json_args = json.load(fp=f)
    return json_args

            
class OnePhaseDict(argparse.Action):
    
    @staticmethod
    def TYPE(default=None):
        return dict(type=json.loads, 
                default=default, #vars(model_parser.parse_args([])),
                action=OnePhaseDict)
    
    def __init__(self, option_strings, subparser, default=None, *args, **kwargs):
        self._subparser=subparser
        if default is None: default=vars(subparser.parse_args([]))
        super(OnePhaseDict, self).__init__(option_strings=option_strings, default=default, *args, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        defaults = from_syntactic_sugar(vars(namespace)[self.dest])
        values = from_syntactic_sugar(values)
        new_values, excluded = update_dict_exclusive(defaults,values)
        new_values['__excluded'] = excluded
        setattr(namespace, self.dest, new_values)

        
class ThreePhaseDict(argparse.Action):
    
    @staticmethod
    def TYPE(default=None):
        return dict(type=json.loads, 
                default=default,
                action=ThreePhaseDict)
    
    def __init__(self, option_strings, subparser, default=None, *args, **kwargs):
        if not isinstance(subparser, abc.Mapping):
            subparser = {'train':copy.deepcopy(subparser),
                         'val':copy.deepcopy(subparser),
                         'test':copy.deepcopy(subparser)}
        self._subparser=subparser
        if default is None: default= {_phase:vars(_parser.parse_args([])) for _phase, _parser in subparser.items()}
        super(ThreePhaseDict, self).__init__(option_strings=option_strings, default=default, *args, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        defaults = expand_three_phase(from_syntactic_sugar(vars(namespace)[self.dest]))
        values = expand_three_phase(from_syntactic_sugar(values), self._subparser)
        new_values, excluded = update_dict_exclusive(defaults, values)
        new_values['__excluded'] = excluded
        setattr(namespace, self.dest, new_values)
        

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
            print("MODEL_ARGS for '{}' (1-phase):".format(model))
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
            print("STRATEGY_ARGS for '{}' (1-phase)".format(strategy))
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

        
def my_test_cases():
    return [
    (
        """--args_file test.json --args2 {"test.version":1.7} """.split(),
        {'args1': {'lr3': 0.3},
            'args2': {'train': {'version': 2.3},
            'eval': {'version': 3.4}},
            'args4': {'model':'abc'}
        },{
            'args1': {'lr1': {'test': 0.1, 'val': 0.1}, 'lr2': 0.2},
            'args2': {
                'test': {'version': 1.7},
                'train': {'version': 2.3},
                'val': {'version': 3.4}},
#             'args4': {'model':'abc'},
            'model': 'model',
            'data': 'data',
            'args_file': 'test.json',
            'args5': 'tochange'
        },
        {'args1': {'lr3': 0.3}, 'args4': {'model': 'abc'}}
    ),
    (
        """""".split(),
        {'args1': {'lr3': 0.3},
            'args2': {
                'train': {'version': 2.3},
                'eval': {'version': 3.4}},
            'args4': {'model':'abc'}
        },{
            'args1': {'lr1': {'test': 0.1, 'val': 0.1}, 'lr2': 0.2},
            'args2': {
                'test': {'version': None},
                'train': {'version': None},
                'val': {'version': None}},
            'args_file': None,
            'model': 'model',
            'data': 'data',
            'args5': 'tochange'
        },
        {}
    ),
    (
        """--args_file test.json""".split(),
        {'args1': {'lr2':0.3, 'lr3': 0.5},
            'args2': {
                'train': {'version': 2.3},
                'eval': {'version': 3.4}},
            'args4': {'model':'abc'}
        },{
            'args1': {'lr1': {'test': 0.1, 'val': 0.1}, 'lr2': 0.3},
            'args2': {
                'test': {'version': 3.4},
                'train': {'version': 2.3},
                'val': {'version': 3.4}},
            'args_file': 'test.json',
#             'args4': {'model':'abc'},
            'model': 'model',
            'data': 'data',
            'args5': 'tochange'
        },
        {'args1': {'lr3': 0.5}, 'args4': {'model': 'abc'}}
    ),
    (
        """--args_file test.json""".split(),
        {'args1': {'lr2':0.3, 'lr3': 0.66},
            'args2': {
                'eval': {'version': 3.4}},
            'args4': {'model':'abc'}
        },{
            'args1': {'lr1': {'test': 0.1, 'val': 0.1}, 'lr2': 0.3},
            'args2': {
                'test': {'version': 3.4},
                'train': {'version': None},
                'val': {'version': 3.4}},
            'args_file': 'test.json',
#             'args4': {'model':'abc'},
            'model': 'model',
            'data': 'data',
            'args5': 'tochange'
        },
        {'args1': {'lr3': 0.66}, 'args4': {'model': 'abc'}}
    ),
    (
        """--args_file test.json --args5 changed""".split(),
        {'args1': {'lr2':0.3, 'lr3': 0.66},
            'args2': {
                'eval': {'version': 3.4}},
            'args4': {'model':'abc'}
        },{
            'args1': {'lr1': {'test': 0.1, 'val': 0.1}, 'lr2': 0.3},
            'args2': {
                'test': {'version': 3.4},
                'train': {'version': None},
                'val': {'version': 3.4}},
            'args_file': 'test.json',
#             'args4': {'model':'abc'},
            'model': 'model',
            'data': 'data',
            'args5': 'changed'
        },
        {'args1': {'lr3': 0.66}, 'args4': {'model': 'abc'}}
    ),
    (
        """--args_file test.json  --args2 {"toexclude":0.123} """.split(),
        {'args1': {'lr2':0.3, 'lr3': 0.66},
            'args2': {
                'eval': {'version': 3.4}},
            'args4': {'model':'abc'},
            'args5': 'changed'
        },{
            'args1': {'lr1': {'test': 0.1, 'val': 0.1}, 'lr2': 0.3},
            'args2': {
                'test': {'version': 3.4},
                'train': {'version': None},
                'val': {'version': 3.4}},
            'args_file': 'test.json',
#             'args4': {'model':'abc'},
            'model': 'model',
            'data': 'data',
            'args5': 'changed'
        },{
            'args1': {'lr3': 0.66}, 
            'args2': {
                 'test': {'toexclude': 0.123}, 
                 'train': {'toexclude': 0.123}, 
                 'val': {'toexclude': 0.123}},
            'args4': {'model': 'abc'}
        }
    )
    ]
    
@pytest.mark.parametrize("sysargv, json_args, target_args, target_excluded",my_test_cases())
def test_parser(sysargv, json_args, target_args, target_excluded):
    
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--args_file', type=str, default=None)
#     base_parser.add_argument('--args_file', type=str, default=None, 
#                              action=ArgsLoad, args_for_test_only=json_args)
    
    base_parser = argparse.ArgumentParser(description="Base Parser", add_help=False, parents=[config_parser])
    base_parser.add_argument('--model', type=str, default='model')
    base_parser.add_argument('--data', type=str, default='data')
    base_parser.add_argument('--args5', type=str, default='tochange')
    
    # Step 1. Get config file
    config_args, remaining_argv = config_parser.parse_known_args(sysargv)
    config_args = vars(config_args)
    if config_args['args_file'] not in [None,'None','none','']:
        #json_args = load_json(base_args['args_file'])
        json_args = from_syntactic_sugar(json_args)
        json_args['args_file'] = config_args['args_file']
    else:
        json_args = {}
    
    # Step 2. Update base args defaults using json
    default_args = vars(base_parser.parse_args([]))
    default_args, excluded_args = update_dict_exclusive(default_args, json_args)
    base_parser.set_defaults(**default_args)
    
    # Step 3. Update base args using command line args
    base_args, remaining_argv = base_parser.parse_known_args(remaining_argv)
    base_args = vars(base_args)
    
    # Step 4. Initilize nested parsers
    model_parser = argparse.ArgumentParser(description="Model args")
    data_parser = argparse.ArgumentParser(description="Data args")
    if base_args['model'] == 'model':
        model_parser.add_argument('--lr1', type=json.loads, default={'test':0.1,'val':0.1})
        model_parser.add_argument('--lr2', type=int, default=0.2)
    if base_args['data'] == 'data':
        data_parser.add_argument('--version', type=str, default=None)
    
    nested_parser = argparse.ArgumentParser(description="Nested Parser", parents=[base_parser], add_help=False)
    nested_parser.add_argument('--args1', **OnePhaseDict.TYPE(), subparser=model_parser)
    nested_parser.add_argument('--args2', **ThreePhaseDict.TYPE(), subparser=data_parser)
    
    # Step 5. Translate and expand nested args in excluded args
    is_three_phase = {a.dest:type(a) is ThreePhaseDict for a in nested_parser._actions}
    for k in excluded_args.keys():
        if k in is_three_phase and is_three_phase[k]:
            excluded_args[k] = expand_three_phase(excluded_args[k])
    
    # Step 6. Updated nested args defaults using base_args
    default_args = vars(nested_parser.parse_args([]))
    default_args, excluded = update_dict_exclusive(default_args, base_args)
    assert excluded == {}
    default_args, excluded_args = update_dict_exclusive(default_args, excluded_args)
    nested_parser.set_defaults(**default_args)
    
    # Step 7. Update nested args using command line args
    nested_args, remaining_argv = nested_parser.parse_known_args(remaining_argv)
    nested_args = vars(nested_args)
    
    # Step 8. Delete excluded args left over by nestedparsers
    for k in list(nested_args.keys()):
        if isinstance(nested_args[k], abc.Mapping) and '__excluded' in nested_args[k]:
            if k not in excluded_args:
                excluded_args[k] = {}
            excluded_args[k] = update_dict(excluded_args[k], nested_args[k]['__excluded'])
            if excluded_args[k] == {}: del excluded_args[k]
            del nested_args[k]['__excluded']
    
    assert nested_args == target_args, 'Assert fail \n*******\n{}\n != \n{}\n*******'.format(nested_args, target_args)
    assert excluded_args == target_excluded, 'Assert fail \n*******\n{}\n != \n{}\n*******'.format(excluded_args, 
                                                                                                   target_excluded)
    return nested_args, excluded_args


if __name__ == '__main__':
    args, excluded = test_parser(*my_test_cases()[-1])
    print('args')
    pprint.pprint(args)
    print()
    print('excluded')
    pprint.pprint(excluded)
    print()
