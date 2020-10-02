import numpy as np
import os
import json
import tqdm
import time
from collections import defaultdict
import torch
import shutil
import datetime

from utils.ptracker import PerformanceTracker
from utils.dataloader import DataLoader
from utils.utils import find, toBunch
from tasks.task_generator import TaskGenerator


class ExperimentBuilder():
    
    def __init__(self, model, tasks, datasets, device, args):
        """
        Builds a single experiment based on the configuration parameters.
        """
        print('Setting up Experiment Builder')
        self.model = model
        self.tasks = tasks
        self.datasets = datasets
        self.device = device
        self.args = args
        self.task_args = args.task_args
        
        self.experiment_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        self.create_experiment_folder()
        
        self.state = State(args.seed)
        self.ptracker = PerformanceTracker(
            folder=self.performance_folder,
            args=self.args.ptracker_args
        )
        self.val_or_test = args.val_or_test
        self.template='epoch-{:03d}'
    
    
    def create_experiment_folder(self):
        """
        Creates the experiment folder where checkpoint, configs, and performance is saved
        """
        if self.args.experiment_folder is None:
            self.args.experiment_folder = os.path.join(os.path.abspath(self.args.results_folder), self.args.experiment_name)
            
        self.experiment_folder    = self.args.experiment_folder
        self.checkpoint_folder    = os.path.join(self.experiment_folder, 'checkpoint')
        self.log_folder           = os.path.join(self.experiment_folder, 'logs')
        self.performance_folder   = os.path.join(self.experiment_folder, 'performance')
        self.visualisation_folder = os.path.join(self.experiment_folder, 'visualisation')
        
        if self.args.dummy_run:
            print('NOT Creating: ', self.experiment_folder)
            return
        
        print('Experiment folder: ', self.experiment_folder)
        if self.args.continue_from in [None, 'None', 'from_scratch'] and self.args.clean_folder:
            print('CLEARING FOLDER')
            shutil.rmtree(self.experiment_folder, ignore_errors=True)
        
        os.makedirs(self.experiment_folder, exist_ok=True)
        os.makedirs(self.checkpoint_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)
        os.makedirs(self.performance_folder, exist_ok=True)
        os.makedirs(self.visualisation_folder, exist_ok=True)
        
        config_path =  os.path.join(self.experiment_folder, 'config_{}.json'.format(self.experiment_timestamp))
        
        # Save args into a file
        with open(config_path, 'w') as f:
            json.dump(json.loads(str(self.args)), f, indent=2, sort_keys=True)
    
    
    def load_from_checkpoint(self, checkpoint_name_or_path, load_model_only=False):
        """
        Loads the model and state of the experiment from a checkpoint file
        """
        # Find checkpoint
        if os.path.isfile(checkpoint_name_or_path):
            filepath = checkpoint_name_or_path
        else:
            filepath = os.path.join(self.checkpoint_folder, checkpoint_name_or_path)
            if not os.path.isfile(filepath):
                raise Exception('Invalid checkpoint name or path: {}'.format(checkpoint_name_or_path))
        
        # Device mapping
        if torch.cuda.is_available():
            map_location= lambda storage, location: storage.cuda(self.device)
        else:
            map_location='cpu'
        
        print('Loading from checkpoint', filepath)
        checkpoint = torch.load(filepath, map_location=map_location)
        state, model, performance_log = checkpoint['state'], checkpoint['model'], checkpoint['performance_log']
        
        if load_model_only:
            self.model.load_state_dict(model)
            return 
        
        self.state.from_dict(state)
        self.model.load_state_dict(model)
        self.ptracker.load_from_logfile(performance_log)
    
    
    def save_checkpoint(self, checkpoint_name=None):
        """
        Saves the model and state of the experiment in a checkpoint file
        """
        if self.args.dummy_run:
            print('dummy run, no saving')
            return (self.state.epoch + 1 ) >= self.args.num_epochs
        
        if checkpoint_name is None: 
            checkpoint_name = self.template.format(self.state.epoch)
        
        checkpointpath = os.path.join(self.checkpoint_folder, checkpoint_name)
        performance_logfile = os.path.join(self.performance_folder, checkpoint_name + '.json')
        checkpoint_path_to_best = os.path.join(self.checkpoint_folder, 'path_to_best')
        ptracker_path_to_best = os.path.join(self.performance_folder, 'path_to_best')
        
        # Updates best val model if the validation score beats the previous best
        is_val_best_updated = self.ptracker.update_best(checkpointpath, self.state.epoch)['val']
        
        # Update path_to_best if best updated or no previous path to best
        if is_val_best_updated or not os.path.isfile(checkpoint_path_to_best):
            with open(checkpoint_path_to_best,'w') as f:
                f.write(checkpointpath)
            with open(ptracker_path_to_best,'w') as f:
                f.write(performance_logfile)
        
        # Make checkpoint
        checkpoint={}
        checkpoint['state'] = self.state.to_dict()
        checkpoint['model'] = self.model.state_dict()
        checkpoint['performance_log'] = performance_logfile
        
        # Save checkpoint
        print('saving to', checkpointpath, '\t\t')
        torch.save(checkpoint, checkpointpath)
        self.ptracker.save_logfile(performance_logfile, ['train', 'val'])
        
        # Delete checkpoints due to heavy storage
        if self.args.model in ['matchingnet','btaml'] or 'ResNet' in self.args.backbone or self.args.storage_friendly:
            in_allowed_epochs = lambda x: x % 40 == 19  # allowed every 40th epoch, e.i. at 19th, 59th, 99th etc..
            
            current_epoch = self.state.epoch
            previous_epoch = self.state.epoch -1
            current_best_epoch = self.ptracker.current_best['val']['epoch']
            previous_best_epoch = self.ptracker.previous_bests['val'][-1]['epoch']
            
            # remove previous epoch checkpoint (unless current best or in allowed epochs)
            if previous_epoch >= 0 and \
               not in_allowed_epochs(previous_epoch) and \
               previous_epoch != current_best_epoch:
                
                path = os.path.join(self.checkpoint_folder, self.template.format(previous_epoch))
                print('removing', path, '\t\t')
                os.remove(path)
                
            # remove previous best epoch checkpoint if best has been updated (unless is in allowed epochs)
            if is_val_best_updated and \
               previous_best_epoch >= 0 and \
               not in_allowed_epochs(previous_best_epoch) and \
               previous_best_epoch != current_best_epoch and \
               previous_best_epoch != previous_epoch:
                
                path = os.path.join(self.checkpoint_folder, self.template.format(previous_best_epoch))
                print('removing', path, '\t\t')
                os.remove(path)
                
        # Stop criterion 
        if self.args.num_epochs is None:
            current_best_epoch = self.ptracker.current_best['val']['epoch']
            return (self.state.epoch-current_best_epoch) > 30
        
        return (self.state.epoch + 1 ) >= self.args.num_epochs
        
        
    def load_pretrained(self):
        """
        Loads model from self.args.continue_from
        Return value indicates whether to continue from next epoch
        """
        print('Continuing from', self.args.continue_from)
        
        if self.args.continue_from in [None, 'None', 'from_scratch'] or self.args.dummy_run:
            return False  
        
        if self.args.continue_from == 'latest':
            checkpoint_names = find('epoch*', self.checkpoint_folder)
            checkpoint_names = sorted(checkpoint_names)
            self.args.continue_from = checkpoint_names[-1]
            print('LATEST', self.args.continue_from)
            self.load_from_checkpoint(self.args.continue_from)
        
        elif self.args.continue_from == 'best':
            with open(os.path.join(self.checkpoint_folder,'path_to_best'),'r') as f:
                self.args.continue_from = f.read()
            print('BEST', self.args.continue_from)
            self.load_from_checkpoint(self.args.continue_from)
            
        elif self.args.continue_from.isdigit():
            checkpoint_name = 'epoch-{:03d}'.format(int(self.args.continue_from))
            self.args.continue_from = os.path.join(self.checkpoint_folder, checkpoint_name)
            print('EPOCH', self.args.continue_from)
            self.load_from_checkpoint(self.args.continue_from)
            
        else: # assume 'continue_from' contains a checkpoint filename or folder
            if os.path.isdir(self.args.continue_from):
                with open(os.path.join(self.args.continue_from, 'checkpoint', 'path_to_best'), 'r') as f:
                    filename = f.read()
            
            elif os.path.isfile(self.args.continue_from):  
                filename = self.args.continue_from
                
            else:  
                raise Exception("Filename / experiment folder not found! Path given: {}".format(self.args.continue_from))
                
            print('FILE', filename)
            self.load_from_checkpoint(filename, load_model_only=True)
        
        return True
    
    
    def get_task_generator(self, set_name, num_tasks, seed):
        return TaskGenerator(self.datasets[set_name],
                             task=self.tasks[set_name],
                             task_args=self.task_args[set_name],
                             num_tasks=num_tasks,
                             seed=seed, 
                             epoch=self.state.epoch, 
                             mode=set_name, 
                             fix_classes=self.args.fix_class_distribution,
                             deterministic=self.args.deterministic)
    
    
    def get_dataloader(self, dataset, sampler, epoch, mode):
        return DataLoader(dataset, sampler, self.device, epoch, mode)
    
    
    def run_experiment(self):
        """
        Runs the main thread of the experiment
        """
        
        continue_from_next_epoch = self.load_pretrained()
        
        if self.args.evaluate_on_test_set_only:
            self.evaluate_on_test()
            return
        
        if continue_from_next_epoch:
            self.ptracker.reset_epoch_cache()
            self.state.next_epoch()
            self.model.next_epoch()
        
        converged = False if self.args.num_epochs is None else self.state.epoch >= self.args.num_epochs
        
        while not converged:
            train_generator = self.get_task_generator(
                'train', 
                self.args.num_tasks_per_epoch, 
                self.state.epoch_seed)
            
            self.ptracker.set_mode('train')
            
            # train
            with tqdm.tqdm( initial=0, total=self.args.num_tasks_per_epoch, disable=self.args.disable_tqdm) as train_pbar:
                for train_sampler in train_generator:
                    dataloader = self.get_dataloader(self.datasets['train'], train_sampler, self.state.epoch, 'train')
                    self.model.meta_train(dataloader, self.ptracker)
                    train_pbar.set_description('Train phase {} -> {} {}'.format(
                        self.state.epoch, self.ptracker.get_performance_str(), self.model.get_summary_str()))
                    train_pbar.update(1)
            
            val_generator = self.get_task_generator(
                self.val_or_test, 
                self.args.num_tasks_per_validation, 
                self.state.epoch_seed)
            
            if self.args.disable_tqdm:
                print('Train phase {} -> {} {}'.format(
                        self.state.epoch, self.ptracker.get_performance_str(), self.model.get_summary_str()))
            
            self.ptracker.set_mode('val')
            
            # simpleshot calc train dataset mean for normalisation during validation
            if self.args.model == 'simpleshot':
                self.model.set_train_mean(self.datasets['train'], disable_tqdm=self.args.disable_tqdm)
            
            # validation
            with tqdm.tqdm( initial=0, total=self.args.num_tasks_per_validation, disable=self.args.disable_tqdm) as pbar_val:
                for val_sampler in val_generator:
                    val_dataloader = self.get_dataloader(self.datasets[self.val_or_test], val_sampler, self.state.epoch, 'val')
                    self.model.meta_val(val_dataloader, self.ptracker)
                    pbar_val.set_description('Val phase {} -> {} {}'.format(
                        self.state.epoch, self.ptracker.get_performance_str(), self.model.get_summary_str()))
                    pbar_val.update(1)
            
            if self.args.disable_tqdm:
                print('Val phase {} -> {} {}'.format(
                        self.state.epoch, self.ptracker.get_performance_str(), self.model.get_summary_str()))
            
            converged = self.save_checkpoint()
            self.ptracker.reset_epoch_cache()  # call after save_checkpoint() otherwise performance will be lost
            self.state.next_epoch()
            self.model.next_epoch()
            print()
        print()
        
        # Evaluate the best model on test dataset
        self.evaluate_best_model()
        
        
    def evaluate_best_model(self):
        """
        Evaluate final performance on the best model
        """
        if self.args.dummy_run:
            self.evaluate_on_test()
            return
            
        # Load best checkpoint path from path_to_best
        path_to_best = os.path.join(self.checkpoint_folder, 'path_to_best')
        if not os.path.exists(path_to_best):
            raise Exception('path_to_best not found: {}'.format(path_to_best))
        with open(path_to_best,'r') as f:
            checkpointfile = f.read()

        self.load_from_checkpoint(checkpointfile)
        self.evaluate_on_test()
        
    
    def evaluate_on_test(self):
        """
        Evaluates the current model on the test set
        """
        self.ptracker.set_mode('test')
        checkpoint_name =  self.template.format(self.state.epoch)
        
        # Get train mean for simpleshot for faster performance
        if self.args.model == 'simpleshot':
            self.model.set_train_mean(self.datasets['train'])
        
        # Evaluate on test (note: seed set to experiment seed, not epoch seed, which allows for fair evaluation)
        generator = self.get_task_generator('test', self.args.num_tasks_per_testing, self.args.seed)
        with tqdm.tqdm(total=self.args.num_tasks_per_testing, disable=self.args.disable_tqdm) as pbar_val:
            for sampler in generator:
                dataloader = self.get_dataloader(self.datasets['test'], sampler, self.state.epoch, 'test')

                self.model.meta_test(dataloader, self.ptracker)
                pbar_val.update(1)
                pbar_val.set_description('Testing ({}) -> {} {}'.format(checkpoint_name,
                                            self.ptracker.get_performance_str(),
                                            self.model.get_summary_str()))
        
        if self.args.disable_tqdm:
            print('Testing ({}) -> {} {}'.format(checkpoint_name,
                                            self.ptracker.get_performance_str(),
                                            self.model.get_summary_str()))
                
        if self.args.dummy_run:  # no saving
            return
        
        performance_logfile = '{}_{}_{}.json'.format(
            os.path.join(self.performance_folder, checkpoint_name),
            self.args.test_performance_tag, 
            self.experiment_timestamp)
        self.ptracker.save_logfile(performance_logfile, ['test'])
        self.ptracker.reset_epoch_cache()
        

class State():
    
    def __init__(self, experiment_seed):
        """
        Keeps track of the current training epoch and seed
        """
        self.epoch = 0
        self.epoch_completed_in_this_run = 0
        self.epoch_rng = np.random.RandomState(experiment_seed)
        self.epoch_seed = self.epoch_rng.randint(999999999)
    
    def next_epoch(self):
        self.epoch += 1
        self.epoch_completed_in_this_run += 1
        self.epoch_seed = self.epoch_rng.randint(999999999)
        
    def to_dict(self):
        return {
            'epoch_seed': self.epoch_seed,
            'epoch': self.epoch,
            'epoch_rng': self.epoch_rng
        }
    
    def from_dict(self, adict):
        self.epoch_seed = adict['epoch_seed']
        self.epoch = adict['epoch']
        self.epoch_rng = adict['epoch_rng']
    
        