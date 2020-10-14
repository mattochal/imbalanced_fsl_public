import numpy as np
import torch
import os
import json
from sklearn.metrics import precision_recall_fscore_support
import argparse
import copy
from collections import deque

def compute_accuracy(y_pred, y_true):
    size_pred = np.shape(y_pred)
    size_true = np.shape(y_true)
    
    if len(size_pred) == 1:
        n = size_pred[0]
        acc = np.sum(y_pred==y_true) * 1.0 / n
        
    elif len(size_pred) == 2:
        n = size_pred[0]
        acc = np.sum(np.all(y_pred==y_true, 1)) * 1.0 / n
    
    return acc

def update_confusion_matrix(y_pred, y_true, conf_matrix, normalise=True):
    for y, y_p in zip(y_true, y_pred):
        conf_matrix[y, y_p] += 1
        
    if normalise:
        conf_matrix /= conf_matrix.sum()
        
    return conf_matrix


def update_class_freq(labels, class_freq, normalise=False):
    for y in labels:
        class_freq[y, y_p] += 1
        
    if normalise:
        class_freq /= class_freq.sum()
        
    return conf_matrix
    
def update_confusion_matrix_with_checks(y_pred, y_true, conf_matrix, normalise=True, max_lbl=None):
    if max_lbl is None:
        max_lbl = max(max(y_pred), max(y_true))
    conf_len = np.shape(conf_matrix)[0]
    if conf_len <= max_lbl:
        new_conf_matrix=np.zeros((max_lbl+1, max_lbl+1))
        new_conf_matrix[:conf_len, :conf_len] = conf_matrix
        cm = update_confusion_matrix(y_pred, y_true, new_conf_matrix)
    else:
        cm = update_confusion_matrix(y_pred, y_true, conf_matrix)
    if normalise:
        cm = cm / cm.sum()
    return cm

def confusion_matrix_combine(conf_matrices, normalise=True):
    lens = [np.shape(cm)[0] for cm in conf_matrices]
    max_len = max(lens)
    new_cm = np.zeros((max_len, max_len))
    for i, cm in enumerate(conf_matrices):
        cm_len = lens[i]
        new_cm[:cm_len, :cm_len] += cm
    if normalise:
        new_cm /= np.sum(new_cm)
    return new_cm
    
def save_conf_matrix_visualisation(conf_matrix, filepath):
    """
    Saves visualisation of the confusion matrix saved in the task performance
    """
    n_row, n_col = np.shape(conf_matrix)
    df_cm = pd.DataFrame(array, index=np.arange(n_row), columns=np.arange(n_col))
    plt.figure(figsize = (35,35))
    ax = sn.heatmap(df_cm, annot=True)
    fig = ax.get_figure()
    fig.savefig(filepath)
    
def class_freq_combine(class_freqs, normalise=False):
    lens = [np.shape(cf)[0] for cf in class_freqs]
    max_len = max(lens)
    new_cf = np.zeros(max_len)
    for i, cf in enumerate(class_freqs):
        cf_len = lens[i]
        new_cf[:cm_len] += cf
    if normalise:
        new_cf /= np.sum(new_cf)
    return new_cf

def subtract_confusion_matrices(conf_matrix1, conf_matrix2):
    shape1 = np.shape(conf_matrix1)
    shape2 = np.shape(conf_matrix2)
    if shape1[0] > shape2[0]:
        conf_matrix1 = conf_matrix1[:shape2[0], :shape2[0]]
    elif shape1[0] < shape2[0]:
        conf_matrix2 = conf_matrix2[:shape1[0], :shape1[0]]
    return conf_matrix1 - conf_matrix2

def compute_errors(conf_matrix):
    TP = np.diag(conf_matrix)
    FN = np.sum(conf_matrix - np.diag(TP), axis=1)
    FP = np.sum(conf_matrix - np.diag(TP), axis=0)
    TN = np.sum(conf_matrix) - (TP + FN + FP)
    return TP, TN, FP, FN

def compute_precision_and_recall(TP, TN, FP, FN):
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return precision, recall

def compute_precision_and_recall_from_matrix(conf_matrix):
    TP, TN, FP, FN = compute_errors(conf_matrix)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return precision, recall

def compute_per_class_accuracies_from_matrix(conf_matrix):
    TP, TN, FP, FN = compute_errors(conf_matrix)
    return (TP+TN)/(TP+FP+FN+TN)
    
def get_bwt(task_performances):
    prev_task = task_performances[0]
    acc = []
    for task in task_performances[1:]:
        acc.append(prev_task['accuracy'] - task['accuracy'])
        prev_task = task
    return np.mean(acc)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

    
class TaskPerformance(dict):
    
    METRICS = ["preds", "true", "loss", "accuracy", "conf_matrix", "class_freq", "per_cls_stats"]
    
    def __init__(self, y_pred, y_true, loss, metrics, other_metrics_dict={}):
        super().__init__()
        
        if "preds" in metrics:
            self["preds"]=list(y_pred)
            
        if "true" in metrics:
            self["true"]=list(y_true)
        
        if "loss" in metrics: 
            self["loss"]=float(loss)
            
        if "accuracy" in metrics:
            self["accuracy"]=float(compute_accuracy(y_pred, y_true))
        
        if "conf_matrix" in metrics:
            max_lbl = max(max(y_pred), max(y_true))
            conf_matrix=np.zeros((max_lbl+1, max_lbl+1))
            conf_matrix=update_confusion_matrix(y_pred, y_true, conf_matrix)
            self["conf_matrix"]=conf_matrix
        
        if "class_freq" in metrics:
            max_lbl = max(y_true)
            class_freq = np.zeros(max_lbl+1)
            class_freq=update_class_freq(y_true, class_freq)
            self["class_freq"]=class_freq
        
        if "per_cls_stats" in metrics:
            output = precision_recall_fscore_support(y_true, y_pred, beta=1.0, zero_division=0)
            self["precision"]=output[0]
            self["recall"]=output[1]
            self["f1"]=output[2]
        
        self.update(other_metrics_dict)
    
    def metrics(self):
        return list(self.keys())
    
    def to_dict(self):
        return self
    

class PerformanceBatch(dict):
    
    def __init__(self, performances=None):
        """
        Holds TaskPerformance objects and averages the metrics over the given TaskPerformance objects. A PerformanceBatch
        handles the performances obtained from within a single epoch - either from the training iterations or evaluation 
        tasks, also See PerformanceTracker. Assumes all TaskPerformances within the batch contain the same metrics.
        :param performances: a list of TaskPerformance objects
        """
        super().__init__()
        if performances is not None:
            # summarise the performances 
            metrics = performances[0].metrics()
            for m in metrics:
                self['avr_' + m] = np.mean([tp[m] for tp in performances], axis=0)
            self["performances"] = performances
        else:
            self["performances"] = []
            
        self['num_performances'] = len(self['performances'])
        
    def add_performance(self, p):
        self['performances'].append(p)
        self['num_performances'] += 1
        n = self['num_performances']
        
        # incremental average for each metric
        for m in p.metrics():
            if 'avr_' + m not in self:
                self['avr_' + m] = p[m]
                if "conf_matrix" not in m:
                    self['var_' + m] = 0.0
            elif "conf_matrix" in m:
                self['avr_' + m] = confusion_matrix_combine([p[m], (n-1) * self['avr_' + m]], normalise=False) / n
            else:
                self['var_' + m] = ((n - 2.)/(n - 1.)) * self['var_' + m] + (1./n) * (p[m] - self['avr_' + m])**2
                self['avr_' + m] += (p[m] - self['avr_' + m]) / n
        
    def to_dict(self, with_performances=True):
        """
        
        """
        if len(self['performances']) > 0:
            p = self['performances'][0]
            for m in p.metrics():
                if 'conf_matrix' not in m and 'precision' not in m and 'recall' not in m and 'f1' not in m:
                    np_mean = np.mean([p[m] for p in self['performances']], axis=0)
                    assert np.isclose(np_mean, self['avr_' + m]) or (np.isnan(self['avr_'+m]) and np.isnan(np_mean))
        
        if not with_performances:
            temp = self.copy() # shallow copy
            del temp['performances']
            return temp
        else:
            return self
        
    def from_dict(self, dct):
        self.update(dct)
        return self

    
    
class PerformanceTracker():
    
    @staticmethod
    def get_parser(parser=None):
        if parser is None: parser = argparse.ArgumentParser()
        
        parser.add_argument('--save_task_performance', type=bool, default=False, 
                            help='Saves performance for individual task in the ptracker log files, not just the average')
        parser.add_argument('--metrics', type=list, default=['accuracy', 'loss'],
                            help="Metrics to assess performance for each. Choices={}".format(TaskPerformance.METRICS))
        return parser
    
    def __init__(self, folder=None, args=None):
        """
        Keeps track of performance for epochs.
        :param folder: folder where to save performance statistics, typically in the form of [experiment_name]/[folder]/
        """
        self.folder = folder
        self.args = args
        
        # Checks validity of given metric names
        for s in self.args:
            for m in self.args[s]['metrics']:
                if m not in TaskPerformance.METRICS:
                    raise Exception("metrics.{}.{} not found. Choices={}".format(s,m,TaskPerformance.METRICS))
            
        self.mode='train' # or 'val' or 'test'
        self.epoch_cache = {"train":PerformanceBatch(), 
                            "val":PerformanceBatch(), 
                            "test":PerformanceBatch()}
        
        self.deque_maxlen = 5
        
        self.current_best  = {
            "train":{'acc':0.0, 'path':None, 'epoch':-1}, 
            "val":{'acc':0.0, 'path':None, 'epoch':-1},
            "test":{'acc':0.0, 'path':None, 'epoch':-1}}
        
        # Stores previous 5 best performances
        self.previous_bests = {
            "train":deque(maxlen=self.deque_maxlen), 
            "val":deque(maxlen=self.deque_maxlen),
            "test":deque(maxlen=self.deque_maxlen)
        }
        
        self.lastest_task_performance = None
        
    def set_mode(self, mode):
        self.mode = mode
        
    def add_task_performance(self, y_pred, y_true, loss, other_metrics_dict={}):
        self.lastest_task_performance = TaskPerformance(y_pred, y_true, loss, metrics=self.args[self.mode]['metrics'], 
                                                        other_metrics_dict=other_metrics_dict)
        self.epoch_cache[self.mode].add_performance(self.lastest_task_performance)
        
    def get_lastest_task_performance(self):
        return self.lastest_task_performance
        
    def update_best(self, checkpointpath, current_epoch, metric='avr_accuracy'):
        """
        Updates best statistics for the epoch/checkpoint.
        :param checkpointpath: path of the current epoch
        :param metric: metric to evaluate the best performance by 
        """
        is_updated={}
        for setname in ['train', 'val', 'test']:
            
            is_updated[setname]=False
            if metric not in self.epoch_cache[setname]:
                continue
            
            if self.epoch_cache[setname][metric] >= self.current_best[setname]['acc']:
                self.previous_bests[setname].append(copy.copy(self.current_best[setname]))
                self.current_best[setname]['acc'] = self.epoch_cache[setname][metric]
                self.current_best[setname]['path'] = checkpointpath
                self.current_best[setname]['epoch'] = current_epoch
                is_updated[setname]=True
        
        return is_updated
    
    def reset_epoch_cache(self):  
        """ Resets the epoch cache. Call this before the next epoch, but after save_to_logfile() otherwise all the performances 
        logged during the epoch will never be recovered. """
        self.epoch_cache = {"train":PerformanceBatch(), 
                            "val":PerformanceBatch(), 
                            "test":PerformanceBatch()}
    
    def get_performance_str(self, metrics=['avr_accuracy', 'avr_loss'], 
                                       abbrev={'avr_accuracy':'acc','avr_loss':'loss'}):
        """
        Returns the accumulated performances so far in the format of a string
        :param metrics: metrics to be returned in from of a string
        :param abbrev: abbreviations for the metric names to display for easier viewing
        """
        mystr = ""
        if self.mode == 'val':
            mystr += "best_epoch={} best_acc={:.3f} ".format(
                self.current_best[self.mode]['epoch'], 
                self.current_best[self.mode]['acc']
            )
        
        if self.epoch_cache[self.mode]["num_performances"] > 0:
            for m in metrics:
                mystr += "{}={:.3f} ".format(abbrev[m], self.epoch_cache[self.mode][m])
            return mystr
        else:
            return "Use ptracker.add_task_performance() to track performance! "
    
    def save_logfile(self, filepath, setnames=[]):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(setnames), f, indent=2, cls=NumpyEncoder) # NumpyEncoder turns numpy arrays into lists
        return filepath
        
    def load_from_logfile(self, filepath):
        with open(filepath, 'r') as f:
            # note: numpy objects should be loaded as numpy objects again - currently they are lists.
            self.from_dict(json.load(f))
    
    def to_dict(self, setnames=["train", "val", "test"]):
        """
        Turn into dict object
        """
        dct = {s:{} for s in setnames}
        
        dct['current_best'] = self.current_best
        dct['previous_bests'] = {
            'train' : list(self.previous_bests['train']),
            'val' : list(self.previous_bests['val']),
            'test' : list(self.previous_bests['test'])
        }
        
        for s in setnames:
            dct[s].update({
                "best_acc": self.current_best[s]['acc'],
                "best_epoch_checkpoint_path": self.current_best[s]['path']
            })
            epoch_performance = self.epoch_cache[s].to_dict(with_performances=self.args[s]['save_task_performance'])
            dct[s].update(epoch_performance)
            if s == 'train':
                if np.isinf(epoch_performance['avr_loss']):
                    raise Exception("Loss is out of control.")
            
        return dct
    
    def from_dict(self, dct, reset_epoch=False):
        """
        Load from dict object
        """
        if 'current_best' in dct:
            self.current_best =  dct['current_best']
            del dct['current_best']
        
        if 'previous_bests' in dct:
            pb_dct = dct['previous_bests']
            self.previous_bests = {
                'train' : deque(pb_dct['train'], maxlen=self.deque_maxlen),
                'val' : deque(pb_dct['val'], maxlen=self.deque_maxlen),
                'test' : deque(pb_dct['test'], maxlen=self.deque_maxlen)
            }
            del dct['previous_bests']
        
        for setname in dct:
            self.current_best[setname]['acc'] = dct[setname]["best_acc"]
            self.current_best[setname]['path'] = dct[setname]["best_epoch_checkpoint_path"]
            
            if reset_epoch:
                del dct[setname]["best_acc"]
                del dct[setname]["best_epoch_checkpoint_path"]
                self.epoch_cache[setname] = PerformanceBatch().from_dict(dct[setname])
                
        return dct