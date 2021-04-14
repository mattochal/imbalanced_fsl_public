# Few-Shot Learning with Class Imbalance
This repository contains code for:
 * "Few-Shot Learning with Class Imbalance" [(arXiv)](http://arxiv.org/abs/2101.02523).
 * "How Sensitive are Meta-Learners to Dataset Imbalance?" [(ICLR2021 Workshop)](https://openreview.net/forum?id=-vafHc91wO8&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2021%2FWorkshop%2FLearning_to_Learn%2FAuthors%23your-submissions)) [(arXiv)](http://arxiv.org/abs/2104.05344).

![Task Imbalance](https://github.com/mattochal/imbalanced_fsl_public/blob/master/imbalance_diagram.png?raw=true)

This respository is also compatible with my demo app for Few-Shot Learning [(available here)](http://github.com/mattochal/demo_fsl_public).

<!-- ![GUI Screenshot](https://github.com/mattochal/demo_fsl_dev/blob/master/demo.gif?raw=true) -->

## UPDATES
\[11 April 2021\] - Major update including corrections to BTAML, ProtoMAML, experiments on long-tail distribution, task-by-task demo app compatibility. 


### Dependecies

* numpy
* python 3.8+
* pytorch
* tqdm
* pillow
* scikit-learn
* gpytorch


Our code was tested on Ubuntu 16.04.7 LTS, cuda release 10.1, V10.1.243


To set up the specific conda environment run:
```
conda create -n ci_fsl python=3.8
conda activate ci_fsl
conda install pytorch torchvision -c pytorch
conda install gpytorch -c gpytorch
conda install -c conda-forge tqdm
conda install -c anaconda pillow scikit-learn
```

### Structure

The framework is structured as follows:

```
.
├── generator.py          # Experiment generator to reproduce settings from the paper
├── data/                 # Default data source
├── [experiments/]        # Default script, config and results destination
└── src
    ├── main.py           # Main program
    ├── datasets          # Code for loading datasets
    ├── models            # FSL methods, baselines, backbones
    ├── strategies        # Imbalance strategies
    ├── tasks             # Standard FSL, Imbalanced FSL tasks
    └── utils             # Utils, experiment builder, performance tracker, dataloader
```

### Data

See ```./data/README.md```



## Generating Main Experiments

To generate the experiment scripts and files for the main experiments in the paper:
```
python generator.py --imbalanced_supports
python generator.py --imbalanced_dataset
```
Add ```--minimal``` flag to generate a reduced subset of experiments.

Add ```--gpu <GPU>``` to specify the GPU ID or ```cpu```

To generate the evaluation scripts for imbalanced support set:
```
python generator.py --imbalanced_supports --test
```

For ROS/ROS+ inference on imbalanced support sets run:
```
python generator.py --imbalanced_supports --inference
```

For CUB inference on imbalanced datasets run:
```
python generator.py --imbalanced_dataset --inference
```

More details can also be obtained through the ```--help``` command.


## Running main program

To run a specific experiment setting from a configuration file:
```
python src/main.py --args_file <CONFIGPATH> --gpu <GPU>
```
<!--
Arguments from the ```CONFIGPATH``` can be overwriten by arguments passed through the command line.

Run ```python src/main.py --help``` for general help

For sepecific model/task/stategy arguments substitute key words and run any of the following:

```
python src/main.py  --model <MODEL> --help_model
python src/main.py  --task <TASK> --help_task
python src/main.py  --strategy <STRATEGY> --help_stategy
python src/main.py  --model <MODEL>   --task <TASK>  --task <STRATEGY>   --help_all```
```
-->

____

### Contributions
This repository contains parts of code from the following GITHUB repositories:

https://github.com/dsc/bunch

https://github.com/wyharveychen/CloserLookFewShot/

https://github.com/jakesnell/ove-polya-gamma-gp/

https://github.com/BayesWatch/deep-kernel-transfer/

https://github.com/haebeom-lee/l2b 

https://github.com/katerakelly/pytorch-maml 

https://github.com/dragen1860/MAML-Pytorch

https://github.com/cnguyen10/few_shot_meta_learning
