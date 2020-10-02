# Imbalanced Few-Shot Learning Framework in Pytorch


### Dependecies


Our code was tested on Ubunto ### and ### 
* numpy
* pandas
* python 3.5+
* pytorch ###
* tqdm


To set up the specific conda environment run:
```
conda create --name imb_fsl --python 3.5
conda install 

```

Code was tested on 


### Datasets

See ```./data/README.md```

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

## Generating Main Experiments

This repository contains code for "Class-Imbalance in Few-Shot Learning".

To generate the experiment scripts and files for the main imbalanced support set experiments in the paper:
```
python generator.py --imbalanced_task
python generator.py --imbalanced_dataset
```
Add `--minimal` flag to generate a reduced subset of experiments.

Add `--gpu <GPU#>` to specify the GPU ID or `cpu`

To generate the evaluation scripts for imbalanced support set:
```
python generator.py --imbalanced_task --test
```

For ROS\+ inference on imbalanced support sets run:
```
python generator.py --imbalanced_task --inference
```

For CUB inference on imbalanced datasets run:
```
python generator.py --imbalanced_dataset --inference
```

More details can also be obtained through the `--help` command.


### Running main program

To run a specific experiment setting from a configuration file:
```
python src/main.py --args_file <CONFIGPATH> --gpu <GPU>
```

Arguments from the `CONFIGPATH` can be overwriten by arguments passed through the command line.

Run ```python main.py --help``` for general help

For sepecific model/task/stategy arguments substitute key words and run any of the following:

```
python main.py  --model <MODEL> --help_model
python main.py  --task <TASK> --help_task
python main.py  --strategy <STRATEGY> --help_stategy
python main.py  --model <MODEL>   --task <TASK>  --task <STRATEGY>   --help_all```
```


### Contributions
This repository contains parts of taken code from the following GITHUB repositories:
https://github.com/wyharveychen/CloserLookFewShot/
https://github.com/jakesnell/ove-polya-gamma-gp/
https://github.com/BayesWatch/deep-kernel-transfer/
https://github.com/haebeom-lee/l2b 
https://github.com/katerakelly/pytorch-maml 
https://github.com/dragen1860/MAML-Pytorch
https://github.com/cnguyen10/few_shot_meta_learning
