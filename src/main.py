from utils.utils import set_torch_seed, set_gpu, get_tasks, get_data, get_model, get_backbone, get_strategy, get_main_parser
from utils.utils import compress_and_print, get_raw_args, torch_summarize
from utils.builder import ExperimentBuilder
import sys
import pprint

if __name__ == '__main__':
    parser = get_main_parser()
    args = get_raw_args(parser)
    
    set_torch_seed(args.seed)
    device = set_gpu(args.gpu)
    
    datasets = get_data(args)
    tasks    = get_tasks(args)
    backbone = get_backbone(args, device)
    strategy = get_strategy(args, device)
    model    = get_model(backbone, tasks, datasets, strategy, args, device)
    
    compress_and_print(args)
    
    system = ExperimentBuilder(model, tasks, datasets, device, args)
    system.run_experiment()