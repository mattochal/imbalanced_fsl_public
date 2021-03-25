from utils.utils import set_torch_seed, set_gpu, get_tasks, get_data, get_model, get_backbone, get_strategy
from utils.utils import compress_args, get_args, torch_summarize
from utils.builder import ExperimentBuilder
from utils.bunch import bunch
import sys
import pprint

if __name__ == '__main__':
    
    args, excluded_args, parser = get_args()
    args = bunch.bunchify(args)
    
    set_torch_seed(args.seed)
    device = set_gpu(args.gpu)
    
    datasets = get_data(args)
    tasks    = get_tasks(args)
    backbone = get_backbone(args, device)
    strategy = get_strategy(args, device)
    model    = get_model(backbone, tasks, datasets, strategy, args, device)
    
    compressed_args = compress_args(bunch.unbunchify(args), parser)
    print(" ----------------- FULL ARGS (COMPACT) ----------------")
    pprint.pprint(compressed_args, indent=2)
    print(" ------------------------------------------------------")
    print(" ------------------ UNRECOGNISED ARGS -----------------")
    pprint.pprint(excluded_args, indent=2)
    print(" ------------------------------------------------------")
    
    system = ExperimentBuilder(model, tasks, datasets, device, args)
    system.load_pretrained()
    system.train_model()
    system.evaluate_model()
        