from datasets.dataset_utils import load_dataset_from_pkl, load_dataset_from_from_folder 
from datasets.dataset_template import ColorDatasetInMemory, ColorDatasetOnDisk
import os

def get_MiniImageNet(args_per_set, setnames=["train", "val", "test"]):
    """
    Returns MiniImagenet datasets.
    """
    datasets = {}
    for setname in setnames:
        args = args_per_set[setname]
        
        if args.dataset_version not in [None, "ravi", "from_folder"]:
            raise Exception("Dataset version not found {}".format(args.dataset_version))
        
        data_path = os.path.abspath(args.data_path)
        
        if args.dataset_version in [None, "ravi"]:
            filepath = os.path.join(data_path, "mini", "mini-cache-{0}.pkl".format(setname))
            data = load_dataset_from_pkl(filepath)
            dataset_class = ColorDatasetInMemory
            
        elif args.dataset_version in ["from_folder"]:
            filepath = os.path.join(data_path, "mini_from_folder", setname)
            data = load_dataset_from_from_folder(filepath, use_cache=args.use_cache)
            dataset_class = ColorDatasetInMemory if args.use_cache else ColorDatasetOnDisk
                
        datasets[setname] = [data['image_data'], data['class_dict'], args, dataset_class]
        
    return datasets