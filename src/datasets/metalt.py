from datasets.dataset_utils import load_dataset_from_pkl, load_dataset_from_from_folder 
from datasets.dataset_template import ColorDatasetInMemory, ColorDatasetOnDisk
import os

def get_MetaImageNetLongTail(args_per_set, setnames=["train", "val", "test"]):
    """
    Returns MiniImagenet datasets.
    """
    datasets = {}
    for setname in setnames:
        args = args_per_set[setname]
        
        if args.dataset_version not in [None]:
            raise Exception("Dataset version not found {}".format(args.dataset_version))
            
        data_path = os.path.abspath(args.data_path)
        
        if args.dataset_version in [None]:
            filepath = os.path.join(data_path, "metalt", "metalt-cache-{0}.pkl".format(setname))
            data = load_dataset_from_pkl(filepath)
            dataset_class = ColorDatasetInMemory
            
        datasets[setname] = [data['image_data'], data['class_dict'], args, dataset_class]
        
    return datasets