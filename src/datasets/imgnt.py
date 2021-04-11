from datasets.dataset_utils import load_dataset_from_pkl, load_dataset_from_from_folder 
from datasets.dataset_template import ColorDatasetInMemory, ColorDatasetOnDisk
import os

def get_ImageNet(args_per_set, setnames=["train", "val", "test"]):
    """
    Returns MiniImagenet datasets.
    """
    datasets = {}
    for setname in setnames:
        args = args_per_set[setname]
        
        if args.dataset_version not in ['random','longtail','balanced']:
            raise Exception("Dataset version not found {}".format(args.dataset_version))
            
        data_path = os.path.abspath(args.data_path)
        version = args.dataset_version if setname == 'train' else 'balanced'
        if version == 'random':
            filepath = os.path.join(data_path, "imgnt", "imgnt-{0}-{1}-cache-{2}.pkl".format(version,args.seed,setname))
        else:
            filepath = os.path.join(data_path, "imgnt", "imgnt-{0}-cache-{1}.pkl".format(version,setname))
        data = load_dataset_from_pkl(filepath)
        dataset_class = ColorDatasetInMemory
        datasets[setname] = [data['image_data'], data['class_dict'], args, dataset_class]
        
    return datasets