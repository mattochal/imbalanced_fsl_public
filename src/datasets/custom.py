from datasets.dataset_utils import load_dataset_from_from_folder
from datasets.dataset_template import ColorDatasetInMemory, ColorDatasetOnDisk
from torchvision import transforms
import os

def get_custom_dataset_from_folders(args_per_set, setnames=["train", "val", "test"]):
    """
    Returns a datasets from folder "train", "val", "test" folder structure
    """
    
    datasets = {}
    for setname in setnames:
        args = args_per_set[setname]
        
        if args.dataset_version != None:
            raise Exception("Custom dataset version '{}' does not exist".format(version))
        
        datasetpath = os.path.join(args.data_path, "custom", setname)
        augment = args.augment
        
        if args.use_cache:
            data = load_dataset_from_from_folder(datasetpath, use_cache=True, image_size=(args.image_height, args.image_width))
            dataset = ColorDatasetInMemory(data["image_data"], data["class_dict"], args)
            
        else:
            data_paths = load_dataset_from_from_folder(datasetpath, use_cache=False)
            dataset = ColorDatasetOnDisk(data_paths["image_data"],  data_paths["class_dict"], args, folder_path=setname)
            
        datasets[setname] = dataset
    return datasets