from datasets.dataset_utils import load_dataset_from_pkl, load_dataset_from_from_folder 
from datasets.dataset_template import ColorDatasetInMemory, ColorDatasetOnDisk
from datasets.mini import get_MiniImageNet
from datasets.cub import get_CUB200
import os


def get_MiniImageNet_to_CUB200(args_per_set):
    """
    Returns MiniImagenet datasets.
    """
    datasets = {}
    datasets.update(get_MiniImageNet(args_per_set, setnames=["train", "val"]))
    datasets.update(get_CUB200(args_per_set, setnames=["test"]))
    return datasets