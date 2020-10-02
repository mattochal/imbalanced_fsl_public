import os, sys
sys.path.insert(0,'../src/')
from datasets.dataset_utils import load_dataset_from_from_folder

if __name__ == "__main__":
    dataset = sys.argv[1]
    for split in ["train","test","val"]:
        source = os.path.join("./",dataset,"raw",split)
        assert os.path.isdir(source), "source not found: {}".format(source)
        dest = os.path.join("./",dataset,"{}-cache-{}.pkl".format(dataset,split))
        load_dataset_from_from_folder(source, dest, use_cache=True, image_size=(84,84), use_cache_if_exists=False)