from datasets.dataset_utils import load_dataset_from_pkl, load_dataset_from_from_folder 
from datasets.dataset_template import ColorDatasetInMemory, ColorDatasetOnDisk
import os
import numpy as np

def get_MiniImageNet(args_per_set, setnames=["train", "val", "test"]):
    """
    Returns MiniImagenet datasets.
    """
    datasets = {}
    for setname in setnames:
        args = args_per_set[setname]
        
        if args.dataset_version not in [None, "ravi", "from_folder", "step-animal"]:
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

        elif args.dataset_version in ["step-animal"] and setname in ["train"]:
            filepath = os.path.join(data_path, "mini", "mini-cache-{0}.pkl".format(setname))
            data = load_dataset_from_pkl(filepath)
            data = animalreduced_step_imbalance_miniimagenet(data, args)
            dataset_class = ColorDatasetInMemory
                
        datasets[setname] = [data['image_data'], data['class_dict'], args, dataset_class]
        
    return datasets

def animalreduced_step_imbalance_miniimagenet(data, args):

    animal_classes = list(map(lambda x: x.split()[0], """n02606052 rock_beauty # animal 
n02108089 boxer # animal 
n04275548 spider_web # animal 
n02120079 Arctic_fox # animal 
n01910747 jellyfish # animal 
n02457408 three-toed_sloth # animal
n02108915 French_bulldog # animal 
n01770081 harvestman # animal 
n01749939 green_mamba # animal 
n02111277 Newfoundland # animal 
n01704323 triceratops # animal (deceased)
n02113712 miniature_poodle # animal 
n02091831 Saluki # animal 
n02089867 Walker_hound # animal 
n02165456 ladybug # animal 
n02101006 Gordon_setter # animal 
n02074367 dugong # animal 
n02105505 komondor # animal 
n02108551 Tibetan_mastiff # animal 
n01532829 house_finch # bird
n01558993 robin # bird
n01843383 toucan # bird """.split('\n')))

    # for cls in data["class_dict"].keys():

    #     if cls in animal_classes:
    #         print(cls)
    #         idx = data["class_dict"][cls]
    #         selected = np.random.choice(idx, 25, replace=False)
    #         data["class_dict"][cls] = selected
    #     else:
    #         idx = data["class_dict"][cls]
    #         selected = np.random.choice(idx, 444, replace=False)
    #         data["class_dict"][cls] = selected


    # delete images and shift index
    class_labels = list(sorted(data["class_dict"].keys()))
    rng = np.random.RandomState(args.seed)
    class_dict = data["class_dict"]
    image_data = data["image_data"]

    new_image_data = []
    new_class_dict = {}
    index_offset = 0
    
    for l, label in enumerate(class_labels):
        class_idx = np.array(class_dict[label])
        n = 25 if label in animal_classes else 444
        selected_idx = rng.choice(class_idx, n, replace=False)
        new_image_data.append(image_data[selected_idx])
        new_class_dict[label] = index_offset + np.arange(n)
        index_offset += n

    new_image_data = np.vstack(new_image_data)

    new_data = {}
    new_data['class_dict']= new_class_dict
    new_data['image_data']= new_image_data

    return new_data


def __animalreduced_step_imbalance_miniimagenet(data):

    animal_classes = list(map(lambda x: x.split()[0], """n02606052 rock_beauty # animal 
n02108089 boxer # animal 
n04275548 spider_web # animal 
n02120079 Arctic_fox # animal 
n01910747 jellyfish # animal 
n02457408 three-toed_sloth # animal
n02108915 French_bulldog # animal 
n01770081 harvestman # animal 
n01749939 green_mamba # animal 
n02111277 Newfoundland # animal 
n01704323 triceratops # animal (deceased)
n02113712 miniature_poodle # animal 
n02091831 Saluki # animal 
n02089867 Walker_hound # animal 
n02165456 ladybug # animal 
n02101006 Gordon_setter # animal 
n02074367 dugong # animal 
n02105505 komondor # animal 
n02108551 Tibetan_mastiff # animal 
n01532829 house_finch # bird
n01558993 robin # bird
n01843383 toucan # bird """.split('\n')))

    for cls in data["class_dict"].keys():

        if cls in animal_classes:
            print(cls)
            idx = data["class_dict"][cls]
            selected = np.random.choice(idx, 25, replace=False)
            data["class_dict"][cls] = selected
        else:
            #print(cls)
            idx = data["class_dict"][cls]
            selected = np.random.choice(idx, 444, replace=False)
            data["class_dict"][cls] = selected

    return data
