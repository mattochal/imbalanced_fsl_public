# Dataset Instructions

Our framework reads pickle (```.pkl```) dataset files with the following structure:

```
{
    "image_data": <array of shape (n, 84, 84, 3) >,
    "class_dict": <dict with a class name as a key, and a list of corresponding image indexes as value>
}
```
where `n` is the total number of images in dataset.

You can create the dataset files from a subdirectory structure using ```create_cache_from_folders.py``` (or write your own method to create the pickle files). 

## Create from class subdirectory structure
The images should be contained in class subfolders, and the class subfolders contained in: ```train/```, ```val/``` and ```test/``` folders, i.e. the following directory structure should be located inside ```./data/```:
```
.<DATASET>/
└── raw/
    ├── train/
    |    ├── class1/
    |    |   ├─ image1.jpg
    |    |   ├─ image2.jpg
    |    |   ...
    |    ├── class2/
    |    |   ├─ image1.jpg
    |    |   ├─ image2.jpg
    |    |   ...
    |    ...
    |
    ├── test/
    |    ├── class3/
    |    |   ├─ image1.jpg
    |    |   ├─ image2.jpg
    |    |   ...
    |    ├── class4/
    |    |   ├─ image1.jpg
    |    |   ├─ image2.jpg
    |    |   ...
    |    ...
    |
    └── val/
         ├── class5/
         |   ├─ image1.jpg
         |   ├─ image2.jpg
         |   ...
         ├── class6/
         |   ├─ image1.jpg
         |   ├─ image2.jpg
         |   ...
         ...
```
where `<DATASET>` is `mini` or `cub`. Classes and images can have arbitary names. Accepted extentions: '.jpg', '.jpeg', '.gif', '.png', '.JPEG' 

To create dataset from the subdirectories run:
```
create_cache_from_folders.py `<DATASET>`
```


### MiniImageNet

MiniImageNet can be downloaded and created using: https://github.com/yaoyao-liu/mini-imagenet-tools 

To run main experiments, `./data/mini/` should contain the following files:
```
mini-cache-train.pkl
mini-cache-test.pkl
mini-cache-val.pkl
```

### CUB-200 2011

[Download from official source](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

Randomly select 50 classes for testing, and arbirary split the remaining classes for validation and training.

To run meta-dataset inference experiments, `./data/cub/` should contain the following files:
```
cub-cache-train.pkl
cub-cache-test.pkl
cub-cache-val.pkl
```


