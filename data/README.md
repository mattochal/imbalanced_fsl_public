# Dataset Instructions

*Automatic download and dataset creation hidden due to anonymity*

All dataset files should have the following data structure:

```
{
    "image_data": <array of shape (n_total, 84, 84, 3) >,
    "class_dict": <dict>
}
```
where `n` is the total number of images, and `<dict>` with keys representing class names, and values indicating indices of corresponding of class images in `image_data`.

### MiniImageNet
MiniImageNet should have the following three files 
```
mini-cache-train.pkl
mini-cache-test.pkl
mini-cache-val.pkl
```
These should be placed in `./data/mini/`

### CUB-200 2011
CUB should be the following files placed in `./data/cub/`
```
cub-cache-train.pkl
cub-cache-test.pkl
cub-cache-val.pkl
```


## Create from class subdirectory stracture

You can also create the files from class substracture, containing images in classes for each of the dataset splits: train, val and test. Images must be placed under the following directory structure in `data/` folder of the repository:
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
where `<DATASET>` is `mini` or `cub`, and classes and images can have arbitary names, images can have any of the following extentions: '.jpg', '.jpeg', '.gif', '.png', '.JPEG' 

To run
```
create_cache_from_folders.py `<DATASET>`
```