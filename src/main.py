# Data transformation (into tensor and torch.utils.data.Dataset -> torch.utils.data.DataLoader)
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List
from PIL import Image 
import matplotlib.pyplot as plt

import torch 
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from secondary_module import read_yaml
# Assuming data_exploration.py is in src\main.py
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config = read_yaml(os.path.join(project_root_path, 'conf', 'config.yml'))


class OurCustomDataset(Dataset):
    '''
    Inherit from torch.utils.data.Dataset to create custom Dataset that replicate ImageFolder (torchvision.datasets.ImageFolder)
    '''
    def __init__(self,
                 root: str,
                 transform: List = None,
                 target_transform = None):
        
        self.root = root
        if not os.path.exists(self.root): raise FileNotFoundError(f"The root path is not valid: {self.root}")
        self.transform = transform
        self.target_transform = target_transform
        # All .jpg images in root dir (expect following path format: root/class/image.jpg)
        self.all_img_paths_in_root = list(Path(self.root).glob("*/*.jpg"))
        # Run get_classes method during initialization to get self.classes and self.classes_dict
        self.classes, self.class_to_idx = self.get_classes()

    def get_classes(self) -> Tuple[List[str], Dict[str, int]]:
        # Get the class names (dir names in self.root)
        classes = sorted([entry.name for entry in list(os.scandir(self.root)) if entry.is_dir()])
        # Get the dict of classes and associated labels
        class_to_idx = {this_class:label for label, this_class in enumerate(classes)}
        return classes, class_to_idx
    
    def load_image(self, index: int):
        return Image.open(self.all_img_paths_in_root[index])
    
    def convert_mode_L_to_RGB(self, image):
        # Convert to RGB if the mode is not already RGB
        return image.convert('RGB') if image.mode != 'RGB' else image

    def __len__(self) -> int:
        # Overwrite Dataset's __len__ method with the len of the list of all .jpg file found in root dir
        return len(self.all_img_paths_in_root)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Overwrite Dataset's __getitem__ method to return one data sample (data, label) potentially transformed
        image = self.load_image(index)
        image = self.convert_mode_L_to_RGB(image)
        image_class = self.all_img_paths_in_root[index].parent.name
        class_label = self.class_to_idx[image_class]
        
        image = self.transform(image) if self.transform else image 

        class_label = self.target_transform(class_label) if self.target_transform else class_label
        return image, class_label
        



class LoadOurData():
    
    def __init__(self,
                 train_dir,
                 test_dir,
                 transform,
                 test_transform=None,
                 target_transform=None):
        
        self.train_dir = train_dir 
        self.test_dir = test_dir
        
        self.transform = transform
        # self.test_transform to transform test_data differently than train_data
        self.test_transform = test_transform
        self.target_transform = target_transform
    
        self.train_dataset = None
        self.train_len = None
        self.train_classes = None
        self.train_class_to_idx = None
         
        self.test_dataset = None
        self.test_len = None
        self.test_classes = None
        self.test_class_to_idx = None
    
    def load_data(self, DatasetClass: Dataset):
        self.train_dataset = DatasetClass(root=self.train_dir,
                                       transform=self.transform,
                                       target_transform=self.target_transform)
        
        test_transform = self.test_transform if self.test_transform is not None else self.transform
        self.test_dataset = DatasetClass(root=self.test_dir,
                                      transform=test_transform,
                                      target_transform=self.target_transform)
        
        self.train_len, self.test_len = len(self.train_dataset), len(self.test_dataset)
        self.train_classes, self.test_classes = self.train_dataset.classes, self.test_dataset.classes
        self.train_class_to_idx, self.test_class_to_idx = self.train_dataset.class_to_idx, self.test_dataset.class_to_idx
        return self.train_dataset, self.test_dataset
        
    def load_using_ImageFolderDataset(self):
        return self.load_data(datasets.ImageFolder)
    
    def load_using_OurCustomDataset(self):
        return self.load_data(OurCustomDataset)    
        
    def create_dataloaders(self, BATCH_SIZE=config['DATA_LOADER']['BATCH_SIZE'], train_shuffle=True, test_shuffle=False):
        
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=BATCH_SIZE,
                                           num_workers=os.cpu_count(),
                                           shuffle=train_shuffle)

        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                           batch_size=BATCH_SIZE,
                                           num_workers=os.cpu_count(),
                                           test_shuffle=False)
        
        return self.train_dataloader, self.test_dataloader
    

    
if __name__ == "__main__":
    
    train_dir = os.path.join(project_root_path, 'data', 'train')
    test_dir = os.path.join(project_root_path, 'data', 'test')
    transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])
    
    # Compare our custom dataset loading VS ImageFolder loading
    # Our custom dataset
    instance_our_dataset = LoadOurData(train_dir,
                                       test_dir,
                                       transform)
    instance_our_dataset.load_using_OurCustomDataset()
    
    ### ImageFolder dataset
    #instance_imagefolder = LoadOurData(train_dir,
    #                                   test_dir,
    #                                   transform)  
    #instance_imagefolder.load_using_ImageFolderDataset()
    
    # Print random transformed images
    instance_our_dataset.show_random_images()
    