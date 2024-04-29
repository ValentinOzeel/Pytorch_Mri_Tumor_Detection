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

from colorama import init, Fore, Back, Style
init() # Initialize Colorama to work on Windows


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
        

class ConfigLoad():
    def __init__(self, config):
        self.config = config
        
    def get_transform(self, dict_name='DATA_TRANSFORM_AND_AUGMENTATION') -> List:
        '''
        Access transformation dict defined in config
        Transform it as a list of torchvision.transforms steps
        '''
        yml_dict = self.config[dict_name]
        steps = []
        for step_name, params in yml_dict.items():
            # Get the transforms method
            transform_step = getattr(transforms, step_name)
            # Initialize the transform method with its defined parameters and append in list
            if params: 
                steps.append(transform_step(**params))
            else:
                steps.append(transform_step()) 
        return steps

    
    
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

    def count_per_class_train_and_test(self):
        def count_samples_per_class(dataset: Dataset, train_or_test: str):        
            # Initialize a defaultdict to count samples per class
            if train_or_test not in ['train', 'test']: raise ValueError('train_or_test parameter should be "train" or "test".') 
            classes = self.train_classes if train_or_test.lower() == 'train' else self.test_classes

            samples_per_class = defaultdict(int)
            # Iterate over all samples and count occurrences of each class  
            for _, label in dataset:
                img_class = classes[label]
                samples_per_class[img_class] += 1
            return samples_per_class
        
        return count_samples_per_class(self.train_dataset, 'train'), count_samples_per_class(self.test_dataset, 'test')
        
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
        self.train_count_per_class, self.test_count_per_class = self.count_per_class_train_and_test()
        return self.train_dataset, self.test_dataset
        
    def load_using_ImageFolderDataset(self):
        return self.load_data(datasets.ImageFolder)
    
    def load_using_OurCustomDataset(self):
        return self.load_data(OurCustomDataset)
    
    def color_print(self, to_print, color):
        return f"{color + to_print + Style.RESET_ALL}"
        
    def print_info_on_loaded_data(self):
        print(
            self.color_print("---------- DATA INFO ----------", Fore.LIGHTGREEN_EX)
        )
        print(
            self.color_print("OurCustomDataset (TRAIN dataset):", Fore.RED),
            self.color_print("\nLength: ", Fore.BLUE), self.train_len,       
            self.color_print("\nClasses/labels: ", Fore.BLUE), self.train_class_to_idx,   
            self.color_print("\nImages per class: ", Fore.BLUE), self.train_count_per_class, '\n'
            )
        print(
            self.color_print("OurCustomDataset (TEST dataset):", Fore.RED),
            self.color_print("\nLength: ", Fore.BLUE), self.test_len,       
            self.color_print("\nClasses/labels: ", Fore.BLUE), self.test_class_to_idx,   
            self.color_print("\nImages per class: ", Fore.BLUE), self.test_count_per_class, '\n\n'
            )    
        
        
    def create_dataloaders(self, BATCH_SIZE=config['DATA_LOADER']['BATCH_SIZE'], train_shuffle=True, test_shuffle=False):
        
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=BATCH_SIZE,
                                           num_workers=os.cpu_count(),
                                           shuffle=train_shuffle)

        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                           batch_size=BATCH_SIZE,
                                           num_workers=os.cpu_count(),
                                           shuffle=test_shuffle)
        
        return self.train_dataloader, self.test_dataloader
    
    
    def show_random_images(self,
                           str_dataset: str = 'train',
                           n: int = 6,
                           RANDOM_SEED = config['RANDOM_SEED']):
        
        if RANDOM_SEED: random.seed(RANDOM_SEED)
        dataset = self.train_dataset if str_dataset.lower() == 'train' else self.test_dataset
        # Get random indexes in the range 0 - length dataset
        random_idxs = random.sample(range(len(dataset)), k=n)
            
        # Initiate plot
        plt.figure(figsize=(20, 10))
        
        # Loop over indexes and plot corresponding image
        for i, random_index in enumerate(random_idxs):
            image, label = dataset[random_index]
            
            # Adjust tensor's dimensions for plotting : Color, Height, Width -> Height, Width, Color
            image = image.permute(1, 2, 0)
            # Set up subplot (number rows in subplot, number cols in subplot, index of subplot)
            plt.subplot(1, n, i+1)
            plt.imshow(image)
            plt.axis(False)
            plt.title(f"Class: {dataset.classes[label]}\n Shape: {image.shape}")

        plt.tight_layout()
        plt.show()
        return
            
    
if __name__ == "__main__":
    
    train_dir = os.path.join(project_root_path, 'data', 'train')
    test_dir = os.path.join(project_root_path, 'data', 'test')
    
    conf_load = ConfigLoad(config)
    transform_steps = conf_load.get_transform()
    transform = transforms.Compose(transform_steps)
    
    # Compare our custom dataset loading VS ImageFolder loading
    # Our custom dataset
    instance_our_dataset = LoadOurData(train_dir,
                                       test_dir,
                                       transform)
    instance_our_dataset.load_using_OurCustomDataset()
    instance_our_dataset.print_info_on_loaded_data()
    
    ### ImageFolder dataset
    #instance_imagefolder = LoadOurData(train_dir,
    #                                   test_dir,
    #                                   transform)  
    #instance_imagefolder.load_using_ImageFolderDataset()
    #instance_our_dataset.print_info_on_loaded_data()
    
    # Create DataLoaders to load images per in batches
    instance_our_dataset.create_dataloaders()
    # Get one iteration of train_dataloader (loading in batches)
    img, label = next(iter(instance_our_dataset.train_dataloader))
    print('Dataloader batches:', 'Image shapes', img.shape, 'label shapes', label.shape)
    
    # Print random transformed images
    instance_our_dataset.show_random_images()
    