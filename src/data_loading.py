# Data transformation (into tensor and torch.utils.data.Dataset -> torch.utils.data.DataLoader)
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List
from PIL import Image 
import matplotlib.pyplot as plt

import torch 
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from secondary_module import color_print

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
    def __init__(self, data_dir, DatasetClass:Dataset):
        
        self.data_dir = data_dir
        self.DatasetClass = DatasetClass

        self.dataset_types = ['train', 'valid', 'test']
        
        '''
        After running load_data method we will have the 
        following attributes for each dataset_type:

        self.{DATASET_TYPE}_dataset
        self.{DATASET_TYPE}_dataloader
        self.{DATASET_TYPE}_len
        self.{DATASET_TYPE}_classes
        self.{DATASET_TYPE}_class_to_idx
        '''


    def count_samples_per_class(self, dataset: Dataset, dataset_type: str):     
        # Initialize a defaultdict to count samples per class
        if dataset_type.lower() not in self.dataset_types: 
            raise ValueError('dataset_type should be "train", "valid" or "test".') 
        classes = getattr(self, ''.join([dataset_type, '_classes']))
        
        samples_per_class = defaultdict(int)
        # Iterate over all samples and count occurrences of each class  
        for _, label in dataset:
            img_class = classes[label]
            samples_per_class[img_class] += 1
            
        return samples_per_class

        
    def load_data(self,                  
                  transform:transforms.Compose,
                  test_transform:transforms.Compose = None,
                  target_transform:transforms.Compose = None,
                  train_ratio:int = 0.8,
                  valid_ratio:int = 0.1,
                  test_ratio:int = 0.1
                  ):
        
        if sum([train_ratio, valid_ratio, test_ratio]) != 1:
            raise ValueError("Sum of train_ratio, valid_ratio and test_ratio must equal 1.")
        
        # Load all our data without any transform
        original_dataset = self.DatasetClass(root=self.data_dir, transform=None, target_transform=target_transform)
        # Define the sizes of training, validation, and test sets
        train_size = int(train_ratio * len(original_dataset))
        valid_size = int(valid_ratio * len(original_dataset))
        # Set all remaining images to test dataset (to avoid leaving one image unaccounted for)
        test_size = len(original_dataset) - train_size - valid_size
        
        # Split the original dataset into training, validation, and test sets
        self.train_dataset, self.valid_dataset, self.test_dataset = random_split(original_dataset, [train_size, valid_size, test_size])

        # Apply the corresponding transformations to each dataset
        self.train_dataset.dataset.transform = transform
        self.valid_dataset.dataset.transform = transform
        self.test_dataset.dataset.transform = test_transform if test_transform is not None else transform

        for dataset_type in self.dataset_types:
            # Access dataset
            dataset = getattr(self, ''.join([dataset_type, '_dataset']))
            # Calculate dataset's length
            setattr(self, ''.join([dataset_type, '_len']), len(dataset)) 
            # Get its classes (same as original dataset)
            setattr(self, ''.join([dataset_type, '_classes']), original_dataset.classes) 
            # Get its class_to_idx (same as original dataset)
            setattr(self, ''.join([dataset_type, '_class_to_idx']), original_dataset.class_to_idx)
            # Get count per class
            setattr(self, ''.join([dataset_type, '_count_per_class']), self.count_samples_per_class(dataset, dataset_type))
            
        return self.train_dataset, self.valid_dataset, self.test_dataset
     
        
    def print_info_on_loaded_data(self):
        print(
            color_print("---------- DATASETS INFO ----------", "LIGHTGREEN_EX")
        )
        
        for dataset_type in self.dataset_types:
            print(
                color_print(f"Info regarding {dataset_type}_dataset:", "RED"),
                color_print("\nLength: ", "BLUE"), getattr(self, ''.join([dataset_type, '_len'])),       
                color_print("\nClasses/labels: ", "BLUE"), getattr(self, ''.join([dataset_type, '_class_to_idx'])), 
                color_print("\nImages per class: ", "BLUE"), getattr(self, ''.join([dataset_type, '_count_per_class'])), '\n'     
            )
  
        
        
    def create_dataloaders(self, BATCH_SIZE:int, train_shuffle:bool=True, valid_shuffle:bool=True, test_shuffle:bool=False):
        shuffle = {"train":train_shuffle, "valid":valid_shuffle, "test":test_shuffle}
        for dataset_type in self.dataset_types:
            data_loader = DataLoader(dataset=getattr(self, ''.join([dataset_type, '_dataset'])),
                                     batch_size=BATCH_SIZE,
                                     num_workers=os.cpu_count(),
                                     shuffle=shuffle[dataset_type])
            setattr(self, ''.join([dataset_type, '_dataloader']), data_loader) 
            
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader
    
    
    def show_random_images(self,
                           RANDOM_SEED:int = None,
                           dataset_type:str = 'train',
                           n:int = 6,
                           display_seconds:int= 30
                           ):
        
        if isinstance(RANDOM_SEED, int): 
            random.seed(RANDOM_SEED)

        dataset = getattr(self, ''.join([dataset_type.lower(), '_dataset']))
        classes = getattr(self, ''.join([dataset_type.lower(), '_classes']))
        # Get random indexes in the range 0 - length dataset
        random_idxs = random.sample(range(len(dataset)), k=n)
        
        # Initiate plot and start interactive mode (for non blocking plot)
        plt.figure(figsize=(20, 5))
        plt.ion()
          
        # Loop over indexes and plot corresponding image
        for i, random_index in enumerate(random_idxs):
            image, label = dataset[random_index]
            # Adjust tensor's dimensions for plotting : Color, Height, Width -> Height, Width, Color
            image = image.permute(1, 2, 0)
            # Set up subplot (number rows in subplot, number cols in subplot, index of subplot)
            plt.subplot(1, n, i+1)
            plt.imshow(image)
            plt.axis(False)
            plt.title(f"Class: {classes[label]}\n Shape: {image.shape}")
        # Show the plot with tight layout for some time and then close the plot and deactivate interactive mode
        plt.tight_layout()
        plt.draw() 
        plt.pause(display_seconds)
        plt.ioff()
        plt.close()
        return