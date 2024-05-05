# Data transformation (into tensor and torch.utils.data.Dataset -> torch.utils.data.DataLoader)
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Dict, List
from PIL import Image 
import matplotlib.pyplot as plt

import torch 
from torch.utils.data import DataLoader, Dataset, Subset, random_split, SubsetRandomSampler
from torchvision import datasets, transforms

from secondary_module import color_print

class CustomImageFolder(Dataset):
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
    def __init__(self, data_dir, DatasetClass:Dataset, random_seed=None):
        
        self.data_dir = data_dir
        self.DatasetClass = DatasetClass
        
        self.original_dataset = self.get_original_dataset()
        self.classes = self.original_dataset.classes
        self.class_to_idx = self.original_dataset.class_to_idx

        self.train_dataset=None
        self.valid_dataset=None
        self.test_dataset=None
        
        self.train_dataloader=None
        self.valid_dataloader=None
        self.test_dataloader=None
        
        self.datasets_metadata = {'train':None,
                                  'valid':None,
                                  'test' :None}
 
        self.cv = False
        self.cross_valid_datasets = {'train': [], 'valid': []}       
        self.cross_valid_dataloaders = {'train': [], 'valid': []}
        self.cross_valid_datasets_metadata = {'train': {}, 'valid': {}}
             

    def get_original_dataset(self, transform:transforms=None, target_transform:transforms=None):
        return self.DatasetClass(root=self.data_dir, transform=transform, target_transform=target_transform)


    def train_test_split(self, train_size:float):
        if not train_size >= 0 and not train_size <= 1: raise ValueError('train_size must be comprised between 0 and 1.')
        self.train_dataset, self.test_dataset = random_split(self.original_dataset, [train_size, 1-train_size])
    
    def train_valid_split(self, train_size:float):
        if not train_size >= 0 and not train_size <= 1: raise ValueError('train_size must be comprised between 0 and 1.')
        self.train_dataset, self.valid_dataset = random_split(self.train_dataset, [train_size, 1-train_size])
                  
    def apply_transformations(self, dataset_transform:List[Tuple]):
        for dataset, transform in dataset_transform:
           # Apply the corresponding transformations to each dataset
           dataset.transform = transform

    def get_dataset_metadata(self, dataset:Dataset):
        def count_samples_per_class():     
            # Initialize a defaultdict to count samples per class
            classes = self.original_dataset.classes
            samples_per_class = defaultdict(int)
            # Iterate over all samples and count occurrences of each class  
            for _, label in dataset:
                img_class = classes[label]
                samples_per_class[img_class] += 1
            return samples_per_class
        
        return {'length':len(dataset), 'count_per_class':count_samples_per_class()}



        
    def print_dataset_info(self, datasets_types:List[str]=['train', 'valid', 'test'], n_splits=None,
                                 dataset_color = {'train':'LIGHTRED_EX', 'valid':'LIGHTYELLOW_EX', 'test':'LIGHTMAGENTA_EX'}):
        '''
        If kf is not assigned: Print metadata of train, valid and test datasets (no cv)
        Else: print metadata of train and valid dataseta for each cross_validation fold and finally that of test dataset
        '''
        print(color_print("---------- DATASETS INFO ----------", "LIGHTGREEN_EX"))
        
        print(color_print("\nAll classes/labels: ", "BLUE"), self.class_to_idx, '\n')
    
        for dataset_type in datasets_types:
            if self.datasets_metadata.get(dataset_type) is not None:
                print(
                    color_print(f"Info regarding {dataset_type}_dataset:", dataset_color[dataset_type]),
                    color_print("\nLength: ", "LIGHTBLUE_EX"), self.datasets_metadata[dataset_type]['length'],       
                    color_print("\nImages per class: ", "LIGHTBLUE_EX"), self.datasets_metadata[dataset_type]['count_per_class'], '\n'     
                )        
        
        if n_splits:
            for i in range(n_splits):
                # Print info for train and valid datasets for each cross-validation fold
                for dataset_type in self.cross_valid_datasets:
                    print(
                        color_print(f"Info regarding {dataset_type}_dataset, fold -- {i} -- of cross-validation:", dataset_color[dataset_type]),
                        color_print("\nLength: ", "LIGHTBLUE_EX"), self.cross_valid_datasets_metadata[dataset_type][i]['length'],       
                        color_print("\nImages per class: ", "LIGHTBLUE_EX"), self.cross_valid_datasets_metadata[dataset_type][i]['count_per_class'], '\n'     
                    )  
        
        
    def _create_dataloaders(self, dataset, BATCH_SIZE:int, sampler=None, shuffle:bool=True) -> DataLoader:
        return DataLoader(dataset=dataset,
                          batch_size=BATCH_SIZE,
                          sampler=sampler,
                          num_workers=os.cpu_count(),
                          shuffle=shuffle)
         
    def generate_dataloaders(self, BATCH_SIZE:int, dataset_types=['train', 'valid', 'test']):
        for dataset_type in dataset_types:
            # Create dataloader
            dataloader = self._create_dataloaders(dataset=getattr(self, ''.join([dataset_type, '_dataset'])),
                                                  BATCH_SIZE=BATCH_SIZE,
                                                  shuffle=True)
            # Set dataloader as attribute
            setattr(self, ''.join([dataset_type, '_dataloader']), dataloader)




    def generate_cv_datasets(self, kf):
        self.cv = True
        for train_idx, valid_idx in kf.split(self.train_dataset):
            self.cross_valid_datasets['train'].append(torch.utils.data.Subset(self.train_dataset, train_idx))
            self.cross_valid_datasets['valid'].append(torch.utils.data.Subset(self.train_dataset, valid_idx))
            


    def generate_cv_dataloaders(self, BATCH_SIZE:int):
        
        for train_dataset, valid_dataset in zip(self.cross_valid_datasets['train'], self.cross_valid_datasets['valid']):
            self.cross_valid_dataloaders['train'].append(self._create_dataloaders(
                                                                    dataset=train_dataset,
                                                                    BATCH_SIZE=BATCH_SIZE,
                                                                    shuffle=True)
                                                    )

            self.cross_valid_dataloaders['valid'].append(self._create_dataloaders(
                                                                    dataset=valid_dataset,
                                                                    BATCH_SIZE=BATCH_SIZE,
                                                                    shuffle=True)
                                                    )
                                                                
                              

    
    
    
    
    
    
    
    
    
    
    
    def show_random_images(self,
                           RANDOM_SEED:int = None,
                           dataset_type:str = 'train',
                           n:int = 6,
                           display_seconds:int= 30
                           ):
        
        if isinstance(RANDOM_SEED, int): 
            random.seed(RANDOM_SEED)

        # Get dataset or first dataset if cv as well as classes
        dataloader = getattr(self, ''.join([dataset_type.lower(), '_dataloader'])) if not self.cv else self.cross_valid_dataloaders[dataset_type][0]
        # Combine all batches into one large tensor
        all_images_labels = torch.cat([batch for batch in dataloader], dim=0)
        # Select n random indices
        random_indices = torch.randperm(len(all_images_labels))[:n]
        # Use the selected indices to extract the random images
        random_images_labels = all_images_labels[random_indices]
        
        # Initiate plot and start interactive mode (for non blocking plot)
        plt.figure(figsize=(20, 5))
        plt.ion()
          
        # Loop over indexes and plot corresponding image
        for i, (image, label) in enumerate(random_images_labels):
            # Adjust tensor's dimensions for plotting : Color, Height, Width -> Height, Width, Color
            image = image.permute(1, 2, 0)
            # Set up subplot (number rows in subplot, number cols in subplot, index of subplot)
            plt.subplot(1, n, i+1)
            plt.imshow(image)
            plt.axis(False)
            plt.title(f"Class: {self.classes[label]}\n Shape: {image.shape}")
        # Show the plot with tight layout for some time and then close the plot and deactivate interactive mode
        plt.tight_layout()
        plt.draw() 
        plt.pause(display_seconds)
        plt.ioff()
        plt.close()
        return
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
# 
#    
#class LoadOurData():
#    def __init__(self, data_dir, DatasetClass:Dataset):
#        
#        self.data_dir = data_dir
#        self.DatasetClass = DatasetClass
#
#        self.dataset_types = ['train', 'valid', 'test']
#        
#        
#        '''
#        After running load_data method we will have the 
#        following attributes for each dataset_type:
#
#        self.{DATASET_TYPE}_dataset
#        self.{DATASET_TYPE}_dataloader
#        self.{DATASET_TYPE}_len
#        self.{DATASET_TYPE}_classes
#        self.{DATASET_TYPE}_class_to_idx
#        '''
#
#
#    def count_samples_per_class(self, dataset: Dataset, dataset_type: str):     
#        # Initialize a defaultdict to count samples per class
#        if dataset_type.lower() not in self.dataset_types: 
#            raise ValueError('dataset_type should be "train", "valid" or "test".') 
#        classes = getattr(self, ''.join([dataset_type, '_classes']))
#        
#        samples_per_class = defaultdict(int)
#        # Iterate over all samples and count occurrences of each class  
#        for _, label in dataset:
#            img_class = classes[label]
#            samples_per_class[img_class] += 1
#            
#        return samples_per_class
#
#        
#    def load_data(self,                  
#                  transform:transforms.Compose,
#                  test_transform:transforms.Compose = None,
#                  target_transform:transforms.Compose = None,
#                  train_ratio:int = 0.8,
#                  valid_ratio:int = 0.1,
#                  test_ratio:int = 0.1,
#                  cross_valid_kf=None
#                  ):
#        
#        if sum([train_ratio, valid_ratio, test_ratio]) != 1:
#            raise ValueError("Sum of train_ratio, valid_ratio and test_ratio must equal 1.")
#        
#        # Load all our data without any transform
#        original_dataset = self.DatasetClass(root=self.data_dir, transform=None, target_transform=target_transform)
#        
#        if not cross_valid_kf:
#            # Define the sizes of training, validation, and test sets
#            train_size = int(train_ratio * len(original_dataset))
#            valid_size = int(valid_ratio * len(original_dataset))
#            # Set all remaining images to test dataset (to avoid leaving one image unaccounted for)
#            test_size = len(original_dataset) - train_size - valid_size
#            # Split the original dataset into training, validation, and test sets
#            self.train_dataset, self.valid_dataset, self.test_dataset = random_split(original_dataset, [train_size, valid_size, test_size])
#        
#        #else:
#            # MAYBE SAVE ALL THE SPLIUTTED DATASETS SO THAT WE CAN ITERATE OVER THEM LATER ON
#
#        # Apply the corresponding transformations to each dataset
#        self.train_dataset.dataset.transform = transform
#        self.valid_dataset.dataset.transform = transform
#        self.test_dataset.dataset.transform = test_transform if test_transform is not None else transform
#
#        for dataset_type in self.dataset_types:
#            # Access dataset
#            dataset = getattr(self, ''.join([dataset_type, '_dataset']))
#            # Calculate dataset's length
#            setattr(self, ''.join([dataset_type, '_len']), len(dataset)) 
#            # Get its classes (same as original dataset)
#            setattr(self, ''.join([dataset_type, '_classes']), original_dataset.classes) 
#            # Get its class_to_idx (same as original dataset)
#            setattr(self, ''.join([dataset_type, '_class_to_idx']), original_dataset.class_to_idx)
#            # Get count per class
#            setattr(self, ''.join([dataset_type, '_count_per_class']), self.count_samples_per_class(dataset, dataset_type))
#            
#        return self.train_dataset, self.valid_dataset, self.test_dataset
#     
#        
#    def print_info_on_loaded_data(self):
#        print(
#            color_print("---------- DATASETS INFO ----------", "LIGHTGREEN_EX")
#        )
#        
#        for dataset_type in self.dataset_types:
#            print(
#                color_print(f"Info regarding {dataset_type}_dataset:", "RED"),
#                color_print("\nLength: ", "BLUE"), getattr(self, ''.join([dataset_type, '_len'])),       
#                color_print("\nClasses/labels: ", "BLUE"), getattr(self, ''.join([dataset_type, '_class_to_idx'])), 
#                color_print("\nImages per class: ", "BLUE"), getattr(self, ''.join([dataset_type, '_count_per_class'])), '\n'     
#            )
#  
#        
#        
#    def create_dataloaders(self, BATCH_SIZE:int, train_shuffle:bool=True, valid_shuffle:bool=True, test_shuffle:bool=False):
#        shuffle = {"train":train_shuffle, "valid":valid_shuffle, "test":test_shuffle}
#        for dataset_type in self.dataset_types:
#            data_loader = DataLoader(dataset=getattr(self, ''.join([dataset_type, '_dataset'])),
#                                     batch_size=BATCH_SIZE,
#                                     num_workers=os.cpu_count(),
#                                     shuffle=shuffle[dataset_type])
#            setattr(self, ''.join([dataset_type, '_dataloader']), data_loader) 
#            
#        return self.train_dataloader, self.valid_dataloader, self.test_dataloader
#    
#    
#    def show_random_images(self,
#                           RANDOM_SEED:int = None,
#                           dataset_type:str = 'train',
#                           n:int = 6,
#                           display_seconds:int= 30
#                           ):
#        
#        if isinstance(RANDOM_SEED, int): 
#            random.seed(RANDOM_SEED)
#
#        dataset = getattr(self, ''.join([dataset_type.lower(), '_dataset']))
#        classes = getattr(self, ''.join([dataset_type.lower(), '_classes']))
#        # Get random indexes in the range 0 - length dataset
#        random_idxs = random.sample(range(len(dataset)), k=n)
#        
#        # Initiate plot and start interactive mode (for non blocking plot)
#        plt.figure(figsize=(20, 5))
#        plt.ion()
#          
#        # Loop over indexes and plot corresponding image
#        for i, random_index in enumerate(random_idxs):
#            image, label = dataset[random_index]
#            # Adjust tensor's dimensions for plotting : Color, Height, Width -> Height, Width, Color
#            image = image.permute(1, 2, 0)
#            # Set up subplot (number rows in subplot, number cols in subplot, index of subplot)
#            plt.subplot(1, n, i+1)
#            plt.imshow(image)
#            plt.axis(False)
#            plt.title(f"Class: {classes[label]}\n Shape: {image.shape}")
#        # Show the plot with tight layout for some time and then close the plot and deactivate interactive mode
#        plt.tight_layout()
#        plt.draw() 
#        plt.pause(display_seconds)
#        plt.ioff()
#        plt.close()
#        return