# Data transformation (into tensor and torch.utils.data.Dataset -> torch.utils.data.DataLoader)
import os
import random
import copy
from collections import defaultdict
from typing import Tuple, Dict, List

import matplotlib.pyplot as plt

import torch 
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler
from torchvision import datasets, transforms

from secondary_module import color_print
from splitted_datasets import SplittedDataset

   
class LoadOurData():
    def __init__(self, data_dir, DatasetClass:Dataset, RANDOM_SEED:int=None, 
                 initial_transform:transforms=None, initial_target_transform:transforms=None, 
                 inference:bool=False):
        
        self.data_dir = data_dir
        self.DatasetClass = DatasetClass

        if isinstance(RANDOM_SEED, int): 
            random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
        
        self.original_dataset = self.get_original_dataset(initial_transform, initial_target_transform)
        
        if not inference:
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
            
            # Mean and std are computed based on training dataset for dataset normalization
            self.mean = None 
            self.std = None
             

    def get_original_dataset(self, transform:transforms=None, target_transform:transforms=None):
        return self.DatasetClass(root=self.data_dir, transform=transform, target_transform=target_transform) if self.data_dir else None
    
    def calculate_normalization(self, batch_size:int=8, resize=(224, 224)):
        if not self.train_dataset: 
            raise ValueError('Normalization should be calculated on training dataset, but there is no self.train_dataset initiated.')
        
        # Compute the mean and standard deviation of the pixel values across all images in your dataset
        
        aug = transforms.Compose([
            transforms.Resize(resize),       # Resize the images to a fixed size
            transforms.ToTensor(),           # Convert the images to PyTorch tensors
        ])
        
        dataset = copy.deepcopy(self.train_dataset)
        list_transformed = self.apply_transformations([(dataset, aug)])
        dataset = list_transformed[0]
        # Define your dataset and DataLoader
        dataloader = DataLoader(dataset,
                                batch_size=batch_size, shuffle=True)

        # Calculate mean and standard deviation
        self.mean, self.std, total_samples = 0, 0, 0

        # Iterate through the DataLoader to calculate mean and std
        for images, _ in dataloader:
            # Get the batch size
            batch_samples = images.size(0)
            # Flatten the images to calculate mean and std across all pixels
            images = images.view(batch_samples, images.size(1), -1)
            # Calculate mean and std across all pixels and channels (second dimension now represents all pixels for each channel)
            self.mean += images.mean(2).sum(0)
            self.std += images.std(2).sum(0)
            # Count the total number of samples
            total_samples += batch_samples

        # Calculate the mean and std across the entire dataset
        self.mean /= total_samples
        self.std /= total_samples
        
        #print('mean =', self.mean, ' ||| ' 'std =', self.std )
        return {'mean':self.mean, 'std':self.std}


    def create_splitted_dataset(self, dataset, transform):
        # This class is for splitted datasets (so that they can carry out the transformations)
        # Enable to apply various transform to different dataset that has been splitted from an initial dataset
        return SplittedDataset(dataset, transform) if dataset else None


    def train_test_split(self, train_ratio:float):
        if not train_ratio >= 0 and not train_ratio <= 1: raise ValueError('train_size must be comprised between 0 and 1.')
        self.train_dataset, self.test_dataset = random_split(self.original_dataset, [train_ratio, 1-train_ratio])
    
    def train_valid_split(self, train_ratio:float):
        if not train_ratio >= 0 and not train_ratio <= 1: raise ValueError('train_size must be comprised between 0 and 1.')
        dataset = self.original_dataset if not self.test_dataset else self.train_dataset
        self.train_dataset, self.valid_dataset = random_split(dataset, [train_ratio, 1-train_ratio])
                  
    def apply_transformations(self, dataset_transform:List[Tuple]):
        #for dataset, transform in dataset_transform:
        #    dataset = self.create_splitted_dataset(dataset, transform)
        return [self.create_splitted_dataset(dataset, transform) for dataset, transform in dataset_transform]

        
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
        
        return {'length':len(dataset), 'count_per_class':count_samples_per_class()} if dataset else None



        
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
        
        
    def create_dataloaders(self, dataset, BATCH_SIZE:int, num_workers:int, pin_memory:bool, sampler=None, shuffle:bool=True,
                           ) -> DataLoader:
        return DataLoader(dataset=dataset,
                          batch_size=BATCH_SIZE,
                          sampler=sampler,
                          num_workers=num_workers,
                          shuffle=shuffle,
                          pin_memory=pin_memory)
         
    def generate_dataloaders(self, BATCH_SIZE:int, dataset_types=['train', 'valid', 'test'], shuffle={'train':True, 'valid':False, 'test':False},
                             num_workers=os.cpu_count(), pin_memory=True):
        for dataset_type in dataset_types:
            # Create dataloader
            dataset=getattr(self, ''.join([dataset_type, '_dataset']))
            if dataset:
                dataloader = self.create_dataloaders(dataset=dataset,
                                                     BATCH_SIZE=BATCH_SIZE,
                                                     num_workers=num_workers,
                                                     pin_memory=pin_memory,
                                                     shuffle=shuffle[dataset_type])

                # Set dataloader as attribute
                setattr(self, ''.join([dataset_type, '_dataloader']), dataloader)




    def generate_cv_datasets(self, kf):
        self.cv = True
        for train_idx, valid_idx in kf.split(self.train_dataset):
            self.cross_valid_datasets['train'].append(torch.utils.data.Subset(self.train_dataset, train_idx))
            self.cross_valid_datasets['valid'].append(torch.utils.data.Subset(self.train_dataset, valid_idx))
            


    def generate_cv_dataloaders(self, BATCH_SIZE:int, num_workers=os.cpu_count(), pin_memory=True):
        
        for train_dataset, valid_dataset in zip(self.cross_valid_datasets['train'], self.cross_valid_datasets['valid']):
            self.cross_valid_dataloaders['train'].append(self.create_dataloaders(
                                                                    dataset=train_dataset,
                                                                    BATCH_SIZE=BATCH_SIZE,
                                                                    num_workers=num_workers,
                                                                    pin_memory=pin_memory,
                                                                    shuffle=True)
                                                    )

            self.cross_valid_dataloaders['valid'].append(self.create_dataloaders(
                                                                    dataset=valid_dataset,
                                                                    BATCH_SIZE=BATCH_SIZE,
                                                                    num_workers=num_workers,
                                                                    pin_memory=pin_memory,
                                                                    shuffle=True)
                                                    )
                                                                
                              
                              
    
    def _get_random_images_dataloader(self, dataloader:DataLoader, n:int):
        # Get the length of the DataLoader (number of samples) and define the indices of the dataset
        indices = list(range(len(dataloader.dataset)))
        # Shuffle the indices
        random.shuffle(indices)
        # Select 6 random indices
        random_indices = indices[:n]
        # Create and return a new DataLoader with the SubsetRandomSampler
        return DataLoader(
            dataset=dataloader.dataset,
            batch_size=1,
            sampler=SubsetRandomSampler(random_indices) # sampler is a SubsetRandomSampler using the selected indices
        ) 

         
    def inverse_normalize_img(self, tensor, mean, std):
        # Ensures mean and std compatibility with image tensors three dimensions (channels, height, and width)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        return tensor * std + mean


    def show_random_images(self,
                           dataloader:DataLoader,
                           n:int = 6,
                           display_seconds:int= 30,
                           unnormalize:bool=False
                           ):
        # Get random images (in the form of a dataloader)
        random_dataloader = self._get_random_images_dataloader(dataloader, n)
        
        # Initiate plot and start interactive mode (for non blocking plot)
        plt.figure(figsize=(20, 5))
        plt.ion()

        # Loop over indexes and plot corresponding image
        for i, (image, label) in enumerate(random_dataloader):
            # Remove the batch dimension (which is 1)
            image = image.squeeze(0)
            if unnormalize:
                # Unnormalize image
                image = self.inverse_normalize_img(image, self.mean, self.std)
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
 
 
 
 
 
 
 
 
