# Data transformation (into tensor and torch.utils.data.Dataset -> torch.utils.data.DataLoader)
import random
import copy
from typing import Tuple, Dict, List, Optional

import matplotlib.pyplot as plt

import torch 
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import datasets, transforms

from datasets import DataPrep
from secondary_module import colorize

   
class LoadOurData():
    def __init__(self, data_dir_path:str, DatasetClass:Dataset, 
                 test_data_dir_path=None, random_seed:int=None, inference:bool=False):
        '''
        Assign data_dir so that to get all the files from this path and create train/val as well as potentially test datasets 
        If you have another folder dedicated for your test_data, add the path to test_data_dir_path kwarg
        Otherwise, activate inference Flag to tell that your data_dir is for your inference data
        '''
        
        self.data_prep = DataPrep(root=data_dir_path, random_seed=random_seed)
        self.original_df = self.data_prep.create_path_class_df()
        
        self.DatasetClass = DatasetClass
        
        self.random_seed = random_seed
        if isinstance(random_seed, int): 
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        if not inference:
            self.test_data_dir_path = test_data_dir_path
            self.classes = self.DatasetClass(self.original_df).classes
            self.class_to_idx = self.DatasetClass(self.original_df).class_to_idx

            self.train_dataset=None
            self.val_dataset=None
            self.test_dataset=None

            self.train_dataloader=None
            self.val_dataloader=None
            self.test_dataloader=None

            self.datasets_metadata = {'train':None,
                                      'val':None,
                                      'test' :None}
    
            self.cross_val_datasets = {'train': [], 'val': []}       
            self.cross_val_dataloaders = {'train': [], 'val': []}
            self.cross_val_datasets_metadata = {'train': {}, 'val': {}}
            
            # Mean and std are computed based on training dataset for dataset normalization
            self.mean = None 
            self.std = None
            
            
    def _get_created_dataset_types(self, class_obj, attr_suffix:str, types:List = ['train', 'val', 'test']) -> List:
        # Get types (according to performed presplits/splits in self.data_prep)
        return [dataset_type for dataset_type in types if hasattr(class_obj, ''.join([dataset_type, attr_suffix]))]

    def _get_corresponding_transforms(self, dataset_type: str, dict_transform_steps: Dict, dict_target_transform_steps: Optional[Dict] = None):
        # Get transform and potential target_transform associated to dataset_type
        return dict_transform_steps.get(dataset_type), dict_target_transform_steps.get(dataset_type) if dict_target_transform_steps else None
    
    
    def train_test_presplit(self, train_ratio:float):
        # Split the data (data represents a df (cols = ['path', 'label']) in datasets.DataPrep) into train and valid dfs
        self.data_prep.train_test_presplit(train_ratio)

    def calculate_normalization(self, list_train_transforms, batch_size:int=8):
        # Get mean and standard deviation of the pixel values across all images in our train dataset
        
        if not hasattr(self.data_prep, 'train_df'): 
            raise ValueError('Normalization should be calculated on training data, but there is no self.data_prep.train_df initiated.')
        
        # Remove the normalize flag if it exists
        list_train_transforms = [transform for transform in list_train_transforms if transform != 'Normalize']
        # Create transform compose
        train_transforms = transforms.Compose(list_train_transforms)
        # Compute the mean and standard deviation of the pixel values across all images in your training dataset
        train_df = copy.deepcopy(self.data_prep.train_df)
        # Define your dataset and DataLoader
        dataset = self.DatasetClass(train_df, transform=train_transforms)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
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


    def _create_composes(self, dict_transform_steps, normalize:Dict=None):
        
        for data_type in dict_transform_steps.keys():
            # Replace Normalize flag by actual normalization
            if 'Normalize' in dict_transform_steps[data_type]:
                normalize_flag_index = dict_transform_steps[data_type].index('Normalize')
                dict_transform_steps[data_type][normalize_flag_index] = getattr(transforms, 'Normalize')(**normalize)
            # Create the Compose
            dict_transform_steps[data_type] = transforms.Compose(dict_transform_steps[data_type])
        return dict_transform_steps

        
    def generate_datasets(self, train_ratio:float, dict_transform_steps:Dict, dict_target_transform_steps:Dict=None):
        # Carry out split (and create corresponding df)
        self.data_prep.train_test_presplit(train_ratio)
        # Get normalization if needed
        normalization = self.calculate_normalization(dict_transform_steps['train']) if 'Normalize' in dict_transform_steps['train'] else None
        # Create composes
        dict_transform = self._create_composes(dict_transform_steps, normalize=normalization)
        # Get types (according to performed presplits/splits in self.data_prep)
        dataset_types = self._get_created_dataset_types(self.data_prep, attr_suffix='_df')
        
        for dataset_type in dataset_types:
            # Get the corresponding path_class dataframe
            df = getattr(self.data_prep, ''.join([dataset_type, '_df']))
            # Get the corresponding transform and target_transform
            transform, target_transform = self._get_corresponding_transforms(dataset_type, dict_transform, dict_target_transform_steps=dict_target_transform_steps)
            # Set self."dataset_type"_dataset attribute = self.DatasetClass objet
            setattr(self, ''.join([dataset_type, '_dataset']), self.DatasetClass(df, transform=transform, target_transform=target_transform))
            
        # If test_data in another folder
        if self.test_data_dir_path:
            data_prep = DataPrep(root=self.test_data_dir_path, random_seed=self.random_seed)
            df = data_prep.create_path_class_df()
            transform, target_transform = self._get_corresponding_transforms('test', dict_transform, dict_target_transform_steps=dict_target_transform_steps)
            self.test_dataset = self.DatasetClass(df, transform=transform, target_transform=target_transform)
            
    def generate_cv_datasets(self, dict_transform_steps:Dict, dict_target_transform_steps:Dict=None, 
                             n_splits:int=5, shuffle:bool=True, kf=None):
        # Get the splitted data in a dict (key = fold, value = dict{train:df, valid:df})
        cv_dfs = self.data_prep.cv_splits(n_splits=n_splits, shuffle=shuffle, kf=kf)
        self.cv_n_splits = kf.get_n_splits() if kf else n_splits
        # Get normalization if needed
        normalization = self.calculate_normalization(dict_transform_steps['train']) if 'Normalize' in dict_transform_steps['train'].keys() else None
        # Create composes
        dict_transform = self._create_composes(dict_transform_steps, normalize=normalization)
        
        for fold in range(self.cv_n_splits):
            for dataset_type in ['train', 'val']:
                # Get the corresponding path_class dataframe
                df = cv_dfs[fold][dataset_type]
                # Get the corresponding transform and target_transform
                transform, target_transform = self._get_corresponding_transforms(dataset_type, dict_transform, dict_target_transform_steps=dict_target_transform_steps)
                # Set attributes
                self.cross_val_datasets[dataset_type].append(self.DatasetClass(df, transform=transform, target_transform=target_transform))
                
    def create_dataloader(self, dataset:datasets, shuffle:bool, data_loader_params:Dict) -> DataLoader:
        return DataLoader(dataset=dataset,
                          shuffle=shuffle,
                          **data_loader_params)
        
    def generate_dataloaders(self,  data_loader_params:Dict, shuffle={'train':True, 'val':False, 'test':False}):
        # Get types (according to performed presplits/splits in self.data_prep)
        dataset_types = self._get_created_dataset_types(self, attr_suffix='_dataset')
 
        for dataset_type in dataset_types:
           # Create dataloader
           dataloader = self.create_dataloader(dataset=getattr(self, ''.join([dataset_type, '_dataset'])),
                                                shuffle=shuffle[dataset_type],
                                                data_loader_params=data_loader_params
                                                )
           # Set dataloader as attribute
           setattr(self, ''.join([dataset_type, '_dataloader']), dataloader)       
    
        
        

    def generate_cv_dataloaders(self, data_loader_params:Dict):
        
        for train_dataset, val_dataset in zip(self.cross_val_datasets['train'], self.cross_val_datasets['val']):
            self.cross_val_dataloaders['train'].append(self.create_dataloader(
                                                                    dataset=train_dataset,
                                                                    shuffle=True,
                                                                    data_loader_params=data_loader_params)
                                                    )

            self.cross_val_dataloaders['val'].append(self.create_dataloader(
                                                                    dataset=val_dataset,
                                                                    shuffle=True,
                                                                    data_loader_params=data_loader_params)
                                                    )
                                                                




        ISSSUE OVER DATASET LEN I GUESS (NEED TRO KNOW WHY WITH DATALOADER THERE IS NO INDEXING ISSUE)
        
    def get_dataset_metadata(self, dataset:Dataset):
        def count_samples_per_class():     
            # Initialize a defaultdict to count samples per class
            samples_per_class = copy.deepcopy(self.class_to_idx)
            samples_per_class = {key:0 for key in sorted(self.class_to_idx.keys())}
            # Iterate over all samples and count occurrences of each class  
            for index in range(len(dataset)):
                _, label = dataset.__getitem__(index)
           # for _, label in dataset:
                img_class = self.classes[label]
                samples_per_class[img_class] += 1
            return samples_per_class
        
        return {'length':len(dataset), 'count_per_class':count_samples_per_class()} if dataset else None

    def store_datasets_metadata(self, cv=None):

        if not cv:
            # Get types (according to performed presplits/splits in self.data_prep)
            dataset_types = self._get_created_dataset_types(self, attr_suffix='_dataset')
            # Store datasets' metadata (len, count_per_class)
            for dataset_type in dataset_types:
                dataset = getattr(self, ''.join([dataset_type, '_dataset']))
                self.datasets_metadata[dataset_type] = self.get_dataset_metadata(dataset)
        
        else: 
            for dataset_type in self.cross_val_datasets.keys():
                for fold, dataset in enumerate(self.cross_val_datasets[dataset_type]):
                    self.cross_val_datasets_metadata[dataset_type][fold] = self.get_dataset_metadata(dataset)
        


        
    def print_dataset_info(self, datasets_types:List[str]=['train', 'val', 'test'], n_splits=None,
                                 dataset_color = {'train':'LIGHTRED_EX', 'val':'LIGHTYELLOW_EX', 'test':'LIGHTMAGENTA_EX'}):
        '''
        If kf is not assigned: Print metadata of train, val and test datasets (no cv)
        Else: print metadata of train and val dataset for each cross_validation fold and finally that of test dataset
        '''
        print(colorize("---------- DATASETS INFO ----------", "LIGHTGREEN_EX"))
        
        print(colorize("\nAll classes/labels: ", "BLUE"), self.class_to_idx, '\n')
    
        for dataset_type in datasets_types:
            if self.datasets_metadata.get(dataset_type) is not None:
                print(
                    colorize(f"Info regarding {dataset_type}_dataset:", dataset_color[dataset_type]),
                    colorize("\nLength: ", "LIGHTBLUE_EX"), self.datasets_metadata[dataset_type]['length'],       
                    colorize("\nImages per class: ", "LIGHTBLUE_EX"), self.datasets_metadata[dataset_type]['count_per_class'], '\n'     
                )        
        
        if n_splits:
            for i in range(n_splits):
                # Print info for train and valid datasets for each cross-validation fold
                for dataset_type in self.cross_val_datasets:
                    print(
                        colorize(f"Info regarding {dataset_type}_dataset, fold -- {i} -- of cross-validation:", dataset_color[dataset_type]),
                        colorize("\nLength: ", "LIGHTBLUE_EX"), self.cross_val_datasets_metadata[dataset_type][i]['length'],       
                        colorize("\nImages per class: ", "LIGHTBLUE_EX"), self.cross_val_datasets_metadata[dataset_type][i]['count_per_class'], '\n'     
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

         
    def _inverse_normalize_img(self, tensor, mean, std):
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
                image = self._inverse_normalize_img(image, self.mean, self.std)
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
 
 
 
 
 
 
 
 
