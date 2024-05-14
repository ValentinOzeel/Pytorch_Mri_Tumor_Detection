import os
from pathlib import Path
from typing import Tuple, Dict, List
from PIL import Image 
import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold



class DataPrep():
    def __init__(self, root:str, random_seed:int=None):
        
        if not os.path.exists(root): raise FileNotFoundError(f"The root path is not valid: {root}")
        # List all Path of .jpg images in root dir (expect following path format: root/class/image.jpg)
        self.all_img_paths_in_root = list(Path(root).glob("*/*.jpg"))
        
        self.random_seed = random_seed
        
    def create_path_class_df(self):
        self.original_df = pd.DataFrame(
            {img_path:img_path.parent.name for img_path in self.all_img_paths_in_root},
            columns=['path', 'class']
            )
         
    def train_test_presplit(self, train_ratio:float):
        if hasattr(self, 'val_df') or hasattr(self, 'cv_indices'):
            raise ValueError("Cannot run 'train_test_presplit' method after 'train_valid_split' nor 'cv_splits' methods.")
        if not hasattr(self, 'original_df'):
            raise ValueError("User must create the original df with the 'create_df' method before being able to run the 'train_test_presplit' method.")
        
        if train_ratio > 1 or train_ratio < 0:
            raise ValueError("train_test_presplit's train_ratio parameter must be a float comprised between 0 and 1")
        
        X, y = self.original_df.drop(columns=['class']), self.original_df['class']
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=self.random_seed)
        train_indices, test_indices = next(sss.split(X, y))
    
        self.train_df = self.original_df.loc[train_indices]
        self.test_df = self.original_df.loc[test_indices]
    
        return self.train_df, self.test_df


    def train_valid_split(self, train_ratio:float):
        if train_ratio > 1 or train_ratio < 0:
            raise ValueError("train_valid_presplit's train_ratio parameter must be a float comprised between 0 and 1")
        
        # If class has self.train_df, it means a split has already been done so we consider this, otherwise we consider the original df 
        train_df = self.train_df if hasattr(self, 'train_df') else self.original_df
        
        X, y = train_df.drop(columns=['class']), train_df['class']
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=self.random_seed)
        train_indices, val_indices = next(sss.split(X, y))
    
        self.train_df = train_df.loc[train_indices]
        self.val_df = train_df.loc[val_indices]
    
        return self.train_df, self.val_df
    
    def cv_splits(self, n_splits:int=5, shuffle:bool=True, kf=None):
        # If class has self.train_df, it means a split has already been done so we consider this, otherwise we consider the original df 
        train_df = self.train_df if hasattr(self, 'train_df') else self.original_df

        # Use kf if exists else use StratifiedKFold
        kfold = kf if kf else StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_seed)
            
        self.cv_indices = {}
        X, y = train_df.drop(columns=['class']), train_df['class']
            
        for i, (train_indices, val_indices) in enumerate(kfold.split(X, y)):
            self.cv_dfs[i]['train'] = train_df.loc[train_indices]
            self.cv_dfs[i]['val'] = train_df.loc[val_indices]

        return self.cv_dfs
    
    
    
class CustomImageFolder(Dataset):
    '''
    Inherit from torch.utils.data.Dataset to create custom Dataset that replicate ImageFolder (torchvision.datasets.ImageFolder)
    '''
    def __init__(self,
                 path_class_df: pd.DataFrame,
                 transform = None,
                 target_transform = None):
        
        self.path_class_df = path_class_df
        self.transform = transform
        self.target_transform = target_transform
        
        # Run get_classes method during initialization to get self.classes and self.classes_dict
        self.classes, self.class_to_idx = self._get_classes()

    def _get_classes(self) -> Tuple[List[str], Dict[str, int]]:
        # Get the class names (unique values in label column)
        classes = sorted(self.path_class_df['class'].unique().tolist())
        # Get the dict of classes and associated labels
        class_to_idx = {this_class:label for label, this_class in enumerate(classes)}
        return classes, class_to_idx
    
    def __len__(self) -> int:
        # Overwrite Dataset's __len__ method with the len of path_class_df (number of entries)
        return len(self.path_class_df)
    
    def _load_image(self, path: str):
        return Image.open(path)
    
    def _convert_mode_L_to_RGB(self, image):
        # Convert to RGB if the mode is not already RGB
        return image.convert('RGB') if image.mode != 'RGB' else image
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Overwrite Dataset's __getitem__ method to return one data sample (data, label) potentially transformed
        
        # Get image path and class
        img_path, img_class = self.path_class_df.loc[index, 'path'], self.path_class_df.loc[index, 'class']
        # Load image
        image = self._load_image(img_path)
        image = self._convert_mode_L_to_RGB(image)
        # Get label
        class_label = self.class_to_idx[img_class]
        # Potentially transform image and label
        image = self.transform(image) if self.transform else image 
        class_label = self.target_transform(class_label) if self.target_transform else class_label
        return image, class_label
    
    
    
    
    
    
    
    
    
    

 