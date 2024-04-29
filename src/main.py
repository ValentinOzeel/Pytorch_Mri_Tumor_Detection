# Data transformation (into tensor and torch.utils.data.Dataset -> torch.utils.data.DataLoader)
import os
from torchvision import transforms

from data_loading import LoadOurData
from secondary_module import ConfigLoad



            
    
if __name__ == "__main__":
    
    # Get training and testing datapa paths
    # _______________
    # Assuming data_exploration.py is in src\main.py
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    train_dir = os.path.join(project_root_path, 'data', 'train')
    test_dir = os.path.join(project_root_path, 'data', 'test')
    
    # Get config data
    # _______________
    conf_instance = ConfigLoad()
    transform_steps = conf_instance.get_transform()
    transform = transforms.Compose(transform_steps)
    
    # Load and transform our MRI images
    # _______________
    
    # Compare our custom dataset loading VS ImageFolder loading:
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
    #instance_imagefolder.print_info_on_loaded_data()
    
    # Print random transformed images
    instance_our_dataset.show_random_images()
    
    # Create DataLoaders to load images per in batches
    # _______________
    instance_our_dataset.create_dataloaders()
    # Get one iteration of train_dataloader (loading in batches)
    img, label = next(iter(instance_our_dataset.train_dataloader))
    print('Dataloader batches:', 'Image shapes', img.shape, 'label shapes', label.shape)
    

    