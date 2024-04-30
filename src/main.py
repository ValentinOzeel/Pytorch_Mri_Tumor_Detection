# Data transformation (into tensor and torch.utils.data.Dataset -> torch.utils.data.DataLoader)
import os
import torch
from torchvision import transforms
from torchinfo import summary
from data_loading import LoadOurData
from secondary_module import ConfigLoad, check_cuda_availability
from model import MRINeuralNet, TrainTestEval
       
    
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
    
    # Compare our custom dataset loading (load_instance) VS ImageFolder loading (instance_imagefolder):
    # Our custom dataset
    load_instance = LoadOurData(train_dir,
                                test_dir,
                                transform)
    load_instance.load_using_OurCustomDataset()
    load_instance.print_info_on_loaded_data()
    
    ### ImageFolder dataset
    #instance_imagefolder = LoadOurData(train_dir,
    #                                   test_dir,
    #                                   transform)  
    #instance_imagefolder.load_using_ImageFolderDataset()
    #instance_imagefolder.print_info_on_loaded_data()
    
    # Print random transformed images
    load_instance.show_random_images()
    
    # Create DataLoaders to load images per in batches
    # _______________
    load_instance.create_dataloaders()
    # Get one iteration of train_dataloader (loading in batches)
    img_batch, label_batch = next(iter(load_instance.train_dataloader))
    print('Dataloader batches:', 'Image shapes', img_batch.shape, 'label shapes', label_batch.shape)
    
    # Model
    # Setup device-agnostic device
    check_cuda_availability()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define model 

    # Convert the torch.Size object to a tuple of integers
    image_shape_tuple = tuple(img_batch[0].shape)
    base_model = MRINeuralNet(input_shape=img_batch.shape, 
                              hidden_units=10, 
                              output_shape=len(load_instance.train_classes)
                              ).to(device)
    print(image_shape_tuple)
    # Put img_batch to device
    # Try model with one batch
    output = base_model(img_batch.to(device))
    summary(base_model, input_size=img_batch.shape)
    
    
    TrainTestEval(
                 base_model,
                 load_instance.train_dataloader,
                 load_instance.test_dataloader,
                 optimizer,
                 nn.CrossEntropyLoss(),
                 epochs = 10,
                 random_seed = None
    )
    
    
    #### NEED TO REMOVE THE CONFIG FROM data_loading. All config params should be set in MAIN