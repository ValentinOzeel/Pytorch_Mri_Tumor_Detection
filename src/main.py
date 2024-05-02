# Data transformation (into tensor and torch.utils.data.Dataset -> torch.utils.data.DataLoader)
import os
import torch
from torchvision import transforms
from torchinfo import summary

from timeit import default_timer as timer

from data_loading import LoadOurData
from secondary_module import ConfigLoad, check_cuda_availability, color_print
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
    config = conf_instance.get_config()
    transform_steps = conf_instance.get_transform_steps()
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
    load_instance.show_random_images(RANDOM_SEED = config['RANDOM_SEED'], display_seconds=20)
    
    # Create DataLoaders to load images per in batches
    # _______________
    load_instance.create_dataloaders(BATCH_SIZE = config['DATA_LOADER']['BATCH_SIZE'])
    # Get one iteration of train_dataloader (loading in batches)
    img_batch, label_batch = next(iter(load_instance.train_dataloader))
    print('Dataloader batches:', 'Image shapes', img_batch.shape, 'label shapes', label_batch.shape)
    
    # Model
    # Setup device-agnostic device
    check_cuda_availability()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define model 

    # Convert the torch.Size object to a tuple of integers
    base_model = MRINeuralNet(input_shape=img_batch.shape, 
                              hidden_units=config['MODEL_PARAMS']['HIDDEN_UNITS'], 
                              output_shape=len(load_instance.train_classes)
                              ).to(device)

    # Put img_batch to device
    # Try model with one batch
    output = base_model(img_batch.to(device))
    summary(base_model, input_size=img_batch.shape)
    # Get optimizer and intialize its parameters (all from config)
    optimizer_name = next(iter(config['MODEL_PARAMS']['OPTIMIZER']))
    optimizer = getattr(torch.optim, optimizer_name)
    optimizer_params = config['MODEL_PARAMS']['OPTIMIZER'][optimizer_name]
    # Get loos function
    loss_func =  getattr(torch.nn, config['MODEL_PARAMS']['LOSS_FUNC'])

    # Initiate TrainTestEval class instance
    train_test_eval_inst = TrainTestEval(
                                     model = base_model,
                                     train_dataloader = load_instance.train_dataloader,
                                     test_dataloader = load_instance.test_dataloader,
                                     optimizer = optimizer(params=base_model.parameters(), **optimizer_params),
                                     loss_func = loss_func(),
                                     epochs = config['MODEL_PARAMS']['EPOCHS'],
                                     device = device,
                                     RANDOM_SEED = config['RANDOM_SEED']
                                    )

    start_time = timer()
    base_model_results = train_test_eval_inst.training()
    end_time = timer()
    training_time = f"{(end_time-start_time):.4f}"
    print(f"Training time: {color_print(training_time, 'LIGHTRED_EX')} seconds") 