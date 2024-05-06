import os
import torch
from torchvision import datasets, transforms
from torchinfo import summary

from sklearn.model_selection import KFold

from timeit import default_timer as timer

from data_loading import LoadOurData
from datasets import CustomImageFolder
from secondary_module import ConfigLoad, check_cuda_availability, color_print
from model import MRINeuralNet, TrainTestEval





class Main():
    def __init__(self,
                 data_dir,
                 train_test_split = 0.9,
                 random_seed=None,
                 ):
        
        self.random_seed = random_seed
    
        # Get config data     
        self.config_load = ConfigLoad()
        self.config = self.config_load.get_config()
        
        self.train_transform, self.test_transform = self._get_transforms()

        # Load original as well as train/test datasets
        self.load = LoadOurData(data_dir, self.config_load.get_dataset())
        self.load.train_test_split(train_test_split)
    
    
    def _get_transforms(self):
        # train_transform used for train and valid datasets
        train_transform = transforms.Compose(self.config_load.get_transform_steps())
        # test_transform used for test_dataset (because we want to predict on real life data (only resized and transformed asd tensor))
        test_transform = transforms.Compose(self.config_load.get_transform_steps(dataset_type='test'))  
        return train_transform, test_transform
        
        
    def load_data(self,
                  train_size:float=0.85,
                  cv:int=None,
                  kf=None,
                  verbose=True):
        
        if not cv:
            # Split train_dataset into train_dataset and valid_dataset
            self.load.train_valid_split(train_size=train_size)

            
            self.load.train_dataset, self.load.valid_dataset, self.load.test_dataset = self.load.apply_transformations(dataset_transform=[(self.load.train_dataset, self.train_transform), 
                                                                                                                                          (self.load.valid_dataset, self.train_transform), 
                                                                                                                                          (self.load.test_dataset,  self.test_transform)])
#
   #         self.load.apply_transformations(dataset_transform=[(self.load.train_dataset, self.train_transform), 
   #                                                            (self.load.valid_dataset, self.train_transform), 
   #                                                            (self.load.test_dataset,  self.test_transform)])

            self.load.generate_dataloaders(BATCH_SIZE=self.config['DATA_LOADER']['BATCH_SIZE'])
            
            # Store datasets' metadata (len, count_per_class)
            for dataset_type, dataset in [('train', self.load.train_dataset), 
                                          ('valid', self.load.valid_dataset), 
                                          ('test', self.load.test_dataset)]:
                self.load.datasets_metadata[dataset_type] = self.load.get_dataset_metadata(dataset)
                
            if verbose: self.load.print_dataset_info()
                
            
        else:            
            # Store train and test datasets' metadata (len, count_per_class)
            for dataset_type, dataset in [('train', self.load.train_dataset), 
                                          ('test', self.load.test_dataset)]:
                self.load.datasets_metadata[dataset_type] = self.load.get_dataset_metadata(dataset)
                
            # Generate the datasets for cross-validation
            kf = kf if kf is not None else KFold(n_splits=cv, shuffle=True, random_state=self.random_seed)
            self.load.generate_cv_datasets(kf)   
            
            # Get metadata for train and valid datasets for each cv fold
            for dataset_type in self.load.cross_valid_datasets:
                for fold, dataset in enumerate(self.load.cross_valid_datasets[dataset_type]):
                    self.load.cross_valid_datasets_metadata[dataset_type][fold] = self.load.get_dataset_metadata(dataset)
                    
            if verbose: self.load.print_dataset_info(n_splits=kf.get_n_splits())
            
            # Apply transformation to the test dataset
            self.load.apply_transformations(dataset_transform=[(self.load.test_dataset,  self.test_transform)])
            # Apply transformation to train and valid datasets for each fold
            for dataset_type in ['train', 'valid']:
                self.load.apply_transformations(dataset_transform=[
                    (x, self.train_transform) for x in self.load.cross_valid_datasets[dataset_type]
                    ])

            # Generate dataloader for train - valid datasets for each fold of cv
            self.load.generate_cv_dataloaders(BATCH_SIZE=self.config['DATA_LOADER']['BATCH_SIZE'])
            






            

if __name__ == "__main__":
    # Get training and testing datapa paths
    # _______________
    # Assuming data_exploration.py is in src\main.py
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root_path, 'data')
    
    dl = Main(data_dir, train_test_split = 0.85)
    
    # Load data into datasets and dataloaders
    # _______________
    #dl.load_data(cv=5)
    dl.load_data()
    

    # Get one iteration of train_dataloader (loading in batches)
    img_batch, label_batch = next(iter(dl.load.train_dataloader))
    #img_batch, label_batch = next(iter(dl.load.cross_valid_dataloaders['train']))
    print('Dataloader batches:', 'Image shapes', img_batch.shape, 'label shapes', label_batch.shape)

    # Print random transformed images
    dl.load.show_random_images(RANDOM_SEED=dl.config['RANDOM_SEED'], dataset_type='train', display_seconds=20)
    


    # Model
    # Setup device-agnostic device
    check_cuda_availability()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define model 
    # Convert the torch.Size object to a tuple of integers
    base_model = MRINeuralNet(input_shape=img_batch.shape, 
                              hidden_units=dl.config['MODEL_PARAMS']['HIDDEN_UNITS'], 
                              output_shape=len(dl.load.classes)
                              ).to(device)

    # Put img_batch to device
    # Try model with one batch
    output = base_model(img_batch.to(device))
    summary(base_model, input_size=img_batch.shape)
    # Get optimizer and intialize its parameters (all from config)
    optimizer_name = next(iter(dl.config['MODEL_PARAMS']['OPTIMIZER']))
    optimizer = getattr(torch.optim, optimizer_name)
    optimizer_params = dl.config['MODEL_PARAMS']['OPTIMIZER'][optimizer_name]
    # Get loos function
    loss_func =  getattr(torch.nn, dl.config['MODEL_PARAMS']['LOSS_FUNC'])

    # Initiate TrainTestEval class instance
    train_test_eval_inst = TrainTestEval(
                                     model = base_model,
                                     optimizer = optimizer(params=base_model.parameters(), **optimizer_params),
                                     loss_func = loss_func(),
                                     epochs = dl.config['MODEL_PARAMS']['EPOCHS'],
                                     device = device,
                                     RANDOM_SEED = dl.config['RANDOM_SEED']
                                    )
    
    start_time = timer()
    base_model_results = train_test_eval_inst.training()
    end_time = timer()
    training_time = f"{(end_time-start_time):.4f}"
    print(f"Training time: {color_print(training_time, 'LIGHTRED_EX')} seconds") 
    
    
    start_time = timer()
    inference_loss, inference_acc = train_test_eval_inst.inference()
    end_time = timer()
    training_time = f"{(end_time-start_time):.4f}"
    print(f"Inference time: {color_print(training_time, 'LIGHTRED_EX')} seconds") 
     
    
    
#    
#
#    
#    
#    # Load and transform our MRI images
#    # _______________
#    
#    # Compare our custom dataset loading (load_instance) VS ImageFolder loading (instance_imagefolder):
#    # Our custom dataset
#    load_instance = LoadOurData(data_dir, OurCustomDataset)
#    load_instance.load_data(train_transform,
#                            test_transform=test_transform,
#                            target_transform=None,
#                            train_ratio=0.8,
#                            valid_ratio=0.1,
#                            test_ratio=0.1
#                            )
#    load_instance.print_info_on_loaded_data()
#    
#    ### ImageFolder dataset
#    #load_instance = LoadOurData(data_dir, datasets.ImageFolder)
#    #load_instance.load_data(train_transform,
#    #                        test_transform=test_transform,
#    #                        target_transform=None,
#    #                        train_ratio=0.8,
#    #                        valid_ratio=0.1,
#    #                        test_ratio=0.1
#    #                        )
#    #load_instance.print_info_on_loaded_data()
#
#    # Print random transformed images
#    load_instance.show_random_images(RANDOM_SEED=config['RANDOM_SEED'], dataset_type='train', display_seconds=20)
#    
#    
#    # Create DataLoaders to load images per in batches
#    # _______________
#    load_instance.create_dataloaders(BATCH_SIZE = config['DATA_LOADER']['BATCH_SIZE'])
#    # Get one iteration of train_dataloader (loading in batches)
#    img_batch, label_batch = next(iter(load_instance.train_dataloader))
#    print('Dataloader batches:', 'Image shapes', img_batch.shape, 'label shapes', label_batch.shape)
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    # Model
#    # Setup device-agnostic device
#    check_cuda_availability()
#    device = "cuda" if torch.cuda.is_available() else "cpu"
#    # Define model 
#    # Convert the torch.Size object to a tuple of integers
#    base_model = MRINeuralNet(input_shape=img_batch.shape, 
#                              hidden_units=config['MODEL_PARAMS']['HIDDEN_UNITS'], 
#                              output_shape=len(load_instance.train_classes)
#                              ).to(device)
#
#    # Put img_batch to device
#    # Try model with one batch
#    output = base_model(img_batch.to(device))
#    summary(base_model, input_size=img_batch.shape)
#    # Get optimizer and intialize its parameters (all from config)
#    optimizer_name = next(iter(config['MODEL_PARAMS']['OPTIMIZER']))
#    optimizer = getattr(torch.optim, optimizer_name)
#    optimizer_params = config['MODEL_PARAMS']['OPTIMIZER'][optimizer_name]
#    # Get loos function
#    loss_func =  getattr(torch.nn, config['MODEL_PARAMS']['LOSS_FUNC'])
#
#    # Initiate TrainTestEval class instance
#    train_test_eval_inst = TrainTestEval(
#                                     model = base_model,
#                                     train_dataloader = load_instance.train_dataloader,
#                                     valid_dataloader = load_instance.valid_dataloader,
#                                     test_dataloader = load_instance.test_dataloader,
#                                     optimizer = optimizer(params=base_model.parameters(), **optimizer_params),
#                                     loss_func = loss_func(),
#                                     epochs = config['MODEL_PARAMS']['EPOCHS'],
#                                     device = device,
#                                     RANDOM_SEED = config['RANDOM_SEED']
#                                    )
#    
#    start_time = timer()
#    base_model_results = train_test_eval_inst.training()
#    end_time = timer()
#    training_time = f"{(end_time-start_time):.4f}"
#    print(f"Training time: {color_print(training_time, 'LIGHTRED_EX')} seconds") 
#    
#    
#    start_time = timer()
#    inference_loss, inference_acc = train_test_eval_inst.inference()
#    end_time = timer()
#    training_time = f"{(end_time-start_time):.4f}"
#    print(f"Inference time: {color_print(training_time, 'LIGHTRED_EX')} seconds") 
#    