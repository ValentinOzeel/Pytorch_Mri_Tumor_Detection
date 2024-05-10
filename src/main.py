import os
import torch
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import lr_scheduler

from torchinfo import summary

from sklearn.model_selection import KFold

from timeit import default_timer as timer

from data_loading import LoadOurData
from secondary_module import ConfigLoad, colorize, check_cuda_availability
from model import MRINeuralNet, EarlyStopping, TrainTestEval





class Main():
    def __init__(self, data_dir, dataset_class, device, RANDOM_SEED):
        
        self.device = device
        self.RANDOM_SEED = RANDOM_SEED
        # Create an instance of LoadOurData (enable to create datasets, dataloaders..)
        self.load = LoadOurData(data_dir, dataset_class, RANDOM_SEED=self.RANDOM_SEED)
        

    def train_test_presplit(self, train_test_ratio:float=0.9):
        self.load.train_test_split(train_test_ratio)
        
    def load_data(self, data_loader_params:Dict, train_transform:transforms, test_transform:transforms=None, train_valid_ratio:float=0.85, 
                  show_random_image:int=20, verbose=True):
        # Split train_dataset into train_dataset and valid_dataset
        self.load.train_valid_split(train_ratio=train_valid_ratio)
        # Apply transformation to train, valid and test datasets
        self.load.train_dataset, self.load.valid_dataset, self.load.test_dataset = self.load.apply_transformations(
            dataset_transform=[(self.load.train_dataset, train_transform), 
                               (self.load.valid_dataset, train_transform), 
                               (self.load.test_dataset,  test_transform)]
            )
        # Create corresponding dataloaders
        self.load.generate_dataloaders(data_loader_params)

    
        # Store datasets' metadata (len, count_per_class)
        for dataset_type, dataset in [('train', self.load.train_dataset), 
                                      ('valid', self.load.valid_dataset), 
                                      ('test',  self.load.test_dataset)]:
            self.load.datasets_metadata[dataset_type] = self.load.get_dataset_metadata(dataset)
        if verbose: self.load.print_dataset_info() 

        # Print random transformed images
        self.load.show_random_images(self.load.train_dataloader, display_seconds=show_random_image, unnormalize=True)
        
        # Get one iteration of train_dataloader (loading in batches) for model's input shape
        imgs, labels = next(iter(self.load.train_dataloader))
        print('Dataloader batches:', 'Image shapes', imgs.shape, 'label shapes', labels.shape)
        
        return imgs.shape, labels.shape

            
    def load_data_cv(self,
                  data_loader_params:Dict, train_transform:transforms, test_transform:transforms=None,
                  train_valid_ratio:float=0.85,
                  kf=None,
                  show_random_image:int=20, verbose=True):        
        # Store train and test datasets' metadata (len, count_per_class)
        for dataset_type, dataset in [('train', self.load.train_dataset), 
                                      ('test', self.load.test_dataset)]:
            self.load.datasets_metadata[dataset_type] = self.load.get_dataset_metadata(dataset)
            
        # Calculate the number of splits based on the train/validation ratio
        n_splits = int(1 / (1 - train_valid_ratio))
        # Generate the datasets for cross-validation
        kf = kf if kf is not None else KFold(n_splits=n_splits, shuffle=True, random_state=self.RANDOM_SEED)
        self.load.generate_cv_datasets(kf)   
        
        # Get metadata for train and valid datasets for each cv fold
        for dataset_type in self.load.cross_valid_datasets:
            for fold, dataset in enumerate(self.load.cross_valid_datasets[dataset_type]):
                self.load.cross_valid_datasets_metadata[dataset_type][fold] = self.load.get_dataset_metadata(dataset)
        
        if verbose: self.load.print_dataset_info(n_splits=kf.get_n_splits())
        
        # Apply transformation to the test dataset
        self.load.apply_transformations(dataset_transform=[(self.load.test_dataset, test_transform)])
        # Apply transformation to train and valid datasets for each fold
        for dataset_type in ['train', 'valid']:
            self.load.apply_transformations(dataset_transform=[
                (x, train_transform) for x in self.load.cross_valid_datasets[dataset_type]
                ])
        # Generate dataloader for train - valid datasets for each fold of cv
        self.load.generate_cv_dataloaders(data_loader_params)
    
        # Print random transformed images
        self.load.show_random_images(self.load.cross_valid_dataloaders['train'][0], display_seconds=show_random_image, unnormalize=True)
        
        # Get one iteration of train_dataloader (loading in batches) for model's input shape
        imgs, labels = next(iter(self.load.cross_valid_dataloaders['train'][0]))
        print('Dataloader batches:', 'Image shapes', imgs.shape, 'label shapes', labels.shape)
        
        return imgs.shape, labels.shape

    def get_MRINeuralNet_instance(self, input_shape, hidden_units, output_shape, activation_func):
        return MRINeuralNet(input_shape, hidden_units, output_shape, activation_func)
        
    def get_TrainTestEval_instance(self, model, optimizer, loss_func, epochs = 10, lr_scheduler=None, early_stopping=None):
        return TrainTestEval(model, optimizer, loss_func, epochs=epochs, lr_scheduler=lr_scheduler, early_stopping=early_stopping, device=self.device, RANDOM_SEED=self.RANDOM_SEED)

    def run_cross_validation(self, TrainTestEval_instance:TrainTestEval, cross_valid_dataloaders:Dict):
        training_metrics_per_fold = TrainTestEval_instance.cross_validation(cross_valid_dataloaders)
        return training_metrics_per_fold
        
    def train_model(self, TrainTestEval_instance:TrainTestEval, train_dataloader:DataLoader, valid_dataloader:DataLoader=None):
        start_time = timer()
        base_model_results = TrainTestEval_instance.training(train_dataloader, valid_dataloader)
        end_time = timer()
        training_time = f"{(end_time-start_time):.4f}"
        print(f"Training time: {colorize(training_time, 'LIGHTRED_EX')} seconds") 
        
        return TrainTestEval_instance.model, base_model_results

    def evaluate_model(self, TrainTestEval_instance:TrainTestEval, test_dataloader:DataLoader):
        start_time = timer()
        test_loss, test_acc = TrainTestEval_instance.evaluation(test_dataloader)
        end_time = timer()
        eval_time = f"{(end_time-start_time):.4f}"
        print(f"Evaluation time: {colorize(eval_time, 'LIGHTRED_EX')} seconds")
        
        return test_loss, test_acc

    def make_inference(self, data_dir, DatasetClass:Dataset, transform:transforms, batch_size:int, TrainTestEval_instance:TrainTestEval, target_transform:transforms=None):
        
        load = LoadOurData(data_dir, DatasetClass, RANDOM_SEED=self.RANDOM_SEED, 
                           initial_transform=transform, initial_target_transform=target_transform, 
                           inference=True)
        dataset = load.original_dataset
        dataloader = load.create_dataloaders(dataset, batch_size, shuffle=False)
        
        pred_logits, pred_classes = TrainTestEval_instance.inference(dataloader)
        return pred_logits, pred_classes
        

if __name__ == "__main__":
    # Get data path
    # _______________
    # Assuming data_exploration.py is in src\main.py
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root_path, 'data')
    
    # Config
    config_load = ConfigLoad()
    config = config_load.get_config()

    # Setup device-agnostic device
    check_cuda_availability()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    # Initiate Main class
    dl = Main(data_dir, config_load.get_dataset(), device=device, RANDOM_SEED=config['RANDOM_SEED'])
    # Presplit original dataset in train and test datasets 
    dl.train_test_presplit(train_test_ratio=config['DATA_SPLITS']['train_test_ratio'])
    
    
    ###########                 ###########
    ########### GET CONFIG DATA ###########
    ###########                 ###########
    RANDOM_SEED = config['RANDOM_SEED']
    # Dataset class used
    dataset_class = config_load.get_dataset()
    # Get mean and standard deviation of the pixel values across all images in our dataset
    normalize_params = dl.load.calculate_normalization()
    # train_transform used for train and valid datasets
    # test_transform used for test_dataset (because we want to predict on real life data (only resized and transformed asd tensor))
    train_transform_steps, test_transform_steps = config_load.get_transform_steps(normalize_params=normalize_params), config_load.get_transform_steps(dataset_type='test', normalize_params=normalize_params)  
    # Create composes
    train_transform, test_transform = transforms.Compose(train_transform_steps), transforms.Compose(test_transform_steps)
    # Get train ratios
    train_test_ratio, train_valid_ratio = config['DATA_SPLITS']['train_test_ratio'], config['DATA_SPLITS']['train_valid_ratio']
    # Get dataloader params
    data_loader_params = config['DATA_LOADER']
    # Model params
    # Get optimizer and intialize its parameters as well as loss function (all from config)
    optimizer_name = next(iter(config['MODEL_PARAMS']['optimizer']))
    optimizer = getattr(torch.optim, optimizer_name)
    optimizer_params = config['MODEL_PARAMS']['optimizer'][optimizer_name]
    # Get loos function
    loss_func =  getattr(torch.nn, config['MODEL_PARAMS']['loss_func'])
    # Get acctivation function
    activation_func = getattr(torch.nn, config['MODEL_PARAMS']['activation_func'])
    # Get hidden_units and epochs
    hidden_units = config['MODEL_PARAMS']['hidden_units']
    epochs = config['MODEL_PARAMS']['epochs']
    # Define lr_scheduler obj and params
    lr_schd_name, lr_scheduler_params = config_load.get_nested_param(config['MODEL_PARAMS']['lr_scheduler'])
    lr_scheduler_obj = getattr(lr_scheduler, lr_schd_name)
    
    if config['MODEL_PARAMS'].get('early_stopping'):
        early_stopping = EarlyStopping(**config['MODEL_PARAMS']['early_stopping'])


    # Load data into datasets and dataloaders
    # _______________

    input_shape, labels_shape = dl.load_data(data_loader_params, train_transform, test_transform=test_transform, train_valid_ratio=train_valid_ratio)
    #input_shape, labels_shape = dl.load_data_cv(data_loader_params, train_transform, test_transform=test_transform, train_valid_ratio=train_valid_ratio, num_workers=6)

        
    # Define model 
    base_model = dl.get_MRINeuralNet_instance(input_shape=input_shape, 
                                              hidden_units=hidden_units,
                                              output_shape=len(dl.load.classes),
                                              activation_func=activation_func)
    
    # Print torchinfo's model summary
    summary(base_model, input_size=input_shape)




    # Initiate TrainTestEval class instance
    optimizer_inst = optimizer(params=base_model.parameters(), **optimizer_params)

    train_test_eval_inst = dl.get_TrainTestEval_instance(
                                     model = base_model,
                                     optimizer = optimizer_inst,
                                     loss_func = loss_func(),
                                     epochs = epochs,
                                     lr_scheduler = lr_scheduler_obj(optimizer_inst, **lr_scheduler_params),
                                     early_stopping = early_stopping
                                    )
    

    trained_model, results = dl.train_model(train_test_eval_inst, dl.load.train_dataloader, dl.load.valid_dataloader)
    test_loss, test_acc = dl.evaluate_model(train_test_eval_inst, dl.load.test_dataloader)

