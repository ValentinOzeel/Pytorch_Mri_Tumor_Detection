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
from model import MRI_CNN, EarlyStopping, TrainTestEval, MetricsTracker





class Main():
    def __init__(self, data_dir_path, DatasetClass, device, random_seed):

        
        
        self.device = device
        self.random_seed = random_seed
        # Create an instance of LoadOurData (enable to create datasets, dataloaders..)
        self.load = LoadOurData(data_dir_path, DatasetClass, test_data_dir_path=None, random_seed=self.random_seed)
        

    def train_test_presplit(self, train_ratio:float=0.9):
        # Split the data (data represents a df (cols = ['path', 'label']) in datasets.DataPrep) into train and valid dfs
        self.load.train_test_presplit(train_ratio=train_ratio)
        
    def load_data(self, data_loader_params:Dict, dict_transform_steps:Dict[str, List], dict_target_transform_steps:Dict[str, List] = None, train_ratio:float=0.9,
                  show_random_images:int=6, image_display_seconds:int=30, verbose=True):
        # Generate Datasets 
        self.load.generate_datasets(train_ratio, dict_transform_steps, dict_target_transform_steps=dict_target_transform_steps)
        # Store datasets' metadata (len, count_per_class)
        self.load.store_datasets_metadata()
        if verbose: self.load.print_dataset_info() 
        # Generate DataLoaders
        self.load.generate_dataloaders(data_loader_params)
        # Print random transformed images
        if show_random_images:
            self.load.show_random_images(self.load.train_dataloader, n=show_random_images, display_seconds=image_display_seconds, unnormalize=True)
        # Get one iteration of train_dataloader (loading in batches) for model's input shape
        imgs, labels = next(iter(self.load.train_dataloader))
        print('Dataloader batches:', 'Image shapes', imgs.shape, 'label shapes', labels.shape)
        return imgs.shape, labels.shape

            
    def load_data_cv(self,
                  data_loader_params:Dict, dict_transform_steps:Dict[str, List], dict_target_transform_steps:Dict[str, List]=None,
                  train_ratio:float=0.85,
                  kf=None,
                  show_random_images:int=6, image_display_seconds:int=30, verbose=True):
            
        # Calculate the number of splits based on the train/validation ratio
        n_splits = int(1 / (1 - train_ratio))
        # Generate the datasets for cross-validation
        self.load.generate_cv_datasets(dict_transform_steps, dict_target_transform_steps=dict_target_transform_steps, 
                                       n_splits=n_splits, shuffle=True, kf=kf)
        # Get metadata for train and valid datasets for each cv fold
        self.load.store_datasets_metadata(cv=True)        
        if verbose: self.load.print_dataset_info(n_splits=self.load.cv_n_splits)
        # Generate dataloader for train - valid datasets for each fold of cv
        self.load.generate_cv_dataloaders(data_loader_params)
        # Print random transformed images
        if show_random_images:
            self.load.show_random_images(self.load.cross_val_dataloaders['train'][0], n=show_random_images, display_seconds=image_display_seconds, unnormalize=True)
        # Get one iteration of train_dataloader (loading in batches) for model's input shape
        imgs, labels = next(iter(self.load.cross_valid_dataloaders['train'][0]))
        print('Dataloader batches:', 'Image shapes', imgs.shape, 'label shapes', labels.shape)
        
        return imgs.shape, labels.shape

    def get_MRI_CNN_instance(self, input_shape, hidden_units, output_shape, activation_func):
        return MRI_CNN(input_shape, hidden_units, output_shape, activation_func)
        
    def get_MetricsTracker_instance(self, metrics:List[str], n_classes:int, average:str='macro', torchmetrics:Dict={}):
        return MetricsTracker(metrics, n_classes, average=average, torchmetrics=torchmetrics)
    
    def get_TrainTestEval_instance(self, model, optimizer, loss_func, metrics_tracker, epochs = 10, lr_scheduler=None, early_stopping=None):
        return TrainTestEval(model, optimizer, loss_func, metrics_tracker=metrics_tracker, epochs=epochs, lr_scheduler=lr_scheduler, early_stopping=early_stopping, device=self.device, random_seed=self.random_seed)

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

    def make_inference(self, data_dir_path, DatasetClass:Dataset, transform:transforms, batch_size:int, TrainTestEval_instance:TrainTestEval, target_transform:transforms=None):
        
        load = LoadOurData(data_dir_path, DatasetClass, random_seed=self.random_seed, 
                           initial_transform=transform, initial_target_transform=target_transform, 
                           inference=True)
        dataset = load.original_dataset
        dataloader = load.create_dataloaders(dataset, batch_size, shuffle=False)
        
        pred_classes = TrainTestEval_instance.inference(dataloader)
        return pred_classes
        

if __name__ == "__main__":
    # Get data path
    # _______________
    # Assuming data_exploration.py is in src\main.py
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir_path = os.path.join(project_root_path, 'data')

    # Setup device-agnostic device
    check_cuda_availability()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    



    
    

    
    
    ###########                 ###########
    ########### GET CONFIG DATA ###########
    ###########                 ###########
    config_load = ConfigLoad()
    config = config_load.get_config()
    
    random_seed = config['RANDOM_SEED']
    # Dataset class used
    DatasetClass = config_load.get_dataset()
    # test_transform used for test_dataset (because we want to predict on real life data (only resized and transformed asd tensor))
    train_transform_steps, test_transform_steps = config_load.get_transform_steps(dataset_type='train'), config_load.get_transform_steps(dataset_type='test')  
    # Create dict_transform_steps
    dict_transform_steps = {'train':train_transform_steps, 'val':train_transform_steps, 'test':test_transform_steps}
    # Get train ratios
    train_ratio_presplit, train_ratio_split = config['DATA_SPLITS']['train_test_ratio'], config['DATA_SPLITS']['train_val_ratio']
    # Get dataloader params
    data_loader_params = config['DATA_LOADER']
    # Model params
    # Get optimizer and intialize its parameters as well as loss function (all from config)
    optimizer_name = next(iter(config['MODEL_PARAMS']['optimizer']))
    optimizer = getattr(torch.optim, optimizer_name)
    optimizer_params = config['MODEL_PARAMS']['optimizer'][optimizer_name]
    # Get loss function
    loss_func =  getattr(torch.nn, config['MODEL_PARAMS']['loss_func'])
    # Get metrics to track
    metrics_to_track = config['MODEL_PARAMS']['metrics']
    # Get torchmetrics to track
    torch_metrics = config_load.get_torchmetrics_dict(device)
    # Get acctivation function
    activation_func = getattr(torch.nn, config['MODEL_PARAMS']['activation_func'])
    # Get hidden_units and epochs
    hidden_units = config['MODEL_PARAMS']['hidden_units']
    epochs = config['MODEL_PARAMS']['epochs']
    # Define lr_scheduler obj and params
    lr_schd_name, lr_scheduler_params = config_load.get_nested_param(config['MODEL_PARAMS']['lr_scheduler'])
    lr_scheduler_obj = getattr(lr_scheduler, lr_schd_name)
    # Early stopping
    if config['MODEL_PARAMS'].get('early_stopping'):
        early_stopping = EarlyStopping(**config['MODEL_PARAMS']['early_stopping'])



    # Initiate Main class
    dl = Main(data_dir_path, DatasetClass, device=device, random_seed=random_seed)
    # Presplit original dataset into train and test dfs 
    dl.train_test_presplit(train_ratio=train_ratio_presplit)
    
    # Load data into datasets and dataloaders
    # _______________    
    input_shape, labels_shape = dl.load_data(data_loader_params, dict_transform_steps, train_ratio=train_ratio_split)
    #input_shape, labels_shape = dl.load_data_cv(data_loader_params, dict_transform_steps, train_ratio=train_ratio_split)
        
    # Define model 
    base_model = dl.get_MRI_CNN_instance(input_shape=input_shape, 
                                              hidden_units=hidden_units,
                                              output_shape=len(dl.load.classes),
                                              activation_func=activation_func)
    
    # Print torchinfo's model summary
    summary(base_model, input_size=input_shape)

    # Get metrics to track during training/validation
    metrics_tracker = dl.get_MetricsTracker_instance(metrics_to_track, len(dl.load.classes), torchmetrics=torch_metrics)


    # Initiate TrainTestEval class instance
    optimizer_inst = optimizer(params=base_model.parameters(), **optimizer_params)

    train_test_eval_inst = dl.get_TrainTestEval_instance(
                                     model = base_model,
                                     optimizer = optimizer_inst,
                                     loss_func = loss_func(),
                                     metrics_tracker = metrics_tracker,
                                     epochs = epochs,
                                     lr_scheduler = lr_scheduler_obj(optimizer_inst, **lr_scheduler_params),
                                     early_stopping = early_stopping
                                    )
    

    trained_model, results = dl.train_model(train_test_eval_inst, dl.load.train_dataloader, dl.load.valid_dataloader)
    test_loss, test_acc = dl.evaluate_model(train_test_eval_inst, dl.load.test_dataloader)

