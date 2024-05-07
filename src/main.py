import os
import torch
from typing import Dict
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from sklearn.model_selection import KFold

from timeit import default_timer as timer

from data_loading import LoadOurData
from datasets import CustomImageFolder
from secondary_module import ConfigLoad, color_print, check_cuda_availability
from model import MRINeuralNet, TrainTestEval





class Main():
    def __init__(self,
                 data_dir,
                 device,
                 train_test_split = 0.9,
                 RANDOM_SEED=False,
                 ):
        
        self.device = device
        # Get config data     
        self.config_load = ConfigLoad()
        self.config = self.config_load.get_config()
        
        self.RANDOM_SEED = self.config['RANDOM_SEED'] if RANDOM_SEED is False else RANDOM_SEED
        self.train_transform, self.test_transform = self._get_data_transforms()
        # Load original as well as train/test datasets
        self.load = LoadOurData(data_dir, self.config_load.get_dataset(), RANDOM_SEED=self.RANDOM_SEED)
        self.load.train_test_split(train_test_split)
        
    
    def _get_data_transforms(self):
        # train_transform used for train and valid datasets
        train_transform = transforms.Compose(self.config_load.get_transform_steps())
        # test_transform used for test_dataset (because we want to predict on real life data (only resized and transformed asd tensor))
        test_transform = transforms.Compose(self.config_load.get_transform_steps(dataset_type='test'))  
        return train_transform, test_transform
        
    def load_data(self, train_size:float=0.85, verbose=True):
        # Split train_dataset into train_dataset and valid_dataset
        self.load.train_valid_split(train_size=train_size)
        # Apply transformation to train, valid and test datasets
        self.load.train_dataset, self.load.valid_dataset, self.load.test_dataset = self.load.apply_transformations(
            dataset_transform=[(self.load.train_dataset, self.train_transform), 
                               (self.load.valid_dataset, self.train_transform), 
                               (self.load.test_dataset,  self.test_transform)]
            )
        # Create corresponding dataloaders
        self.load.generate_dataloaders(BATCH_SIZE=self.config['DATA_LOADER']['BATCH_SIZE'])

    
        # Store datasets' metadata (len, count_per_class)
        for dataset_type, dataset in [('train', self.load.train_dataset), 
                                      ('valid', self.load.valid_dataset), 
                                      ('test', self.load.test_dataset)]:
            self.load.datasets_metadata[dataset_type] = self.load.get_dataset_metadata(dataset)
        if verbose: self.load.print_dataset_info() 

        # Print random transformed images
        self.load.show_random_images(self.load.train_dataloader, display_seconds=20)
        
        # Get one iteration of train_dataloader (loading in batches) for model's input shape
        imgs, labels = next(iter(self.load.train_dataloader))
        print('Dataloader batches:', 'Image shapes', imgs.shape, 'label shapes', labels.shape)
        
        return imgs.shape, labels.shape

            
    def load_data_cv(self,
                  cv:int=None,
                  kf=None,
                  verbose=True):        
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
    
        # Print random transformed images
        self.load.show_random_images(self.load.cross_valid_dataloaders['train'][0], display_seconds=20)
        
        # Get one iteration of train_dataloader (loading in batches) for model's input shape
        imgs, labels = next(iter(self.load.cross_valid_dataloaders['train'][0]))
        print('Dataloader batches:', 'Image shapes', imgs.shape, 'label shapes', labels.shape)
        
        return imgs.shape, labels.shape


    def get_model_params(self):
        # Get optimizer and intialize its parameters (all from config)
        optimizer_name = next(iter(self.config['MODEL_PARAMS']['OPTIMIZER']))
        self.optimizer = getattr(torch.optim, optimizer_name)
        self.optimizer_params = self.config['MODEL_PARAMS']['OPTIMIZER'][optimizer_name]
        # Get loos function
        self.loss_func =  getattr(torch.nn, self.config['MODEL_PARAMS']['LOSS_FUNC'])
        return self.optimizer, self.optimizer_params, self.loss_func

    def get_MRINeuralNet_instance(self, input_shape, hidden_units, output_shape):
        self.model = MRINeuralNet(input_shape, hidden_units, output_shape)
        return self.model
        
    def get_TrainTestEval_instance(self, model, optimizer, loss_func, epochs = 10):
        return TrainTestEval(model, optimizer, loss_func, epochs=epochs, device=self.device, RANDOM_SEED=self.RANDOM_SEED)

    def run_cross_validation(self, TrainTestEval_instance:TrainTestEval, cross_valid_dataloaders:Dict):
        training_metrics_per_fold = TrainTestEval_instance(cross_valid_dataloaders)
        return training_metrics_per_fold
        
    def train_model(self, TrainTestEval_instance:TrainTestEval, train_dataloader:DataLoader, valid_dataloader:DataLoader=None):
        start_time = timer()
        base_model_results = TrainTestEval_instance.training(train_dataloader, valid_dataloader)
        end_time = timer()
        training_time = f"{(end_time-start_time):.4f}"
        print(f"Training time: {color_print(training_time, 'LIGHTRED_EX')} seconds") 
        
        return TrainTestEval_instance.model, base_model_results

    def evaluate_model(self, TrainTestEval_instance:TrainTestEval, test_dataloader:DataLoader):
        start_time = timer()
        test_loss, test_acc = TrainTestEval_instance.inference(test_dataloader)
        end_time = timer()
        eval_time = f"{(end_time-start_time):.4f}"
        print(f"Evaluation time: {color_print(eval_time, 'LIGHTRED_EX')} seconds")
        
        return test_loss, test_acc

    def make_inference(self, data_dir, DatasetClass:Dataset, transform:transforms, BATCH_SIZE:int, TrainTestEval_instance:TrainTestEval, target_transform:transforms=None):
        
        load = LoadOurData(data_dir, DatasetClass, RANDOM_SEED=self.RANDOM_SEED, 
                           initial_transform=transform, initial_target_transform=target_transform, 
                           inference=True)
        dataset = load.original_dataset
        dataloader = load.create_dataloaders(dataset, BATCH_SIZE, shuffle=False)
        
        pred_logits, pred_classes = TrainTestEval_instance.inference(dataloader)
        return pred_logits, pred_classes
        

if __name__ == "__main__":
    # Get training and testing datapa paths
    # _______________
    # Assuming data_exploration.py is in src\main.py
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root_path, 'data')

    # Model
    # Setup device-agnostic device
    check_cuda_availability()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dl = Main(data_dir, train_test_split = 0.85, device=device)
    
    # Load data into datasets and dataloaders
    # _______________
    #dl.load_data_cv()
    input_shape, labels_shape = dl.load_data()
        
    # Define model 
    base_model = dl.get_MRINeuralNet_instance(input_shape=input_shape, 
                                              hidden_units=dl.config['MODEL_PARAMS']['HIDDEN_UNITS'],
                                              output_shape=len(dl.load.classes))

    # Get optimizer and intialize its parameters as well as loss function (all from config)
    optimizer, optimizer_params, loss_func = dl.get_model_params()

    # Initiate TrainTestEval class instance
    train_test_eval_inst = dl.get_TrainTestEval_instance(
                                     model = base_model,
                                     optimizer = optimizer(params=base_model.parameters(), **optimizer_params),
                                     loss_func = loss_func(),
                                     epochs = dl.config['MODEL_PARAMS']['EPOCHS'],
                                    )
    

    trained_model, results = dl.train_model(train_test_eval_inst, dl.load.train_dataloader, dl.load.valid_dataloader)
    test_loss, test_acc = dl.evaluate_model(train_test_eval_inst, dl.load.test_dataloader)

