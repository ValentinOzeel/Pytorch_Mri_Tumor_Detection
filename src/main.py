import os
import torch
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.optim import lr_scheduler

from torchinfo import summary

from timeit import default_timer as timer

from data_loading import LoadOurData
from secondary_module import ConfigLoad, colorize, check_cuda_availability
from model import MRI_CNN, EarlyStopping, TrainTestEval, MetricsTracker





class DeepLearningVisionWorkflow():
    '''
    A class for orchestrating the training, evaluation, and inference processes of deep learning models.

    Attributes:
    - device (str): Device (e.g., 'cpu', 'cuda') for running the models.
    - random_seed (int): Random seed for reproducibility.
    - load (LoadOurData): An instance of LoadOurData for creating datasets and data loaders.
    '''
    
    def __init__(self, data_dir_path, DatasetClass, device, random_seed, test_data_dir_path:str=None, inference_only:bool=False):
        '''
        Initialize the DeepLearningVisionWorkflow class with the specified parameters.

        Parameters:
        - data_dir_path (str): Path to the directory containing the data.
        - DatasetClass (Dataset): Class for creating datasets.
        - device (str): Device (e.g., 'cpu', 'cuda') for running the models.
        - random_seed (int): Random seed for reproducibility.
        - test_data_dir_path (str, optional): Path to the directory containing test data. (test data can also be generated though pre-splitting data_dir_path)
        - inference_only (str, optional) : Flag to indicate whether user is going to use the class solely for inference purpose.
        '''
        if not inference_only:
            self.device = device
            self.random_seed = random_seed
            # Create an instance of LoadOurData (enable to create datasets, dataloaders..)
            self.load = LoadOurData(data_dir_path, DatasetClass, test_data_dir_path=test_data_dir_path, random_seed=self.random_seed) 

    def train_test_presplit(self, train_ratio:float=0.9):
        '''
        Pre-split the data into training and testing sets.
        Parameters: - train_ratio (float, optional): Ratio of training data to total data, used to generate training and testing datasets. Default is 0.9.
        '''
        # Split the data (data represents a df (cols = ['path', 'label']) in datasets.DataPrep) into train and valid dfs
        self.load.train_test_presplit(train_ratio=train_ratio)
        
    def load_data(self, data_loader_params:Dict, dict_transform_steps:Dict[str, List], dict_target_transform_steps:Dict[str, List] = None, train_ratio:float=0.9,
                  show_random_images:int=6, image_display_seconds:int=30, verbose=True) -> Tuple[Tuple, Dict[str, DataLoader]]:
        '''
        Load and preprocess data for training, validation and potentially testing.

        Parameters:
        - data_loader_params (Dict): Parameters for creating data loaders.
        - dict_transform_steps (Dict[str, List]): Dictionary containing transformation steps for images.
        - dict_target_transform_steps (Dict[str, List], optional): Dictionary containing transformation steps for labels. Default is None.
        - train_ratio (float, optional): Ratio of training data to total data, used to generate training and validation datasets. Default is 0.9.
        - show_random_images (int, optional): Number of random images to display. Default is 6.
        - image_display_seconds (int, optional): Time to display the random images during visualization. Default is 30.
        - verbose (bool, optional): Whether to print dataset information. Default is True.

        Returns:
        - Tuple[Tuple, Dict[str, DataLoader]]: Tuple containing shapes of images and labels, and a dictionary of data loaders.
        '''
        # Generate Datasets 
        self.load.generate_datasets(train_ratio, dict_transform_steps, dict_target_transform_steps=dict_target_transform_steps)
        # Store datasets' metadata (len, count_per_class)
        self.load.store_datasets_metadata()
        if verbose: self.load.print_dataset_info() 
        # Generate DataLoaders
        self.load.generate_dataloaders(data_loader_params)
        # Get generated dataloaders
        dataloader_types = self.load._get_created_dataset_types(self.load, '_dataloader')
        dataloaders = {dataloader_type:getattr(self.load, ''.join([dataloader_type, '_dataloader'])) for dataloader_type in dataloader_types}
        # Print random transformed images
        if show_random_images:
            self.load.show_random_images(self.load.train_dataloader, n=show_random_images, display_seconds=image_display_seconds, unnormalize=True)
        # Get one iteration of train_dataloader (loading in batches) for model's input shape
        imgs, labels = next(iter(self.load.train_dataloader))
        print('\nDataloader batches:', 'Image shapes', imgs.shape, 'label shapes', labels.shape)
        return (imgs.shape, labels.shape), dataloaders

            
    def load_data_cv(self,
                  data_loader_params:Dict, dict_transform_steps:Dict[str, List], dict_target_transform_steps:Dict[str, List]=None,
                  train_ratio:float=0.9,
                  kf=None,
                  show_random_images:int=6, image_display_seconds:int=30, verbose=True) -> Tuple[Tuple, Dict[str, List[DataLoader]]]:
        '''
        Load and preprocess data for cross-validation.

        Parameters:
        - data_loader_params (Dict): Parameters for creating data loaders.
        - dict_transform_steps (Dict[str, List]): Dictionary containing transformation steps for images.
        - dict_target_transform_steps (Dict[str, List], optional): Dictionary containing transformation steps for labels. Default is None.
        - train_ratio (float, optional): Ratio of training data to total data, used to generate training and validation datasets. Default is 0.9.
        - kf (optional): Instance of KFold for cross-validation. Default is None. StratifiedKFold will be used if not provided.
        - show_random_images (int, optional): Number of random images to display. Default is 6.
        - image_display_seconds (int, optional): Time to display each image during visualization. Default is 30.
        - verbose (bool, optional): Whether to print dataset information. Default is True.

        Returns:
        - Tuple[Tuple, Dict[str, List[DataLoader]]]: Tuple containing shapes of images and labels, and a dictionary of cross-validated data loaders.
        '''
        # Calculate the number of splits based on the train/validation ratio
        n_splits = int(1 / (1 - train_ratio))
        # Generate the datasets for cross-validation
        self.load.generate_cv_datasets(dict_transform_steps, dict_target_transform_steps=dict_target_transform_steps, 
                                       n_splits=n_splits, shuffle=True, kf=kf)
        # Get metadata for train and val datasets for each cv fold
        self.load.store_datasets_metadata(cv=True)        
        if verbose: self.load.print_dataset_info(n_splits=self.load.cv_n_splits)
        # Generate dataloader for train - val datasets for each fold of cv
        self.load.generate_cv_dataloaders(data_loader_params)
        # Print random transformed images
        if show_random_images:
            self.load.show_random_images(self.load.cross_val_dataloaders['train'][0], n=show_random_images, display_seconds=image_display_seconds, unnormalize=True)
        # Get one iteration of train_dataloader (loading in batches) for model's input shape
        imgs, labels = next(iter(self.load.cross_val_dataloaders['train'][0]))
        print('Dataloader batches:', 'Image shapes', imgs.shape, 'label shapes', labels.shape)
        return (imgs.shape, labels.shape),  self.load.cross_val_dataloaders

    def get_MRI_CNN_instance(self, input_shape, hidden_units, output_shape, activation_func) -> MRI_CNN:
        '''
        Create an instance of the MRI_CNN model.

        Parameters:
        - input_shape: Shape of the input data.
        - hidden_units: Number of hidden units.
        - output_shape: Shape of the output data.
        - activation_func: Activation function to use.

        Returns:
        - MRI_CNN: Instance of the MRI_CNN model.
        '''
        return MRI_CNN(input_shape, hidden_units, output_shape, activation_func)
        
    def get_MetricsTracker_instance(self, metrics:List[str], n_classes:int, average:str='macro', torchmetrics:Dict={}) -> MetricsTracker:
        '''
        Create an instance of the MetricsTracker class.

        Parameters:
        - metrics (List[str]): List of metrics to track.
        - n_classes (int): Number of classes.
        - average (str, optional): Type of averaging to perform for multiclass/multilabel targets. Default is 'macro'.
        - torchmetrics (Dict, optional): Additional torchmetrics to track. Default is an empty dictionary.

        Returns:
        - MetricsTracker: Instance of the MetricsTracker class.
        '''
        return MetricsTracker(metrics, n_classes, average=average, torchmetrics=torchmetrics)
    
    def get_TrainTestEval_instance(self, model, optimizer, loss_func, metrics_tracker, epochs = 10, lr_scheduler=None, early_stopping=None) -> TrainTestEval:
        '''
        Create an instance of the TrainTestEval class.

        Parameters:
        - model: The deep learning model.
        - optimizer: The optimizer for training the model.
        - loss_func: The loss function to use.
        - metrics_tracker: Instance of the MetricsTracker class for tracking metrics.
        - epochs (int, optional): Number of epochs to train the model. Default is 10.
        - lr_scheduler (optional): Learning rate scheduler. Default is None.
        - early_stopping (optional): Early stopping criterion. Default is None.

        Returns:
        - TrainTestEval: Instance of the TrainTestEval class.
        '''
        return TrainTestEval(model, optimizer, loss_func, metrics_tracker=metrics_tracker, epochs=epochs, lr_scheduler=lr_scheduler, early_stopping=early_stopping, device=self.device, random_seed=self.random_seed)

    def run_cross_validation(self, TrainTestEval_instance:TrainTestEval, cross_val_dataloaders:Dict) -> List:
        '''
        Run cross-validation for training and evaluation.

        Parameters:
        - TrainTestEval_instance: Instance of the TrainTestEval class.
        - cross_val_dataloaders (Dict): Dictionary of cross-validated data loaders.

        Returns:
        - List: List of training metrics per fold.
        '''
        training_metrics_per_fold = TrainTestEval_instance.cross_validation(cross_val_dataloaders)
        return training_metrics_per_fold
        
    def train_model(self, TrainTestEval_instance:TrainTestEval, train_dataloader:DataLoader, val_dataloader:DataLoader=None) -> Tuple:
        '''
        Train the model.

        Parameters:
        - TrainTestEval_instance: Instance of the TrainTestEval class.
        - train_dataloader: Data loader for training data.
        - val_dataloader (optional): Data loader for validation data. Default is None.

        Returns:
        - Tuple: Tuple containing the trained model and training results.
        '''
        start_time = timer()
        base_model_results = TrainTestEval_instance.training(train_dataloader, val_dataloader)
        end_time = timer()
        training_time = f"{(end_time-start_time):.4f}"
        print(f"Training time: {colorize(training_time, 'LIGHTRED_EX')} seconds") 
        
        return TrainTestEval_instance.model, base_model_results

    def evaluate_model(self, TrainTestEval_instance:TrainTestEval, test_dataloader:DataLoader) -> Tuple:
        '''
        Evaluate the model.

        Parameters:
        - TrainTestEval_instance: Instance of the TrainTestEval class.
        - test_dataloader: Data loader for test data.

        Returns:
        - Tuple: Tuple containing the test loss and accuracy.
        '''
        start_time = timer()
        test_loss, test_acc = TrainTestEval_instance.evaluation(test_dataloader)
        end_time = timer()
        eval_time = f"{(end_time-start_time):.4f}"
        print(f"Evaluation time: {colorize(eval_time, 'LIGHTRED_EX')} seconds")
        
        return test_loss, test_acc

    def make_inference(self, data_dir_path, DatasetClass:Dataset, transform_steps:List, dataloader_params:Dict, TrainTestEval_instance:TrainTestEval) -> List:
        '''
        Perform inference on unseen data using a trained model.

        Parameters:
            - data_dir_path (str): Path to the directory containing the inference data.
            - DatasetClass (Dataset): Class for creating datasets.
            - transform_steps (List[Transform]): List of transformation steps to be applied to the inference data.
            - dataloader_params (Dict[str, Any]): Dictionary containing parameters for creating the dataloaders.
            - TrainTestEval_instance (TrainTestEval): An instance of the TrainTestEval class initiated with a trained model. Will be used to perform inference.

        Returns: - List: Predicted classes for the inference data.
        '''
        load = LoadOurData(data_dir_path, DatasetClass, inference=True) 
        inference_dataloader = load.load_inference_data(transform_steps, dataloader_params)
        return TrainTestEval_instance.inference(inference_dataloader)



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
    dl = DeepLearningVisionWorkflow(data_dir_path, DatasetClass, device=device, random_seed=random_seed)
    # Presplit original dataset into train and test dfs 
    dl.train_test_presplit(train_ratio=train_ratio_presplit)
    
    # Load data into datasets and dataloaders
    # _______________    
    (input_shape, labels_shape), dataloaders = dl.load_data(data_loader_params, dict_transform_steps, train_ratio=train_ratio_split)
    #(input_shape, labels_shape), cv_dataloaders = dl.load_data_cv(data_loader_params, dict_transform_steps, train_ratio=train_ratio_split)
        
    # Define model 
    mri_cnn = dl.get_MRI_CNN_instance(input_shape=input_shape, 
                                              hidden_units=hidden_units,
                                              output_shape=len(dl.load.classes),
                                              activation_func=activation_func)
    
    # Print torchinfo's model summary
    summary(mri_cnn, input_size=input_shape)

    # Get metrics to track during training/validation
    metrics_tracker = dl.get_MetricsTracker_instance(metrics_to_track, len(dl.load.classes), torchmetrics=torch_metrics)


    # Initiate TrainTestEval class instance
    optimizer_inst = optimizer(params=mri_cnn.parameters(), **optimizer_params)

    train_test_eval_inst = dl.get_TrainTestEval_instance(
                                     model = mri_cnn,
                                     optimizer = optimizer_inst,
                                     loss_func = loss_func(),
                                     metrics_tracker = metrics_tracker,
                                     epochs = epochs,
                                     lr_scheduler = lr_scheduler_obj(optimizer_inst, **lr_scheduler_params),
                                     early_stopping = early_stopping
                                    )

    trained_model, results = dl.train_model(train_test_eval_inst, dataloaders['train'], dataloaders['val'])
    test_loss, test_acc = dl.evaluate_model(train_test_eval_inst, dataloaders['test'])

