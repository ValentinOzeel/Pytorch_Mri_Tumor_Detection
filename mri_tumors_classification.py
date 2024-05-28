import os
import torch

from torch.optim import lr_scheduler
from torchinfo import summary

from pytorch_vision_workflow.secondary_module import ConfigLoad, check_cuda_availability
from pytorch_vision_workflow.model import EarlyStopping
from pytorch_vision_workflow.workflow_class import DeepLearningVisionWorkflow


if __name__ == "__main__":
    # Get data path
    # _______________
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    data_dir_path = os.path.join(project_root, 'data')
    config_path = os.path.join(project_root, 'conf', 'config.yml')
    
    # Setup device-agnostic device
    check_cuda_availability()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    ###########                 ###########
    ########### GET CONFIG DATA ###########
    ###########                 ###########
    config_load = ConfigLoad(path=config_path)
    config = config_load.get_config()
    
    random_seed = config['RANDOM_SEED']
    # Dataset class used
    DatasetClass = config_load.get_dataset()
    # test_transform used for test_dataset (because we want to predict on real life data (only resized and transformed asd tensor))
    train_transform_steps, val_transform_steps, test_transform_steps = config_load.get_transform_steps(dataset_type='train'), config_load.get_transform_steps(dataset_type='val'), config_load.get_transform_steps(dataset_type='test')  
 #   train_transform_steps, test_transform_steps = config_load.get_transform_steps(dataset_type='train'), config_load.get_transform_steps(dataset_type='test')    
    # Create dict_transform_steps
    dict_transform_steps = {'train':train_transform_steps, 'val':val_transform_steps, 'test':test_transform_steps}
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
        early_stopping = EarlyStopping(project_root=project_root, **config['MODEL_PARAMS']['early_stopping'])



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

    # Train and evaluate
    trained_model, results = dl.train_model(train_test_eval_inst, dataloaders['train'], dataloaders['val'])
    test_loss, test_acc = dl.evaluate_model(train_test_eval_inst, dataloaders['test'])

