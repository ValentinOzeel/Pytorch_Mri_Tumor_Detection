import os
import numpy as np
import copy
from typing import Tuple, Dict, List
from tqdm.auto import tqdm

import torch 
from torch import nn
from torch.utils.data import DataLoader

from secondary_module import colorize, project_root_path

import matplotlib.pyplot as plt

class MRINeuralNet(nn.Module):
    '''
    Convolutionnal neural network to predict tumor in MRIs scan images.
    '''
    def __init__(self,
                 input_shape:Tuple[int],
                 hidden_units:int,
                 output_shape:int,
                 activation_func:torch.nn
                 ):
        
        super().__init__()
        
        self.input_shape = input_shape # [n_images, color_channels, height, width]
        self.hidden_units = hidden_units
        self.output_shape = output_shape # Number of classes
        self.activation_func = activation_func
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_shape[1], # Color channels
                out_channels=self.hidden_units,
                kernel_size=3, stride=1, padding=1),
            self.activation_func(),
            nn.Conv2d(
                in_channels=self.hidden_units,
                out_channels=self.hidden_units,
                kernel_size=3, stride=1, padding=1),
            self.activation_func(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_units,
                out_channels=self.hidden_units,
                kernel_size=3, stride=1, padding=1),
            self.activation_func(),
            nn.Conv2d(
                in_channels=self.hidden_units,
                out_channels=self.hidden_units,
                kernel_size=3, stride=1, padding=1),
            self.activation_func(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        
        self.all_layers_except_last = [self.conv_1, self.conv_2]
        # Calculate the number of input features for the linear layer dynamically
        self.last_layer_output_shape = self._calculate_last_layer_output_shape()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=self.hidden_units * self.last_layer_output_shape[-2] * self.last_layer_output_shape[-1],
                out_features=self.output_shape)
        )

    def _calculate_last_layer_output_shape(self):
        # Helper function to calculate the number of features after convolutional layers
        # Assuming input_shape is in the format (channels, height, width)
        dummy_input_output = torch.randn(*self.input_shape)
        with torch.inference_mode():
            # Pass dummy in all layers except for the last one
            for layer in self.all_layers_except_last:
                dummy_input_output = layer(dummy_input_output)
        return dummy_input_output.shape
        
        
    def forward(self, x):
        # Operator fusion: reducing memory overhead and improving computational efficiency
        return self.classifier(
                 self.conv_2(
                    self.conv_1(x)
                 )
               )
        
        
        
        
        
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, 
                 patience=7, delta=0, 
                 save_checkpoint=True, saving_option='model', save_dir_path=project_root_path, 
                 trace_func=print, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_checkpoint (bool): Save checkpoint when improvement is detected.
                            Default: True
            saving_option (str): Should the class save the entire model ('model'), its dict_state ('dict_state'),
            or should we save checkpoint using torch.onnx.export() ('onnx') for ONNX Format (ONNX is useful for deploying models to production environments or running inference on different platforms)
                            Default: 'model' Possibilities: 'model', 'state_dict', 'onnx'
            use_onnx (bool): 
                            Default: False
            save_dir_path (str): Path pointing to a dir for the checkpoint to be saved to.
                            Default: project_root_path
            trace_func (function): trace print function.
                            Default: print
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.delta = delta
        self.save_checkpoint = save_checkpoint
        self.saving_option = saving_option.lower()
        self.save_dir_path = save_dir_path
        self.verbose = verbose
        self.trace_func = trace_func
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
        if self.save_checkpoint and not os.path.exists(self.save_dir_path):
            raise ValueError("Incorrect save_dir_path to save checkpoint when calling early_stopping.")
        
        if self.saving_option not in ['model', 'state_dict', 'onnx']:
            raise ValueError("EarlyStopping class' saving_option parameter should be either 'model', 'dict_state' or 'onnx'.")
        
    def __call__(self, val_loss, model, input_data_onnx=None):
        """Return self.early_stop value: True or False. Potentially save checkpoints."""
        # input_data_onnx (Tensor or Tuple of Tensors): Needed for data signature when using ONNX. Default: None
                            
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(val_loss, model, input_data_onnx)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model, input_data_onnx)
            self.counter = 0
            
        return self.early_stop

    def _save_checkpoint(self, val_loss, model, input_data_onnx):
        """Saves model when validation loss decrease."""
                
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased compared to the lowest value recorded ({self.val_loss_min:.6f} --> {val_loss:.6f}).\n')
        
        self.val_loss_min = val_loss
            
        if self.save_checkpoint:
            if self.saving_option == 'onnx' and not input_data_onnx:
                raise ValueError("kwargs[0]/input_data_onnx parameter should be assigned when calling early_stopping while using saving_option = 'onnx'.")
        
            if self.saving_option == 'model':
                torch.save(model, os.path.join(self.save_dir_path, 'best_model_checkpoint.pth'))
            elif self.saving_option == 'dict_state':
                torch.save(model.state_dict(), os.path.join(self.save_dir_path, 'best_model_state_checkpoint.pth'))
            else:
                torch.onnx.export(model, input_data_onnx, os.path.join(self.save_dir_path, 'best_model_checkpoint.onnx'))
            










class MetricsTracker:
    def __init__(self, metrics:List[str], n_classes:int, average:str='macro', torchmetrics:Dict={}):
        
        self.metrics = metrics
        self.n_classes = n_classes
        self.average = average.lower()
        self.torchmetrics = torchmetrics
        
        self.all_metrics = self.metrics + list(self.torchmetrics.keys())
        
        if self.average not in ['macro', 'micro']:
            raise ValueError("Invalid average parameter. Please use 'macro' or 'micro'.")
        
        self.available_metrics = ['accuracy', 'precision', 'recall', 'f1']
        if set(self.metrics) - set(self.available_metrics):
            raise ValueError(f"Invalid 'metrics' parameter. Please only select available metrics ({self.available_metrics}) or use torchmetrics parameter.")
        
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.n_classes) if self.n_classes else 0
        self.fp = torch.zeros(self.n_classes) if self.n_classes else 0
        self.fn = torch.zeros(self.n_classes) if self.n_classes else 0
        self.total_correct = 0
        self.total_samples = 0
        
        for _, metric_obj in self.torchmetrics.items():
            metric_obj.reset()

    def update(self, predictions, labels):
        if 'accuracy' in self.metrics:
            self.total_correct += torch.sum(predictions == labels).item()
            self.total_samples += len(labels)

        if any(metric for metric in ['precision', 'recall', 'f1'] if metric in self.metrics):
            for cls in range(self.n_classes):
                self.tp[cls] += torch.sum((predictions == cls) & (labels == cls)).item()
                self.fp[cls] += torch.sum((predictions == cls) & (labels != cls)).item()
                self.fn[cls] += torch.sum((predictions != cls) & (labels == cls)).item()
        
        for _, metric_obj in self.torchmetrics.items():
            metric_obj.update(predictions, labels)
            

    def accuracy(self):
        return self.total_correct / self.total_samples

    def precision(self):
        if self.average == 'macro':
            return torch.mean(self.tp / (self.tp + self.fp + 1e-8))
        elif self.average == 'micro':
            return torch.sum(self.tp) / (torch.sum(self.tp) + torch.sum(self.fp) + 1e-8)

    def recall(self):
        if self.average == 'macro':
            return torch.mean(self.tp / (self.tp + self.fn + 1e-8))
        elif self.average == 'micro':
            return torch.sum(self.tp) / (torch.sum(self.tp) + torch.sum(self.fn) + 1e-8)

    def f1(self):
        prec = self.precision()
        rec = self.recall()
        return 2 * (prec * rec) / (prec + rec + 1e-8)

    def compute_metrics(self):
        # Add metrics
        metrics_dict = {metric:getattr(self, metric)() for metric in self.metrics}
        # Add torchmetrics
        for metric_name, metric_obj in self.torchmetrics.items():
            metrics_dict[metric_name] = metric_obj

        return metrics_dict




        
class TrainTestEval():
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_func: nn.Module,
                 metrics_tracker: MetricsTracker,
                 epochs: int = 10,
                 lr_scheduler: torch.optim.lr_scheduler=None,
                 early_stopping:EarlyStopping=None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 RANDOM_SEED: int = None
                ):
        
        self.model = model 
        self.optimizer = optimizer
        self.loss_func = loss_func 
        self.train_metrics_tracker = copy.deepcopy(metrics_tracker)
        self.val_metrics_tracker = copy.deepcopy(metrics_tracker)
        self.epochs = epochs 
        self.lr_scheduler = lr_scheduler
        self.early_stopping = early_stopping
        self.device = device 
        
        
        self.curve_metrics = copy.deepcopy(self.train_metrics_tracker.metrics)
        self.curve_metrics.insert(0, 'loss')
        
        self.torchmetrics = copy.deepcopy(list(self.train_metrics_tracker.torchmetrics.keys()))
        
        self.all_metrics = self.curve_metrics + self.torchmetrics
        
        # Put model on device
        self.model.to(self.device)
        
        if RANDOM_SEED is not None:
            torch.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed(RANDOM_SEED)
        
    def get_dummy_input(self, dataloader:DataLoader):
        imgs, labels = next(iter(dataloader))
        return imgs.to(self.device)
        
    def training_step(self, train_dataloader:DataLoader):
        # Activate training mode
        self.model.train()
        # Setup training loss and accuracy
        train_loss = 0

        # Loop over dataloader batches
        for i, (imgs, labels) in enumerate(train_dataloader):
            # Data to device
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            # Forward pass
            train_pred_logit = self.model(imgs)
            # Calculate loss and add it to train_loss
            loss = self.loss_func(train_pred_logit, labels)
            train_loss += loss.item()
            # Optimizer zero grad
            self.optimizer.zero_grad()
            # Loss backpropagation
            loss.backward()
            # Optimizer step
            self.optimizer.step()
            # Predictions
            predicted_classes = torch.argmax(torch.softmax(train_pred_logit, dim=1), dim=1)
            # Update metrics
            self.train_metrics_tracker.update(predicted_classes, labels)

        # Average loss per batch
        train_loss = train_loss / len(train_dataloader)
        return train_loss


    def validation_step(self, valid_dataloader:DataLoader):
        # Model in eval mode
        self.model.eval()
        # Setup valid loss and accuracy 
        val_loss = 0

        # Inference mode (not to compute gradient)
        with torch.inference_mode():
            # Loop over batches
            for i, (imgs, labels) in enumerate(valid_dataloader):
                # Set data to device
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                # Forward pass
                val_pred_logit = self.model(imgs)
                # Calculate valid loss
                loss = self.loss_func(val_pred_logit, labels)
                val_loss += loss.item()
                # Predictions
                predicted_classes = val_pred_logit.argmax(dim=1)
                # Update metrics
                self.val_metrics_tracker.update(predicted_classes, labels)
                
        # Average loss per batch
        val_loss = val_loss / len(valid_dataloader)
        return val_loss
    
    def _schedule_lr(self, metric):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(metric)
        else:
            self.lr_scheduler.step()
        #print('Lr value: ', self.optimizer.param_groups[0]['lr'])

    def _organize_metrics_dict(self, gathered_metrics, train_metrics=None, val_metrics=None):
        for metric in self.all_metrics:
            gathered_metrics['train'][metric].append(train_metrics[metric])
            gathered_metrics['val'][metric].append(val_metrics[metric])
            
        return gathered_metrics
            
        
    def training(self, train_dataloader:DataLoader, valid_dataloader:DataLoader, verbose: bool = True, real_time_plot_metrics:bool = True, save_metric_plot = True):
        # Empty dict to track metrics
        gathered_metrics = {'train':{}, 'val':{}}
        for state in gathered_metrics.keys():
            for metric_name in self.all_metrics:
                gathered_metrics[state][metric_name] = []

        # Initialize plot
        if real_time_plot_metrics:
            plt.figure(figsize=(len(self.curve_metrics)*4, 8))
            plt.ion()  # Turn on interactive mode for dynamic plotting
            
        # Loop through epochs 
        for epoch in tqdm(range(self.epochs)):
            # Reset metrics
            self.train_metrics_tracker.reset()
            self.val_metrics_tracker.reset()
            # Train and validation steps
            train_loss = self.training_step(train_dataloader)
            val_loss = self.validation_step(valid_dataloader)
            # Actualize gathered_metrics
            train_metrics = self.train_metrics_tracker.compute_metrics()
            train_metrics['loss'] = train_loss
            val_metrics = self.val_metrics_tracker.compute_metrics()
            val_metrics['loss'] = val_loss
            # Keep track of computed metrics
            gathered_metrics = self._organize_metrics_dict(gathered_metrics, train_metrics=train_metrics, val_metrics=val_metrics)
            # Print training/validation info
            if verbose:
                print_train_metrics, print_val_metrics = '', ''

                for metric_name in self.curve_metrics:
                    print_train_metrics = ''.join([print_train_metrics, colorize(''.join([metric_name, ': ']), "RED"), f"{train_metrics[metric_name]:.4f}", colorize(" | ", "LIGHTMAGENTA_EX")])
       
                for metric_name in self.curve_metrics:
                    print_val_metrics = ''.join([print_val_metrics, colorize(''.join([metric_name, ': ']), "BLUE"), f"{val_metrics[metric_name]:.4f}", colorize(" | ", "LIGHTMAGENTA_EX")])
                # Print metrics at each epoch
                print(
                    colorize("\nEpoch: ", "LIGHTGREEN_EX"), epoch,
                    '\n-- Train metrics --', print_train_metrics,   
                    '\n--  Val metrics  --', print_val_metrics       
                )
                
            # Plot the metrics curves
            if real_time_plot_metrics:
                self.plot_metrics(gathered_metrics)
            # Adjust learning rate
            if self.lr_scheduler:
                self._schedule_lr(val_loss)
            # Check for early_stopping
            if self.early_stopping:
                if self.early_stopping(val_loss, self.model, self.get_dummy_input(train_dataloader)):
                    break

        if save_metric_plot:
            plt.tight_layout()
            plt.ioff()  # Turn off interactive mode
            fig = plt.gcf()  # Get the current figure
            fig.savefig('training_metrics.png')
            plt.clf
            
            self.save_plot_torchmetrics(gathered_metrics)

            
        return self.model, gathered_metrics


        
        
        
    def plot_metrics(self, gathered_metrics:Dict):  
        train_metrics, val_metrics = gathered_metrics['train'], gathered_metrics['val']
        for i, metric_name in enumerate(self.curve_metrics):
            plt.subplot(1, len(self.curve_metrics), i+1)
            plt.plot(range(len(train_metrics[metric_name])), train_metrics[metric_name], label=''.join(['train_', metric_name]), color='red')
            plt.plot(range(len(val_metrics[metric_name])), val_metrics[metric_name], label=''.join(['val_', metric_name]), color='blue')
            if not plt.gca().get_title(): 
                plt.title(f"train_{metric_name} VS val_{metric_name}")
                plt.xlabel('Epochs')
                plt.ylabel(metric_name)
                plt.legend()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)
        
    def save_plot_torchmetrics(self, gathered_metrics:Dict):
        for metric in self.torchmetrics:
            fig = plt.figure(figsize=(10, 6), layout="constrained")
            ax1 = plt.subplot(1, 2, 1)
            ax2 = plt.subplot(1, 2, 2)
            gathered_metrics['train'][metric][-1].plot(ax=ax1)
            ax1.set_title(f"train_{metric}")
            gathered_metrics['val'][metric][-1].plot(ax=ax2)
            ax2.set_title(f"val_{metric}")
            fig.savefig(f'{metric}.png')
            plt.clf
        
    def cross_validation(self, cross_valid_dataloaders:Dict):
        training_metrics_per_fold = []
        for fold, (train_dataloader, valid_dataloader) in enumerate(zip(cross_valid_dataloaders['train'], cross_valid_dataloaders['valid'])):
            training_metrics_per_fold.append(self.training(train_dataloader, valid_dataloader, save_metric_plot=False))
        return training_metrics_per_fold
        
    def evaluation(self, test_dataloader:DataLoader):
        # Model in eval mode
        self.model.eval()
        # Setup test loss and accuracy 
        test_loss, test_acc = 0, 0

        # Inference mode (not to compute gradient)
        with torch.inference_mode():
            # Loop over batches
            for i, (imgs, labels) in enumerate(test_dataloader):
                # Set data to device
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                # Forward pass
                test_pred_logit = self.model(imgs)
                # Calculate valid loss
                loss = self.loss_func(test_pred_logit, labels)
                test_loss += loss.item()
                # Calculate accuracy
                predicted_classes = test_pred_logit.argmax(dim=1)
                test_acc += ((predicted_classes==labels).sum().item()/len(predicted_classes))

        # Average metrics per batch
        test_loss = test_loss / len(test_dataloader)
        test_acc = test_acc / len(test_dataloader)
        
        print(
            colorize("\nModel evaluation: ", "LIGHTGREEN_EX"),
            colorize("test_loss: ", "RED"), f"{test_loss:.4f}", colorize(" | ", "LIGHTMAGENTA_EX"),
            colorize("test_acc: ", "RED"), f"{test_acc:.4f}", colorize(" | ", "LIGHTMAGENTA_EX"),
        )
                
        return test_loss, test_acc


    def inference(self, dataloader:DataLoader):
        # Model in eval mode
        self.model.eval()

        pred_classes = []
        # Inference mode (not to compute gradient)
        with torch.inference_mode():
            # Loop over batches
            for i, (imgs, _) in enumerate(dataloader):
                # Set data to device
                imgs = imgs.to(self.device)
                # Forward pass
                pred_logit = self.model(imgs)
                # Get predicted classes
                predicted_classes = pred_logit.argmax(dim=1)
                # Extend predictions lists
                pred_classes.extend(predicted_classes)
   
        return pred_classes