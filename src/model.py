from typing import Tuple, Dict
from tqdm.auto import tqdm

import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from secondary_module import ConfigLoad
from secondary_module import color_print

import matplotlib.pyplot as plt

class MRINeuralNet(nn.Module):
    '''
    Convolutionnal neural network to predict tumor in MRIs scan images.
    '''
    def __init__(self,
                 input_shape:Tuple[int],
                 hidden_units:int,
                 output_shape:int,
                 ):
        
        super().__init__()
        
        self.input_shape = input_shape # [n_images, color_channels, height, width]
        self.hidden_units = hidden_units
        self.output_shape = output_shape # Number of classes
            
        
        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_shape[1], # Color channels
                out_channels=self.hidden_units,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.hidden_units,
                out_channels=self.hidden_units,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_units,
                out_channels=self.hidden_units,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.hidden_units,
                out_channels=self.hidden_units,
                kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
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
        with torch.no_grad():
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
#class EarlyStopping():
#    def __init__(self, patience, ):
            
        
class TrainTestEval():
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss_func: nn.Module,
                 epochs: int = 10,
                 lr_scheduler: torch.optim.lr_scheduler=None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 RANDOM_SEED: int = None
                ):
        
        self.model = model 
        self.optimizer = optimizer
        self.loss_func = loss_func 
        self.epochs = epochs 
        self.lr_scheduler = lr_scheduler
        self.device = device 
        
        # Put model on device
        self.model.to(self.device)
        
        if RANDOM_SEED is not None:
            torch.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed(RANDOM_SEED)
        

    def training_step(self, train_dataloader:DataLoader):
        # Activate training mode
        self.model.train()
        # Setup training loss and accuracy
        train_loss, train_acc = 0, 0

        # Loop over dataloader batches
        for i, (imgs, labels) in enumerate(train_dataloader):
            # Data to device
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            # Forward pass
            predictions = self.model(imgs)
            # Calculate loss and add it to train_loss
            loss = self.loss_func(predictions, labels)
            train_loss += loss.item()
            # Optimizer zero grad
            self.optimizer.zero_grad()
            # Loss backpropagation
            loss.backward()
            # Optimizer step
            self.optimizer.step()
            # Calculate accuracy
            predicted_classes = torch.argmax(torch.softmax(predictions, dim=1), dim=1)
            train_acc += (predicted_classes==labels).sum().item()/len(predictions)

        # Average metrics per batch
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)
        return train_loss, train_acc


    def validation_step(self, valid_dataloader:DataLoader):
        # Model in eval mode
        self.model.eval()
        # Setup valid loss and accuracy 
        val_loss, val_acc = 0, 0

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
                # Calculate accuracy
                predicted_classes = val_pred_logit.argmax(dim=1)
                val_acc += ((predicted_classes==labels).sum().item()/len(predicted_classes))

        # Average metrics per batch
        val_loss = val_loss / len(valid_dataloader)
        val_acc = val_acc / len(valid_dataloader)
        return val_loss, val_acc
    
    def _schedule_lr(self, metric):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.lr_scheduler.step(metric)
        else:
            self.lr_scheduler.step()
        print('\nLr value: ', self.optimizer.param_groups[0]['lr'])

    def training(self, train_dataloader:DataLoader, valid_dataloader:DataLoader, verbose: bool = True, plot_metrics:bool = True):
        # Empty dict to track metrics
        training_metrics = {"train_loss": [],
                            "train_acc": [],
                            "val_loss": [],
                            "val_acc": []
                            }

        # Initialize plot
        if plot_metrics:
            plt.figure(figsize=(16, 8))
            plt.ion()  # Turn on interactive mode for dynamic plotting
            
        # Loop through epochs 
        for epoch in tqdm(range(self.epochs)):
            train_loss, train_acc = self.training_step(train_dataloader)
            val_loss, val_acc = self.validation_step(valid_dataloader)

            # Actualize result_metrics
            training_metrics["train_loss"].append(train_loss), training_metrics["train_acc"].append(train_acc)
            training_metrics["val_loss"].append(val_loss), training_metrics["val_acc"].append(val_acc)
            
            if verbose:
                # Print metrics for each epoch
                print(
                    color_print("\nEpoch: ", "LIGHTGREEN_EX"), epoch,
                    color_print("train_loss: ", "RED"), f"{train_loss:.4f}", color_print(" | ", "LIGHTMAGENTA_EX"),
                    color_print("train_acc: ", "RED"), f"{train_acc:.4f}", color_print(" | ", "LIGHTMAGENTA_EX"),
                    color_print("val_loss: ", "BLUE"), f"{val_loss:.4f}", color_print(" | ", "LIGHTMAGENTA_EX"),
                    color_print("val_acc: ", "BLUE"), f"{val_acc:.4f}", color_print(" | ", "LIGHTMAGENTA_EX")
                )


            if self.lr_scheduler:
                self._schedule_lr(val_loss)
                
                
            # Plot the metrics curves
            if plot_metrics:
                self.plot_metrics(training_metrics)
            
        if plot_metrics:
            plt.tight_layout()
            #plt.draw()
            #plt.pause(0.1)
            plt.ioff()  # Turn off interactive mode
            fig = plt.gcf()  # Get the current figure
            fig.savefig('training_metrics.png')

        return training_metrics

    def cross_validation(self, cross_valid_dataloaders:Dict):
        training_metrics_per_fold = []
        for fold, (train_dataloader, valid_dataloader) in enumerate(zip(cross_valid_dataloaders['train'], cross_valid_dataloaders['valid'])):
            training_metrics_per_fold.append(self.training(train_dataloader, valid_dataloader))
        return training_metrics_per_fold
        
        
        
    def plot_metrics(self, training_metrics:Dict):       
        plt.subplot(1, 2, 1)
        plt.plot(range(len(training_metrics["train_loss"])), training_metrics["train_loss"], label='train_loss', color='red')
        plt.plot(range(len(training_metrics["val_loss"])), training_metrics["val_loss"], label='val_loss', color='blue')
        if not plt.gca().get_title(): 
            plt.title("train_loss VS val_loss")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(len(training_metrics["train_acc"])), training_metrics["train_acc"], label='train_acc', color='red')
        plt.plot(range(len(training_metrics["val_acc"])), training_metrics["val_acc"], label='val_acc', color='blue')
        if not plt.gca().get_title(): 
            plt.title("train_acc VS val_acc")
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)
        
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
            color_print("\nModel evaluation: ", "LIGHTGREEN_EX"),
            color_print("test_loss: ", "RED"), f"{test_loss:.4f}", color_print(" | ", "LIGHTMAGENTA_EX"),
            color_print("test_acc: ", "RED"), f"{test_acc:.4f}", color_print(" | ", "LIGHTMAGENTA_EX"),
        )
                
        return test_loss, test_acc


    def inference(self, dataloader:DataLoader):
        # Model in eval mode
        self.model.eval()

        pred_logits, pred_classes = [], []
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
                pred_logits.extend(pred_logit)
                pred_classes.extend(predicted_classes)
   
        return pred_logits, pred_classes