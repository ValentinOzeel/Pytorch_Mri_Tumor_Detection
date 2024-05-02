from typing import Tuple
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
                 RANDOM_SEED: int = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"
                 ):
        
        super().__init__()
        
        # Set-up device agnostic model
        self.to(device)
        
        self.input_shape = input_shape # [n_images, color_channels, height, width]
        self.hidden_units = hidden_units
        self.output_shape = output_shape # Number of classes
        
        if RANDOM_SEED is not None:
            torch.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed(RANDOM_SEED)
        
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
           
        
class TrainTestEval():
    def __init__(self,
                 model: nn.Module,
                 train_dataloader: DataLoader,
                 valid_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_func: nn.Module,
                 epochs: int = 10,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 RANDOM_SEED: int = None
                ):
        
        self.model = model 
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_func = loss_func 
        self.epochs = epochs 
        self.device = device 
        
        if RANDOM_SEED is not None:
            torch.manual_seed(RANDOM_SEED)
            torch.cuda.manual_seed(RANDOM_SEED)
        

    def training_step(self):
        # Activate training mode
        self.model.train()
        # Setup training loss and accuracy
        train_loss, train_acc = 0, 0

        # Loop over dataloader batches
        for i, (imgs, labels) in enumerate(self.train_dataloader):
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
        train_loss = train_loss / len(self.train_dataloader)
        train_acc = train_acc / len(self.train_dataloader)
        return train_loss, train_acc


    def validation_step(self):
        # Model in eval mode
        self.model.eval()
        # Setup valid loss and accuracy 
        valid_loss, valid_acc = 0, 0

        # Inference mode (not to compute gradient)
        with torch.inference_mode():
            # Loop over batches
            for i, (imgs, labels) in enumerate(self.valid_dataloader):
                # Set data to device
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                # Forward pass
                valid_pred_logit = self.model(imgs)
                # Calculate valid loss
                loss = self.loss_func(valid_pred_logit, labels)
                valid_loss += loss.item()
                # Calculate accuracy
                predicted_classes = valid_pred_logit.argmax(dim=1)
                valid_acc += ((predicted_classes==labels).sum().item()/len(predicted_classes))

        # Average metrics per batch
        valid_loss = valid_loss / len(self.valid_dataloader)
        valid_acc = valid_acc / len(self.valid_dataloader)
        return valid_loss, valid_acc

    def training(self, verbose: bool = True, plot_metrics:bool = True):
        # Empty dict to track metrics
        self.training_metrics = {"train_loss": [],
                                 "train_acc": [],
                                 "valid_loss": [],
                                 "valid_acc": []
                                 }

        # Initialize plot
        if plot_metrics:
            plt.figure(figsize=(16, 8))
            plt.ion()  # Turn on interactive mode for dynamic plotting
            
        # Loop through epochs 
        for epoch in tqdm(range(self.epochs)):
            train_loss, train_acc = self.training_step()
            valid_loss, valid_acc = self.validation_step()

            # Actualize result_metrics
            self.training_metrics["train_loss"].append(train_loss), self.training_metrics["train_acc"].append(train_acc)
            self.training_metrics["valid_loss"].append(valid_loss), self.training_metrics["valid_acc"].append(valid_acc)
            
            if verbose:
                # Print metrics for each epoch
                print(
                    color_print("\nEpoch: ", "LIGHTGREEN_EX"), epoch,
                    color_print("train_loss: ", "RED"), f"{train_loss:.4f}", color_print(" | ", "LIGHTMAGENTA_EX"),
                    color_print("train_acc: ", "RED"), f"{train_acc:.4f}", color_print(" | ", "LIGHTMAGENTA_EX"),
                    color_print("valid_loss: ", "BLUE"), f"{valid_loss:.4f}", color_print(" | ", "LIGHTMAGENTA_EX"),
                    color_print("valid_acc: ", "BLUE"), f"{valid_acc:.4f}", color_print(" | ", "LIGHTMAGENTA_EX")
                )

            # Plot the metrics curves
            if plot_metrics:
                self.plot_metrics()
            
        if plot_metrics:
            plt.tight_layout()
            #plt.draw()
            #plt.pause(0.1)
            plt.ioff()  # Turn off interactive mode
            fig = plt.gcf()  # Get the current figure
            fig.savefig('training_metrics.png')

        return self.training_metrics

    
    def plot_metrics(self):       
        plt.subplot(1, 2, 1)
        plt.plot(range(len(self.training_metrics["train_loss"])), self.training_metrics["train_loss"], label='train_loss', color='red')
        plt.plot(range(len(self.training_metrics["valid_loss"])), self.training_metrics["valid_loss"], label='valid_loss', color='blue')
        if not plt.gca().get_title(): 
            plt.title("train_loss VS valid_loss")
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(len(self.training_metrics["train_acc"])), self.training_metrics["train_acc"], label='train_acc', color='red')
        plt.plot(range(len(self.training_metrics["valid_acc"])), self.training_metrics["valid_acc"], label='valid_acc', color='blue')
        if not plt.gca().get_title(): 
            plt.title("train_acc VS valid_acc")
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)
        
    def evaluate_on_unseen_data(self):
        return
