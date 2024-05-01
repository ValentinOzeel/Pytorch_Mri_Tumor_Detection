from typing import Tuple
from tqdm.auto import tqdm

import torch 
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


from secondary_module import ConfigLoad
from secondary_module import color_print

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
                 test_dataloader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_func: nn.Module,
                 epochs: int = 10,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 RANDOM_SEED: int = None
                ):
        
        self.model = model 
        self.train_dataloader = train_dataloader
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


    def testing_step(self):
        # Model in eval mode
        self.model.eval()
        # Setup test loss and accuracy 
        test_loss, test_acc = 0, 0

        # Inference mode (not to compute gradient)
        with torch.inference_mode():
            # Loop over batches
            for i, (imgs, labels) in enumerate(self.test_dataloader):
                # Set data to device
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                # Forward pass
                test_pred_logit = self.model(imgs)
                # Calculate test loss
                loss = self.loss_func(test_pred_logit, labels)
                test_loss += loss.item()
                # Calculate accuracy
                predicted_classes = test_pred_logit.argmax(dim=1)
                test_acc += ((predicted_classes==labels).sum().item()/len(predicted_classes))

        # Average metrics per batch
        test_loss = test_loss / len(self.test_dataloader)
        test_acc = test_acc / len(self.test_dataloader)
        return test_loss, test_acc

    def training(self, verbose: bool = True):

        # Empty dict to track metrics
        self.training_metrics = {"train_loss": [],
                                 "train_acc": [],
                                 "test_loss": [],
                                 "test_acc": []
                                 }

        # Loop through epochs 
        for epoch in tqdm(range(self.epochs)):
            train_loss, train_acc = self.training_step()
            test_loss, test_acc = self.testing_step()

            if verbose:
                # Print metrics for each epoch
                print(
                    color_print("\nEpoch: ", "LIGHTGREEN_EX"), epoch,
                    color_print("train_loss: ", "RED"), f"{train_loss:.4f}", color_print(" | ", "LIGHTMAGENTA_EX"),
                    color_print("train_acc: ", "RED"), f"{train_acc:.4f}", color_print(" | ", "LIGHTMAGENTA_EX"),
                    color_print("test_loss: ", "BLUE"), f"{test_loss:.4f}", color_print(" | ", "LIGHTMAGENTA_EX"),
                    color_print("test_acc: ", "BLUE"), f"{test_acc:.4f}", color_print(" | ", "LIGHTMAGENTA_EX")
                )

            # Actualize result_metrics
            self.training_metrics["train_loss"].append(train_loss), self.training_metrics["train_acc"].append(train_acc)
            self.training_metrics["test_loss"].append(test_loss), self.training_metrics["test_acc"].append(test_acc)

        return self.training_metrics