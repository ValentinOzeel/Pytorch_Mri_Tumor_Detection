import torch
from torch import nn 

# Setup device-agnostic device
device = "cuda" if torch.cuda.is_available() else "cpu"
