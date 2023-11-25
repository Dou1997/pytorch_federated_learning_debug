
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from torch.utils.data import Dataset
import os
import pandas as pd
import torchvision.transforms.functional as TF
import random
from preprocessing.baselines_dataloader import ChestXrayDataset, load_data

# Define and create your original_set here
print("Loading ChestXrays_original_dataset")
# Define the transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit the input size of the model
    transforms.ToTensor(),  # Convert images to PyTorch Tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

 # Build the CSV file relative paths
csv_file_path_train = os.path.join(script_dir, '../../data/CheXpert/archive/train.csv')
csv_file_path_test = os.path.join(script_dir, '../../data/CheXpert/archive/valid.csv')

# Build the root directory path
root_dir = os.path.join(script_dir, '../../data/')
original_set = ChestXrayDataset(csv_file=csv_file_path_train, root_dir=root_dir, transform=transform, is_rgb=True)

