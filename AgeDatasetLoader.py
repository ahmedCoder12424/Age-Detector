import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F


class AgeDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Assuming 'pixels' column contains space-separated pixel values as a string
        pixel_str = self.data.iloc[idx]["pixels"]
        pixels = np.array(pixel_str.split(), dtype=np.float32)  # Convert to NumPy array
        pixels = torch.tensor(pixels).view(1, 48, 48)  # Reshape (assuming 48x48 image) #change if colored)
        
        # Assuming 'age' column contains the label

        self.age_ranges = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29",
                   "30-34", "35-39", "40-44", "45-49", "50-54", "55-59",
                   "60-64", "65-69", "70-74", "75-79", "80-84", "85-89",
                   "90-94", "95-99", "100-105", "105-110", "110-115", "115-120"]

        # Create a dictionary to map each age range to an index
        self.age_to_index = {age: i for i, age in enumerate(self.age_ranges)}
        age_range = self.data.iloc[idx]["label"]  # Assume the column already contains age ranges
        age_index = self.age_to_index[age_range]  # Get the index
        age_onehot = F.one_hot(torch.tensor(age_index), num_classes=len(self.age_ranges)).float()
        
        return pixels, age_onehot