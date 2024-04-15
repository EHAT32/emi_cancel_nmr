import torch
from torch.utils.data import Dataset
import os

class Dataset(Dataset):
    def __init__(self, root, device):
        self.root = root
        if self.root[-1] != '/':
            self.root += '/'
        self.data = os.listdir(root)
        self.device = device
        
    def __getitem__(self, index):
        data = torch.load(self.root + str(index) + '.pt', map_location=torch.device(self.device))
        features = data[:, :2, :]
        target = data[:, 2, :]
        return features, target
    
    def __len__(self):
        return len(self.data)