import torch
from torch.utils.data import Dataset
import os
from combs import combination_4
class Dataset(Dataset):
    def __init__(self, root, device, multiplier = 1.):
        self.root = root
        self.multiplier = multiplier
        if self.root[-1] != '/':
            self.root += '/'
        self.data = os.listdir(root)
        self.device = device
    #------------for my dataset    ------------------
    def __getitem__(self, index):
        data = torch.load(self.root + str(index) + '.pt', map_location=torch.device(self.device))
        features = data[:, :2, :] * self.multiplier
        features = torch.permute(features, (0, 2, 1)) #for shield free model
        target = data[:, 2, :] * self.multiplier
        return features, target
    
    #-----------for shield free dataset---------------------
    # def __getitem__(self, index):
    #     data = torch.load(self.root + str(index) + '.pth', map_location=torch.device(self.device))
    #     features = data['k-space'].squeeze()
    #     # features = features.reshape((10, 2, 128))
    #     features = features[:, :, combination_4[2]]
    #     target = data['label'].squeeze()
    #     return features, target
    
    def __len__(self):
        return len(self.data)