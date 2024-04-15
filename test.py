import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dataset import Dataset
from torch.utils.data import random_split, DataLoader
from model import Model
import torch.nn as nn

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = Model().to(device=device)
    checkpoint = torch.load("./models_save/first/model-5.pth", map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()

    criterion = nn.MSELoss()

    root = "E:/nmr/emi_cancel_nmr_dataset"

    torch.manual_seed(0)

    dataset =  Dataset(root=root, device=device)
    lengths = [0.7, 0.15, 0.15]
    _, test, _ = random_split(dataset, lengths)

    test_batch = int(len(dataset) * lengths[1])
    test_loader = DataLoader(test, test_batch, shuffle=True)

    data_iter = iter(test_loader)
    features, target = next(data_iter)

    pred = model(features)
    loss = criterion(pred, target)
    print('Testing error: ', loss.item())


    return 0

if __name__ == '__main__':
    main()
    