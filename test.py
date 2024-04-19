import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from dataset import Dataset
from torch.utils.data import random_split, DataLoader
from model import Model
import torch.nn as nn
from shield_free_model import Net1, Net2

def centering(x):
    mean_real = torch.mean(x[:,0], dim = -1).unsqueeze(1)
    mean_imag = torch.mean(x[:,1], dim = -1).unsqueeze(1)
    centered = x
    centered[:,0] = centered[:,0] - mean_real
    centered[:,1] = centered[:,1] - mean_imag
    return centered

def slicing(x, slice_window=0):
    result = x[:,:, slice_window:x.shape[-1] - slice_window]
    return result

class CustomMSELoss(nn.Module):
    def __init__(self, slice_window=0):
        super(CustomMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.slice_window = slice_window


    def forward(self, pred, target):
        # Extract the slice of the predicted values
        pred_slice = pred[..., self.slice_window : pred.shape[-1] - self.slice_window]
        
        pred_slice = centering(pred_slice)

        target_slice = target[..., self.slice_window : target.shape[-1] - self.slice_window]
        target_slice = centering(target_slice)
        # Calculate the MSE
        mse_loss = self.mse_loss(pred_slice, target_slice)

        # Return the MSE
        return mse_loss


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # model = Model().to(device=device)
    # model = Net1().to(device)
    model = Net2().to(device)
    checkpoint = torch.load("./models_save/shield_free_23/model-15.pth", map_location=torch.device(device))
    model.load_state_dict(checkpoint)
    model.eval()

    criterion = CustomMSELoss()

    root = "E:/nmr/emi_cancel_nmr_dataset"
    # root = "D:/python/nmr/NMR-denoise/shield_free/data/train"
    # root = "E:/nmr/emi_cancel_nmr_dataset_wo_ampl"

    torch.manual_seed(0)

    dataset =  Dataset(root=root, device=device)
    lengths = [0.8, 0.1, 0.1]
    _, test, _ = random_split(dataset, lengths)

    test_batch = int(len(dataset) * lengths[1])
    # test_batch = 32
    test_loader = DataLoader(test, test_batch, shuffle=True)

    data_iter = iter(test_loader)
    features, target = next(data_iter)
    pred = model(features)
    pred = slicing(pred, slice_window=10)
    pred = centering(pred)
    target = slicing(target, slice_window=10)
    target = centering(target)
    initial_sd = torch.std(target, dim = (1,2))
    loss = criterion(pred, target)
    print('Testing error: ', loss.item())
    res = target - pred
    res_sd = torch.std(res, dim = (1,2))
    initial_sd = torch.mean(initial_sd)
    res_sd = torch.mean(res_sd)
    print('Relative sd: ', res_sd.item() / initial_sd.item() * 100)

    target = target.cpu().squeeze().cpu().detach().numpy()
    # features = features.cpu().squeeze().cpu().detach().numpy()
    # ch1 = features[..., 0]
    # ch2 = features[..., 1]
    pred = pred.cpu().squeeze().cpu().detach().numpy()
    # ch1 = features[:, 0].squeeze().cpu().detach().numpy()
    # ch2 = features[:, 1].squeeze().cpu().detach().numpy()
    # plt.plot(target[1, indices])
    idx = 10
    plt.plot(target[idx,1], label = 'ямр катушка')
    # plt.plot(np.ones_like(target[idx, 1]) * np.mean(target[idx, 1]), '--r', label='mean')
    plt.plot(pred[idx,1], label = 'предсказание')
    # plt.plot(target[idx,1] - pred[idx,1], label = 'остаток')
    # plt.plot(ch1[idx,1], label = 'канал 1')
    # plt.plot(ch2[idx,1], label = 'канал 2')
    plt.legend()
    plt.show()

    return 0

if __name__ == '__main__':
    main()
    