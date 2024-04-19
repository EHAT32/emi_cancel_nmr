import torch
from torch.utils.data import random_split, DataLoader
from argparse import ArgumentParser
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from dataset import Dataset
from model import Model
from shield_free_model import Net1, Net2
import matplotlib.pyplot as plt
import math

#------log cosh loss-----
def log_cosh_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    def _log_cosh(x: torch.Tensor) -> torch.Tensor:
        return x + torch.nn.functional.softplus(-2. * x) - math.log(2.0)
    return torch.mean(_log_cosh(y_pred - y_true))

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        return log_cosh_loss(y_pred, y_true)


def centering(x):
    mean_real = torch.mean(x[:,0], dim = -1).unsqueeze(1)
    mean_imag = torch.mean(x[:,1], dim = -1).unsqueeze(1)
    centered = x
    centered[:,0] = centered[:,0] - mean_real
    centered[:,1] = centered[:,1] - mean_imag
    return centered
class CustomMSELoss(nn.Module):
    def __init__(self, slice_window=10):
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
    # root = "E:/nmr/emi_cancel_nmr_dataset"
    root = "E:/nmr/emi_cancel_nmr_dataset_wo_ampl"
    # root = "D:/python/nmr/NMR-denoise/shield_free/data/train"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    dataset =  Dataset(root=root, device=device)
    lengths = [0.8, 0.1, 0.1]
    train, _, validate = random_split(dataset, lengths)
    train_batch = 64
    train_loader = DataLoader(train, train_batch, shuffle=True, drop_last=True)
    validate_batch = 64
    validation_loader = DataLoader(validate, validate_batch, shuffle=True, drop_last=True)
        
    # model = Model().to(device)
    model = Net1().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)   
    criterion = CustomMSELoss(slice_window=10)
    # criterion = nn.MSELoss()
    writer = SummaryWriter()
    num_epochs = 100
    
    validation_idx = int(len(train_loader) / 4)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')
        for i, training_pair in enumerate(tqdm(train_loader)):
            model.train()
            features, target = training_pair
            optimizer.zero_grad()
            
            pred = model(features)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            writer.add_scalar(f'Loss/training', loss.item(), epoch * len(train_loader) + i)
            # Print training progress
            # if i % 100 == 0 and i > 0:
            #     print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], '
            #         f'Generator Loss: {gen_loss.item():.4f}, Discriminator Loss: {disc_loss.item():.4f}')
                
            # Validation
            if i % validation_idx == 0 and i > 0:
                plt.clf()
                model.eval()
                # Perform validation on a separate validation dataset or a subset of training data
                # Calculate validation metrics and monitor model performance
                # print('------------------------------')
                # print('Validation:')
                data_iter = iter(validation_loader)
                val_features, val_target = next(data_iter)
                with torch.no_grad():
                    val_pred = model(val_features)
                val_loss = criterion(val_pred, val_target)
                # plt.plot(val_target[0, 0].detach().cpu().numpy(), label = 'target')
                # plt.plot(val_pred[0, 0].detach().cpu().numpy(), label = 'pred')
                # plt.show(block=False)
                writer.add_scalar(f'Loss/validating', val_loss.item(), epoch * len(train_loader) + i)
                model.zero_grad()
          
        torch.save(model.state_dict(), f'./models_save/shield_free_wo_ampl/model-{epoch + 1}.pth')
         
    writer.close()
    # Save your trained model
        
    return 0
if __name__ == '__main__':
    main()
    