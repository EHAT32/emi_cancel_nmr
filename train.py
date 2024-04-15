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


def main():
    root = "E:/nmr/emi_cancel_nmr_dataset"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    dataset =  Dataset(root=root, device=device)
    lengths = [0.7, 0.15, 0.15]
    train, _, validate = random_split(dataset, lengths)
    train_batch = 32
    train_loader = DataLoader(train, train_batch, shuffle=True, num_workers=4, drop_last=True)
    validate_batch = 32
    validation_loader = DataLoader(validate, validate_batch, shuffle=True, num_workers=4, drop_last=True)
        
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)   
    criterion = nn.MSELoss()

    writer = SummaryWriter()
    num_epochs = 100
    
    
    
    for epoch in range(num_epochs):
        for i, training_pair in enumerate(tqdm(train_loader)):
            model.train()
            features, target = training_pair
            optimizer.zero_grad()
            
            pred = model(features)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)
            # Print training progress
            # if i % 100 == 0 and i > 0:
            #     print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_loader)}], '
            #         f'Generator Loss: {gen_loss.item():.4f}, Discriminator Loss: {disc_loss.item():.4f}')
                
            # Validation
            if i % 100 == 0 and i > 0:
                model.eval()
                # Perform validation on a separate validation dataset or a subset of training data
                # Calculate validation metrics and monitor model performance
                print('------------------------------')
                print('Validation:')
                data_iter = iter(validation_loader)
                val_features, val_target = next(data_iter)
                val_pred = model(val_features)
                val_loss = criterion(val_pred, val_target)
                writer.add_scalar('Validating Loss', val_loss.item(), epoch * len(train_loader) + i)
            
        torch.save(model.generator.state_dict(), f'./models_save/final_on_full_set/generator-{epoch + 1}.pth')
        torch.save(model.discriminator.state_dict(), f'./models_save/final_on_full_set/discriminator-{epoch + 1}.pth')
         
    writer.close()
    # Save your trained model
        
    return 0
if __name__ == '__main__':
    main()
    