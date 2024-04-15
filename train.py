import torch
from torch.utils.data import random_split, DataLoader
from argparse import ArgumentParser
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as transforms
import torchvision.datasets as dset
import cv2
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
            # if i % 100 == 0 and i > 0:
            #     model.eval()
            #     # Perform validation on a separate validation dataset or a subset of training data
            #     # Calculate validation metrics and monitor model performance
            #     print('------------------------------')
            #     print('Validation:')
            #     data_iter = iter(validation_loader)
            #     random_batch = next(data_iter).to(device)
            #     val_disc = model.training_step(random_batch, optimizer_idx=1)
            #     val_gen = model.training_step(random_batch, optimizer_idx=0)
            #     print(f'Generator validation loss: {val_gen.item():.4f}, Discriminator validation loss: {val_disc.item():.4f}')
            
            if i % 10 == 0:
                model.eval()
                
                rand_noise = torch.randn(4, 100, device=device)
                pred = model.generator(rand_noise).detach()
                # pred = postprocess(pred)
                pred = torch.permute(pred, (0, 2, 3, 1)).cpu().numpy()
                row1 = pred[0]
                # row2=pred[5]
                for i in range(3):
                    row1 = np.concatenate((row1, pred[i + 1]), axis=1)
                    # row2 = np.concatenate((row2, pred[i + 6]), axis=1)
                # grid = np.concatenate((row1, row2))
                grid = cv2.cvtColor(row1, cv2.COLOR_RGBA2BGR)
                image = cv2.resize(grid, None, fx = 4, fy = 4)
                cv2.imshow(f'v1', image)
                cv2.waitKey(1)
            
            # if epoch > 0: #pretraining dicriminator
            opt_idx += 1
            opt_idx = opt_idx % 2
        torch.save(model.generator.state_dict(), f'./models_save/final_on_full_set/generator-{epoch + 1}.pth')
        torch.save(model.discriminator.state_dict(), f'./models_save/final_on_full_set/discriminator-{epoch + 1}.pth')
         
    writer.close()
    # Save your trained model
        
    return 0
if __name__ == '__main__':
    main()
    