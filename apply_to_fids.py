import torch
from model import Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from processing_script import process


data = pd.read_csv('G:/without_sample0.csv')

trigger = data['CH4V'].to_numpy()
offset = [i for i in range(len(trigger)) if trigger[i] > 2]
offset = offset[-1]

nmr_signal = data['CH3V'].to_numpy()

nmr_signal = process(nmr_signal, offset)
ch1 = process(data['CH1V'].to_numpy(), offset)
ch2 = process(data['CH2V'].to_numpy(), offset)


model_path = "./models_save/first/model-5.pth"
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
features = torch.zeros((1, 2, 2, 128))

features[0, :, 0] = torch.from_numpy(ch1.copy())
features[0, :, 1] = torch.from_numpy(ch2.copy())
features = features.to(device=device)
model = Model().to(device=device)
checkpoint = torch.load(model_path, map_location=torch.device(device))
model.load_state_dict(checkpoint)
model.eval()

pred = model(features).squeeze().cpu().detach().numpy()

plt.plot(nmr_signal[0])
plt.plot(pred[0])
plt.show()