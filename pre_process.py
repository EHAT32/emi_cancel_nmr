import numpy as np
import pandas as pd
import os
from scipy.signal import decimate, butter, sosfilt
import matplotlib.pyplot as plt

path = "C:/flashka_back/data/with_amplifier/"

dt = 20e-9 #time sample is 20 ns
fs = 1 / dt #sampling frequency

folders = os.listdir(path)
item_idx = 0

for folder in folders:
    files = os.listdir(path + folder)
    for item in files:
        df = pd.read_csv(path + folder + '/' + item)
        ch1 = df['CH1V'].to_numpy()
        
        