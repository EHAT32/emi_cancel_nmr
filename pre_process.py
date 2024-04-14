import numpy as np
import pandas as pd
import os
from scipy.signal import decimate, butter, sosfilt
import matplotlib.pyplot as plt

path = "C:/flashka_back/data/with_amplifier/"

dt = 20e-9 #time sample is 20 ns
fs = 1 / dt #sampling frequency
points_num = 51200 #length of a signal (1.024 ms)
#params for bandpassing
center_freq = 21.6e6 #Hz
lower_freq = center_freq - 0.1e6
higher_freq = center_freq + 0.1e6
filter = butter(1, [lower_freq, higher_freq], 'bp', fs=fs, output='sos')

folders = os.listdir(path)
item_idx = 0

for folder in folders:
    files = os.listdir(path + folder)
    for item in files:
        df = pd.read_csv(path + folder + '/' + item)
        ch1 = df['CH1V'].to_numpy()
        ch2 = df['CH2V'].to_numpy()
        target = df['CH3V'].to_numpy()
        #split each array into 488 arrays
        n = int(ch1.shape[0] / points_num)
        cutoff = points_num * n
        ch1 = ch1[:cutoff]
        ch2 = ch2[:cutoff]
        target = target[:cutoff]

        ch1 = np.vstack(np.split(ch1, n))
        ch2 = np.vstack(np.split(ch2, n))
        target = np.vstack(np.split(target, n))
        #stack into (488, 51200, 3)
        data = np.dstack((ch1, ch2, target))
        plt.plot(data[0, :, 2])
        plt.show(block=False)
        plt.clf()
        #apply bp filter
        data = sosfilt(filter, data, axis=1)
        plt.plot(data[0, :, 2])
        plt.show(block=False)