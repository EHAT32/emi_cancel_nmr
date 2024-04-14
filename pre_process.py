import numpy as np
import pandas as pd
import os
from scipy.signal import decimate, butter, sosfilt
import matplotlib.pyplot as plt

path = "C:/flashka_back/data/with_amplifier/"

dt = 20e-9 #time sample is 20 ns
fs = 1 / dt #sampling frequency
points_num = 51200 #length of a signal (1.024 ms)
#params for cutoff
initial_len = 25000000
n = int(initial_len / points_num)
cutoff = n * points_num
#params for bandpassing
center_freq = 21.6e6 #Hz
lower_freq = center_freq - 0.1e6
higher_freq = center_freq + 0.1e6
filter = butter(4, [lower_freq, higher_freq], 'bp', fs=fs, output='sos')
#params for demodulation
time = np.arange(0, cutoff * dt, dt)
real_part = np.sin(2 * np.pi * time)
imaginary_part = np.cos(2 * np.pi * time)
#params for lowpass filtering
low_freq = 600e3 #Hz
low_pass_filter = butter(4, low_freq, 'lowpass', fs=fs, output='sos')
folders = os.listdir(path)
item_idx = 0

for folder in folders:
    files = os.listdir(path + folder)
    for item in files:
        df = pd.read_csv(path + folder + '/' + item)
        ch1 = df['CH1V'].to_numpy()
        ch2 = df['CH2V'].to_numpy()
        target = df['CH3V'].to_numpy()
        #cutoff so it can split evenly
        ch1 = ch1[:cutoff]
        ch2 = ch2[:cutoff]
        target = target[:cutoff]
        #apply bp filter
        filtered = sosfilt(filter, ch1)
        ch2 = sosfilt(filter, ch2)
        target = sosfilt(filter, target)
        #demodulate
        ch1_real = ch1 * real_part
        ch1_imag = ch1 * imaginary_part
        ch2_real = ch2 * real_part
        ch2_imag = ch2 * imaginary_part
        target_real = target * real_part
        target_imag = target * imaginary_part
        #combine real and imag into (2, cutoff)
        ch1 = np.vstack((ch1_real, ch1_imag))
        ch2 = np.vstack((ch2_real, ch2_imag))
        target = np.vstack((target_real, target_imag))
        #apply low pass filter
        ch1 = sosfilt(low_pass_filter, ch1, axis=1)
        ch2 = sosfilt(low_pass_filter, ch2, axis=1)
        target = sosfilt(low_pass_filter, target, axis=1)
        #decimate in three steps : 10, 10 and 4; cutoff -> 128 * 488
        ch1 = decimate(ch1, 10)
        ch1 = decimate(ch1, 10)
        ch1 = decimate(ch1, 4)
        ch2 = decimate(ch2, 10)
        ch2 = decimate(ch2, 10)
        ch2 = decimate(ch2, 4)
        target = decimate(target, 10)
        target = decimate(target, 10)
        target = decimate(target, 4)
