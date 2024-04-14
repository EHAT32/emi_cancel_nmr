import numpy as np
import pandas as pd
import os
from scipy.signal import decimate, butter, sosfilt
import matplotlib.pyplot as plt
import torch

def apply_decimation(array):
    result = decimate(array, 10, axis=1)
    result = decimate(result, 10, axis=1)
    result = decimate(result, 4, axis=1)
    return result

path = "C:/Users/roman/YandexDisk/with_amplifier/"

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
time = np.arange(0, points_num * dt, dt)
real_part = np.sin(2 * np.pi * center_freq * time)
imaginary_part = np.cos(2 * np.pi * center_freq * time)
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
        #split into n mini samples 
        ch1 = np.vstack(np.split(ch1, n))
        ch2 = np.vstack(np.split(ch2, n))
        target = np.vstack(np.split(target, n)) #shape is 2 x 128 x 488
        #apply bp filter
        ch1 = sosfilt(filter, ch1, axis=1)
        ch2 = sosfilt(filter, ch2, axis=1)
        target = sosfilt(filter, target, axis=1)
        #demodulate
        ch1_real = ch1 * real_part
        ch1_imag = ch1 * imaginary_part
        ch2_real = ch2 * real_part
        ch2_imag = ch2 * imaginary_part
        target_real = target * real_part
        target_imag = target * imaginary_part
        #apply low pass filter
        ch1_real = sosfilt(low_pass_filter, ch1_real, axis=1)
        ch1_imag = sosfilt(low_pass_filter, ch1_imag, axis=1)
        ch2_real = sosfilt(low_pass_filter, ch2_real, axis=1)
        ch2_imag = sosfilt(low_pass_filter, ch2_imag, axis=1)
        target_real = sosfilt(low_pass_filter, target_real, axis=1)
        target_imag = sosfilt(low_pass_filter, target_imag, axis=1)
        #decimate in three steps : 10, 10 and 4; cutoff -> 128 * 488
        ch1_real = apply_decimation(ch1_real)
        ch1_imag = apply_decimation(ch1_imag)
        ch2_real = apply_decimation(ch2_real)
        ch2_imag = apply_decimation(ch2_imag)
        target_real = apply_decimation(target_real)
        target_imag = apply_decimation(target_imag)
        #save as .pth tensor in shape 2 x 3 x 128, where first is real/imag, second is ch1/ch2/target and third is signal num
        for i in range(ch1_real.shape[0]):
            data = torch.zeros((2, 3, 128))
            data[0, 0, :] = torch.from_numpy(ch1_real[i].copy())
            data[1, 0, :] = torch.from_numpy(ch1_imag[i].copy())
            data[0, 1, :] = torch.from_numpy(ch2_real[i].copy())
            data[1, 1, :] = torch.from_numpy(ch2_imag[i].copy())
            data[0, 2, :] = torch.from_numpy(target_real[i].copy())
            data[1, 2, :] = torch.from_numpy(target_imag[i].copy())