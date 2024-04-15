import numpy as np
import pandas as pd
import os
from scipy.signal import decimate, butter, sosfilt
import torch

def apply_decimation(array):
    result = decimate(array, 10, axis=1)
    result = decimate(result, 10, axis=1)
    result = decimate(result, 4, axis=1)
    return result

def process(signal, offset):
    dt = 20e-9 #time sample is 20 ns
    fs = 1 / dt #sampling frequency
    #cutting off
    points_num = 51200 #length of a signal (1.024 ms)
    start_point = 0.000003* fs + offset
    sig = signal.copy()
    sig = sig[int(start_point):]
    sig = sig[:points_num]
    #bp filtering
    center_freq = 21.6e6 #Hz
    lower_freq = center_freq - 0.1e6
    higher_freq = center_freq + 0.1e6
    filter = butter(4, [lower_freq, higher_freq], 'bp', fs=fs, output='sos')
    sig = sosfilt(filter, sig)
    #demodulating
    time = np.arange(0, points_num * dt, dt)
    real_part = np.sin(2 * np.pi * center_freq * time)
    imaginary_part = np.cos(2 * np.pi * center_freq * time)
    sig_real = sig * real_part
    sig_imag = sig * imaginary_part
    #lowpass filtering
    low_freq = 600e3 #Hz
    low_pass_filter = butter(4, low_freq, 'lowpass', fs=fs, output='sos')
    sig_real = sosfilt(low_pass_filter, sig_real)
    sig_imag = sosfilt(low_pass_filter, sig_imag)
    sig = np.vstack((sig_real, sig_imag))
    #demodulating
    sig = apply_decimation(sig)
    return sig