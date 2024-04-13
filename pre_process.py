import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import decimate
from scipy.fft import fft, fftshift, fftfreq
import  scipy.signal as signal

Td = 20e-9 # шаг дискретизации
fs = 1/Td # частота дискретизации

row_data = pd.read_csv('D:/without_sample0.csv')
noise_data = pd.read_csv('D:/without_sample0.csv')


# Time = row_data['Time'].values
Filtered = row_data['CH3V'].values
Trigger = row_data['CH4V'].values
Trigger_in = [i for i in range(len(Trigger)) if Trigger[i] > 2]
FID_row = Filtered[Trigger_in[-1]:]

Filtered_noise = noise_data['CH2V'].values
FID_noise = Filtered_noise[Trigger_in[-1]:]

start_point = 0.000003* fs + Trigger_in[-1]
# FID = Filtered[int(start_point):51200]
FID = Filtered[int(start_point):]
FID = FID[:51200]

FID_noise = Filtered_noise[int(start_point):]
FID_noise = FID_noise[:51200]
# plt.plot(Trigger[int(start_point):51200])
# BP filtring
center = 21.6e6
f1_ps = center - 0.1e6 # Hz
f2_ps = center + 0.1e6 # Hz
sos_FID = signal.butter(1, [f1_ps, f2_ps], 'bp', fs=1/Td, output='sos')
filtered_FID = signal.sosfilt(sos_FID, FID)

filtered_FID_noise = signal.sosfilt(sos_FID, FID_noise)

# Demodulation
operating_f = 21.6e6

FID_real = filtered_FID * np.sin(2*np.pi*operating_f*np.arange(0, len(filtered_FID)/fs, Td))
FID_imag = filtered_FID * np.cos(2*np.pi*operating_f*np.arange(0, len(filtered_FID)/fs, Td))

FID_real_noise = filtered_FID_noise * np.sin(2*np.pi*operating_f*np.arange(0, len(filtered_FID_noise)/fs, Td))
FID_imag_noise = filtered_FID_noise * np.cos(2*np.pi*operating_f*np.arange(0, len(filtered_FID_noise)/fs, Td))


# filtering LP
cutoff = 600e3
sos_real = signal.butter(4, cutoff, 'lowpass', fs=1/Td, output='sos')
filtered_real = signal.sosfilt(sos_real, FID_real)#[100:]
#
sos_imag = signal.butter(4, cutoff, 'lowpass', fs=1/Td, output='sos')
filtered_imag = signal.sosfilt(sos_imag, FID_imag)#[100:]
#noise
filtered_real_noise = signal.sosfilt(sos_real, FID_real_noise)#[100:]
filtered_imag_noise = signal.sosfilt(sos_imag, FID_imag_noise)#[100:]

# Decimation

decimated_filtered_real = decimate(FID_real, 10)#[:24576], 6)
decimated_filtered_real = decimate(decimated_filtered_real, 10)
decimated_filtered_real = decimate(decimated_filtered_real, 4)

decimated_filtered_imag = decimate(FID_imag, 10)#[:24576], 6)
decimated_filtered_imag = decimate(decimated_filtered_imag, 10)
decimated_filtered_imag = decimate(decimated_filtered_imag, 4)
#noise
decimated_filtered_real_noise = decimate(FID_real_noise, 10)#[:24576], 6)
decimated_filtered_real_noise = decimate(decimated_filtered_real_noise, 10)
decimated_filtered_real_noise = decimate(decimated_filtered_real_noise, 4)

decimated_filtered_imag_noise = decimate(FID_imag_noise, 10)#[:24576], 6)
decimated_filtered_imag_noise = decimate(decimated_filtered_imag_noise, 10)
decimated_filtered_imag_noise = decimate(decimated_filtered_imag_noise, 4)


# combine real and imaginary components
filtered_all = decimated_filtered_real + 1j * decimated_filtered_imag

filtered_all_noise = decimated_filtered_real_noise + 1j * decimated_filtered_imag_noise

# спектр
decimation_coef = 51200/128
Td_new = Td * decimation_coef
sp_row = np.abs(fftshift(fft(filtered_all)))
# sp_row = np.abs(fft(filtered_all))
freq = np.array(sorted(fftfreq(len(decimated_filtered_real), Td_new)))
freq /= 1000

sp_noise = np.abs(fftshift(fft(filtered_all_noise)))

plt.figure(1)
plt.subplot(2, 2, 1)
plt.plot(FID)
plt.subplot(2, 2, 2)
plt.plot(FID_real)
plt.plot(FID_real_noise)
plt.subplot(2, 2, 3)
plt.plot(filtered_real)
plt.plot(filtered_real_noise)
plt.subplot(2, 2, 4)
plt.plot(decimated_filtered_real)
plt.plot(decimated_filtered_real_noise)
# plt.show()

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot((filtered_all))
plt.title('FID')
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(freq, sp_row)
# plt.plot(freq, sp_noise)
plt.title('Spectrum')
plt.xlabel('Frequency, KHz')
plt.grid()
plt.show()
