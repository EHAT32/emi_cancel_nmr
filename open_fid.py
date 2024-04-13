import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, decimate

data = []
with open("C:/flashka_back/spectrometer/correct_channels appr 1 ms (water 0.4 ms)/05.04.2024_spinecho_160806_c.dat", 'r') as f:
    d = f.read()
    rows = d.split('\n')

    spec_time = np.zeros(len(rows))
    spec_data = np.zeros(len(rows), dtype=np.complex128)
    i = 0
    for row in rows:
        if len(row) < 1:
            continue
        row = row.split(' ')
        row = [c for c in row if c != '']
        row = [float(num) for num in row]
        spec_time[i] = row[0] / 1e6
        spec_data[i] = row[1] + 1j*row[2]
        i += 1

print(len(rows) - len(spec_data))
plt.figure(0)
spec_time = spec_time[np.where(spec_time > 0.0015)]
spec_data = spec_data[np.where(spec_time > 0.0015)]
plt.plot(spec_time * 1000, spec_data.real)
plt.plot(spec_time * 1000, spec_data.imag)
plt.xlabel('Время, мс')


path = "C:/flashka_back/experiment/echo appr 1 ms (water appr 0.4 ms)/loop39.csv"

df = pd.read_csv(path)

fid = df['CH3V'].to_numpy()
channel2 = df['CH2V'].to_numpy()
channel1=df['CH1V'].to_numpy()
trigger = df['CH4V'].to_numpy()
dt = 2*1e-8
time = np.linspace(0, dt * len(fid), len(fid))
slice = 0.2
indices = np.where(trigger < 0.5)[0]
offset = 0
plt.figure(1)
plt.plot(time[indices[offset:]], fid[indices[offset:]])
# plt.plot(time[indices[offset:]], trigger[indices[offset:]])
# fid = fid[indices[offset:]]

analytical = hilbert(fid)

envelope = np.abs(analytical)
plt.figure(2)
plt.plot(time[indices[offset:]], fid[indices[offset:]], label='сигнал')
plt.plot(time[indices[offset:]], envelope[indices[offset:]], label = 'огибающая')
#oscilloscope
freq = np.fft.rfftfreq(len(time[indices[offset:]]), np.diff(time)[0])
F_spec = np.fft.rfft(fid[indices[offset:]])
#spectrometer
spec_freq = np.fft.fftfreq(len(spec_time), np.diff(spec_time)[0])
F_spec_data = np.fft.fft(spec_data)
res = np.where(F_spec < 18.2, F_spec, 0 * F_spec)
plt.figure(3)

half_height = np.ones_like(freq) * 0.5
half_height_2 = np.ones_like(spec_freq) * 0.705/2
plt.plot((freq - 21.6*1e6) / 1e3, F_spec / np.max(res), label='Осциллограф')
shift = 31

plt.plot(spec_freq / 1e3, F_spec_data.real / np.max(np.abs(F_spec_data)), label='Спектрометр, вещ. часть')
plt.plot(spec_freq / 1e3, F_spec_data.imag / np.max(np.abs(F_spec_data)), label='Спектрометр, мним. часть')
plt.ylabel('Амплитуда')
plt.xlabel('Частота, кГц')
plt.legend()
plt.xlim((-200, 200))

plt.show()