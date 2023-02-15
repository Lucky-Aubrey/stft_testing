# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:16:24 2023

@author: Lucky
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal  # import signal module

# Define parameters
Fs = 1000  # Sampling frequency (Hz)
T = 1/Fs  # Sampling period (s)
dur = 2  # Signal duration (s)
A = 1  # Signal amplitude
F = 50  # Signal frequency (Hz)

# Generate sinusoidal signal
t = np.arange(0, dur, T)  # Time vector
x = A * np.sin(2 * np.pi * F * t)  # Sinusoidal signal

# Apply short-time Fourier transform
window_size = int(0.02 * Fs)  # Window size for STFT (20 ms)
f, t_stft, Zxx = signal.stft(x, Fs, window='hamming', nperseg=window_size)

# Find maximum value on the spectrogram and corresponding frequency
max_val = np.max(np.abs(Zxx))
max_idy, _  = np.unravel_index(np.abs(Zxx).argmax(), np.abs(Zxx).shape)
max_freq = f[max_idy]

# Plot signal and spectrogram
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, x)
ax1.set_ylabel('Amplitude')
ax1.set_title(f'Sinusoidal signal (F={F} Hz)')
ax2.pcolormesh(t_stft, f, np.abs(Zxx), cmap='plasma')
ax2.set_ylabel('Frequency (Hz)')
ax2.set_xlabel('Time (s)')
ax2.set_title('Short-time Fourier transform')
plt.show()

print(f'The maximum value on the spectrogram is {max_val} and occurs at {max_freq} Hz.')
