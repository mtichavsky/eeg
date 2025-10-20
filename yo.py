import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.signal import spectrogram, get_window
from sklearn.metrics import confusion_matrix


from scipy.signal import ShortTimeFFT

import librosa

sampling_rate=256
#%%
print("hello!")

# Typical EEG sampling rates are 128 Hz – 512 Hz, so target_sr=256 made sense for EEG data.
def load_audio(filename, target_sr=sampling_rate):
    # MP3 or other formats; convert to mono, resample
    x, sr = librosa.load(filename, sr=target_sr, mono=True)
    return x, sr

x, sr = load_audio("get-lucky.mp3")

print(x.shape)

# Then chunk it into 10 sec intervals, section III.A
CHUNK_DURATION=10*sampling_rate
win = get_window("hamming", CHUNK_DURATION)
stft = ShortTimeFFT(win=win, hop=int(CHUNK_DURATION/2), fs=sr)
Zxx = stft.stft(x)
print(Zxx.shape)

# chatgpt plot
spec = np.abs(Zxx)**2
spec_db = 10 * np.log10(spec + 1e-12)  # convert to dB

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.pcolormesh(stft.t, stft.f, spec_db, shading="auto")
plt.title("Power Spectrogram (STFT, Hamming window)")
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.colorbar(label="Power [dB]")
plt.tight_layout()
plt.show()

