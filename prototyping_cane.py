#!/usr/bin/env python
"""
Prototyping script for visualizing CANE dataset preprocessing.

This script loads CANE EEG data, applies preprocessing via CANEDataset.load_and_preprocess_cane_raw_file,
and visualizes it using MNE's plotting capabilities.

Note: CANE preprocessing includes z-score normalization, CAR, detrending, and artifact removal,
which differs from the MDD preprocessing pipeline.
"""

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from thesis.dataset import CANE_DIR, CANEDataset
import matplotlib.pyplot as plt

example_file = CANE_DIR / "normal" / "eo" / "1060_EO.csv"


# Load preprocessed chunks for all channels
channel_names = list(CANEDataset.CHANNEL_MAPPING.values())
all_chunks = {}
for ch in channel_names:
    ch_data = CANEDataset.load_and_preprocess_cane_raw_file(example_file, ch)
    all_chunks[ch] = ch_data  # Shape: (num_chunks, 1, chunk_samples)

# Concatenate all chunks to get full continuous signal
num_chunks = all_chunks[channel_names[0]].shape[0]
chunk_samples = all_chunks[channel_names[0]].shape[2]
print(f"Total chunks available: {num_chunks}")
print(f"Samples per chunk: {chunk_samples}")
print(f"Total duration: {num_chunks * 10} seconds")

# For each channel, concatenate all chunks along the time axis
# Shape per channel: (num_chunks, 1, chunk_samples) -> (1, num_chunks * chunk_samples)
data = np.array([
    all_chunks[ch][:, 0, :].numpy().flatten() for ch in channel_names
])

print(f"Data shape for MNE: {data.shape}")  # Should be (n_channels, total_samples)

# Create MNE Raw object
info = mne.create_info(ch_names=channel_names, sfreq=CANEDataset.FS)
raw = mne.io.RawArray(data, info, verbose=False)

print(raw.info)
_ = raw.plot(scalings="auto", title=f"CANE Preprocessed: {example_file.name} (Full recording)")
plt.show()


