### MDD/CANE/SAD Preprocessing

Each EDF file undergoes the following preprocessing (see `thesis/dataset.py`):

1. Load EDF file
2. Bandpass filter: 1-70 Hz (IIR)
3. Notch filter: 50 Hz (remove power line noise)
4. Channel selection:
    - 8 channels (when `--channel all`)
    - Specific single channel (e.g., `--channel Fp1`)
    - Synthetic in-ear: Load T7 and T8, compute bipolar derivation T8 - T7 (when `--channel in-ear`)
5. Channel name standardization: T3→T7, T4→T8 for cross-dataset compatibility
6. Average reference
7. Optional artifact removal for CANE: (`--skip-artifact-removal` to disable)
8. Segmentation: 10-second chunks
9. For in-ear: Apply 50% sign flip augmentation per chunk during training
10. STFT transformation: Convert to spectrograms
11. Normalization: Log-magnitude spectrograms

### IDUN Preprocessing
IDUN CSV files undergo a separate preprocessing pipeline (see `IDUNDataset.load_and_preprocess_idun_file()`):

1. Load CSV file (timestamp, ch1 columns)
2. Z-score normalize raw values
3. Detrend (linear)
4. Bandpass filter: 1-70 Hz (IIR)
5. Notch filter: 50 Hz (remove power line noise)
6. Segmentation: 10-second chunks (2500 samples at 250 Hz)
7. Quality-based chunk rejection (threshold=0.0 by default, rejects unmeasured chunks)
8. Per-chunk z-score normalization
9. Apply 50% sign flip augmentation per chunk during training
10. STFT transformation: Convert to spectrograms
11. Normalization: Log-magnitude spectrograms
