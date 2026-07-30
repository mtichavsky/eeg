"""Test script for CANE preprocessing and spectrogram generation. Used during prototyping."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from thesis.dataset import (
    CANEDataset,
    SpectrogramDataset,
)
from thesis.stft import (
    EXPECTED_SPECTROGRAM_SHAPE,
    MODEL_FS,
    STFT_NOVERLAP,
    STFT_NPERSEG,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CANE_DIR = Path("../CANE/")

# Test file
test_file = CANE_DIR / "anxiety/ec/1065_EC.csv"

logger.info(f"Testing CANE preprocessing with file: {test_file}")

# Load and preprocess using the static method
chunks = CANEDataset.load_and_preprocess_cane_raw_file(test_file, channel="Fp1")
fs = CANEDataset.FS

logger.info(f"Chunks shape: {chunks.shape}")
logger.info(f"Detected sampling rate: {fs:.2f} Hz")
chunk_duration = chunks.shape[2] / fs

# Convert to spectrograms using SpectrogramDataset.convert_to_spectrograms
if chunks.shape[0] > 0:
    logger.info("\nConverting to spectrograms using SpectrogramDataset.convert_to_spectrograms")
    logger.info(f"  nperseg: {STFT_NPERSEG}")
    logger.info(f"  noverlap: {STFT_NOVERLAP}")
    logger.info(f"  source fs: {fs:.2f} (resampled to {MODEL_FS} Hz before STFT)")

    spectrograms = SpectrogramDataset.convert_to_spectrograms(chunks, source_fs=fs)

    logger.info(f"\nNumber of spectrograms created: {len(spectrograms)}")

    if len(spectrograms) > 0:
        first_spec = spectrograms[0]
        logger.info(f"First spectrogram shape: {first_spec.shape}")
        logger.info(
            f"Expected shape: (1, {EXPECTED_SPECTROGRAM_SHAPE[0]}, {EXPECTED_SPECTROGRAM_SHAPE[1]})"
        )

        # Print details
        spec_shape = first_spec.shape[1:]  # Remove channel dimension
        logger.info(f"Frequency bins: {spec_shape[0]} (expected {EXPECTED_SPECTROGRAM_SHAPE[0]})")
        logger.info(f"Time frames: {spec_shape[1]} (expected {EXPECTED_SPECTROGRAM_SHAPE[1]})")

        if spec_shape == torch.Size(list(EXPECTED_SPECTROGRAM_SHAPE)):
            logger.info(f"✅ SUCCESS: Spectrogram shape matches {EXPECTED_SPECTROGRAM_SHAPE}!")
        else:
            logger.warning(
                f"⚠️  WARNING: Shape mismatch! Got {spec_shape}, "
                f"expected {EXPECTED_SPECTROGRAM_SHAPE}"
            )

        # Print sample statistics
        logger.info("\nSpectrogram statistics:")
        logger.info(f"  Min: {first_spec.min():.4f}")
        logger.info(f"  Max: {first_spec.max():.4f}")
        logger.info(f"  Mean: {first_spec.mean():.4f}")
        logger.info(f"  Std: {first_spec.std():.4f}")

        # Plot and save the spectrogram
        filename = "spectrogram_cane.png"
        logger.info(f"\nSaving spectrogram plot to {filename}")

        # Use the already-computed spectrogram (inverse the log1p transform)
        magnitude = np.expm1(first_spec[0].numpy())  # expm1 is inverse of log1p

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(
            np.linspace(0, chunk_duration, first_spec.shape[2]),
            np.arange(first_spec.shape[1]),
            magnitude,
            shading="gouraud",
            cmap="viridis",
            vmax=1,
        )
        plt.colorbar()
        plt.xlabel("Sec")
        plt.ylabel("Hz")
        plt.title(f"CANE Spectrogram ({fs} Hz, Shape: {spec_shape[0]}x{spec_shape[1]})")
        plt.ylim([0, 120])
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        logger.info("Spectrogram saved successfully!")
        plt.close()
    else:
        logger.error("No spectrograms were created!")
else:
    logger.error("No chunks were created!")
