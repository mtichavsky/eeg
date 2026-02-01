"""Test script for CANE preprocessing and spectrogram generation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import stft

from thesis.dataset import (
    MDDDataset,
    SpectrogramDataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MDD_DIR = Path("../MDD/")


# Test file
test_file = MDD_DIR / "MDD S5 EC.edf"

logger.info(f"Testing MDD preprocessing with file: {test_file}")

# Load and preprocess using the static method
chunks = MDDDataset.load_and_preprocess_mdd_raw_file(test_file, channel="Fp1")


logger.info(f"Chunks shape: {chunks.shape}")

# Convert to spectrograms using SpectrogramDataset.convert_to_spectrograms
if chunks.shape[0] > 0:
    logger.info("\nConverting to spectrograms using SpectrogramDataset.convert_to_spectrograms")
    spectrograms = SpectrogramDataset.convert_to_spectrograms(
        chunks, nperseg=256, fs=250, noverlap=192, window="hamming"
    )

    logger.info(f"\nNumber of spectrograms created: {len(spectrograms)}")

    if len(spectrograms) > 0:
        first_spec = spectrograms[0]
        logger.info(f"First spectrogram shape: {first_spec.shape}")
        logger.info("Expected shape: (1, 129, 41)")

        # Print details
        spec_shape = first_spec.shape[1:]  # Remove channel dimension

        if spec_shape == torch.Size([129, 41]):
            logger.info("✅ SUCCESS: Spectrogram shape matches MDD target (129, 41)!")
        else:
            logger.warning(f"⚠️  WARNING: Shape mismatch! Got {spec_shape}, expected (129, 41)")

        # Plot and save the spectrogram
        logger.info("\nSaving spectrogram plot to spectrogram_mdd.png")

        # Get raw signal from first chunk
        first_chunk = chunks[0, 0, :].numpy()

        # Compute STFT for visualization
        f_viz, t_viz, Zxx_viz = stft(
            first_chunk, nperseg=256, fs=250, noverlap=192, window="hamming"
        )

        # Convert to absolute magnitude and scale by 5 to match MDD levels
        magnitude = np.abs(Zxx_viz)

        plt.figure(figsize=(10, 6))
        plt.pcolormesh(
            t_viz,
            f_viz,
            magnitude,
            shading="gouraud",
            cmap="viridis",
            vmax=1,  # Back to 5 since we scaled the signal by 5
        )
        plt.colorbar()
        plt.xlabel("Sec")
        plt.ylabel("Hz")
        plt.title(f"MDD Spectrogram (250 Hz, Shape: {spec_shape[0]}x{spec_shape[1]})")
        plt.ylim([0, 120])
        plt.tight_layout()
        plt.savefig("spectrogram_mdd.png", dpi=150, bbox_inches="tight")
        logger.info("Spectrogram saved successfully!")
        plt.close()
    else:
        logger.error("No spectrograms were created!")
else:
    logger.error("No chunks were created!")
