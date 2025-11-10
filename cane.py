"""Test script for CANE preprocessing and spectrogram generation."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import stft

from thesis.dataset import (
    CANEDataset,
    SpectrogramDataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CANE_DIR = Path("../CANE/")

# Test file
test_file = CANE_DIR / "anxiety/ec/1065_EC.csv"

logger.info(f"Testing CANE preprocessing with file: {test_file}")

# Load and preprocess using the static method
chunks = CANEDataset.load_and_preprocess_cane_raw_file(test_file, channel="d4")
fs = CANEDataset.FS

logger.info(f"Chunks shape: {chunks.shape}")
logger.info(f"Detected sampling rate: {fs:.2f} Hz")

# Convert to spectrograms using SpectrogramDataset.convert_to_spectrograms
if chunks.shape[0] > 0:
    logger.info("\nConverting to spectrograms using SpectrogramDataset.convert_to_spectrograms")
    logger.info(f"  nperseg: {CANEDataset.STFT_NPERSEG}")
    logger.info(f"  noverlap: {CANEDataset.STFT_NOVERLAP}")
    logger.info(f"  fs: {fs:.2f}")

    spectrograms = SpectrogramDataset.convert_to_spectrograms(
        chunks,
        nperseg=CANEDataset.STFT_NPERSEG,
        fs=fs,
        noverlap=CANEDataset.STFT_NOVERLAP,
        window="hamming",
    )

    logger.info(f"\nNumber of spectrograms created: {len(spectrograms)}")

    if len(spectrograms) > 0:
        first_spec = spectrograms[0]
        logger.info(f"First spectrogram shape: {first_spec.shape}")
        logger.info("Expected shape: (1, 129, 41)")

        # Print details
        spec_shape = first_spec.shape[1:]  # Remove channel dimension
        logger.info(f"Frequency bins: {spec_shape[0]} (expected 129)")
        logger.info(f"Time frames: {spec_shape[1]} (expected ~41)")

        if spec_shape == torch.Size([129, 41]):
            logger.info("✅ SUCCESS: Spectrogram shape matches MDD target (129, 41)!")
        else:
            logger.warning(f"⚠️  WARNING: Shape mismatch! Got {spec_shape}, expected (129, 41)")
            logger.warning("Adjust CANE_STFT_NOVERLAP to fix this.")

            # Calculate correct noverlap
            chunk_length = chunks.shape[2]
            target_frames = 41
            correct_noverlap = CANEDataset.STFT_NPERSEG - (
                chunk_length - CANEDataset.STFT_NPERSEG
            ) / (target_frames - 1)
            logger.info(f"Suggested noverlap: {int(correct_noverlap)}")

        # Print sample statistics
        logger.info("\nSpectrogram statistics:")
        logger.info(f"  Min: {first_spec.min():.4f}")
        logger.info(f"  Max: {first_spec.max():.4f}")
        logger.info(f"  Mean: {first_spec.mean():.4f}")
        logger.info(f"  Std: {first_spec.std():.4f}")

        # Plot and save the spectrogram
        logger.info("\nSaving spectrogram plot to spectrogram.png")

        # Get raw signal from first chunk
        first_chunk = chunks[0, 0, :].numpy()

        # Compute STFT for visualization
        f_viz, t_viz, Zxx_viz = stft(
            first_chunk,
            fs=fs,
            nperseg=CANEDataset.STFT_NPERSEG,
            noverlap=CANEDataset.STFT_NOVERLAP,
            window="hamming",
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
        plt.title(f"CANE Spectrogram ({fs} Hz, Shape: {spec_shape[0]}x{spec_shape[1]})")
        plt.ylim([0, 120])
        plt.tight_layout()
        plt.savefig("spectrogram_cane.png", dpi=150, bbox_inches="tight")
        logger.info("Spectrogram saved successfully!")
        plt.close()
    else:
        logger.error("No spectrograms were created!")
else:
    logger.error("No chunks were created!")
