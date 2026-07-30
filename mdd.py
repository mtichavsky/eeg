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
    load_and_preprocess_edf_file,
)
from thesis.stft import (
    EXPECTED_SPECTROGRAM_SHAPE,
    FREQ_CUTOFF_HZ,
    MODEL_FS,
    NUM_FREQ_BINS,
    STFT_NOVERLAP,
    STFT_NPERSEG,
    STFT_WINDOW,
    resample_to_model_fs,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MDD_DIR = Path("../MDD/")


# Test file
test_file = MDD_DIR / "MDD S5 EC.edf"

logger.info(f"Testing MDD preprocessing with file: {test_file}")

# Load and preprocess using the module-level function
chunks = load_and_preprocess_edf_file(
    test_file,
    channel="Fp1",
    fs=250,
    channel_mapping=MDDDataset.CHANNEL_MAPPING,
    channel_order=list(MDDDataset.CHANNEL_MAPPING.keys()),
)


logger.info(f"Chunks shape: {chunks.shape}")

# Convert to spectrograms using SpectrogramDataset.convert_to_spectrograms
if chunks.shape[0] > 0:
    logger.info("\nConverting to spectrograms using SpectrogramDataset.convert_to_spectrograms")
    spectrograms = SpectrogramDataset.convert_to_spectrograms(chunks, source_fs=MDDDataset.FS)

    logger.info(f"\nNumber of spectrograms created: {len(spectrograms)}")

    if len(spectrograms) > 0:
        first_spec = spectrograms[0]
        logger.info(f"First spectrogram shape: {first_spec.shape}")
        logger.info(
            f"Expected shape: (1, {EXPECTED_SPECTROGRAM_SHAPE[0]}, {EXPECTED_SPECTROGRAM_SHAPE[1]})"
        )

        # Print details
        spec_shape = first_spec.shape[1:]  # Remove channel dimension

        if spec_shape == torch.Size(list(EXPECTED_SPECTROGRAM_SHAPE)):
            logger.info(f"✅ SUCCESS: Spectrogram shape matches {EXPECTED_SPECTROGRAM_SHAPE}!")
        else:
            logger.warning(
                f"⚠️  WARNING: Shape mismatch! Got {spec_shape}, "
                f"expected {EXPECTED_SPECTROGRAM_SHAPE}"
            )

        # Plot and save the spectrogram
        logger.info("\nSaving spectrogram plot to spectrogram_mdd.png")

        # Get raw signal from first chunk
        first_chunk = chunks[0, 0, :].numpy()

        # Compute STFT for visualization, cropped the same way the model input is
        f_viz, t_viz, Zxx_viz = stft(
            resample_to_model_fs(first_chunk, MDDDataset.FS),
            nperseg=STFT_NPERSEG,
            fs=MODEL_FS,
            noverlap=STFT_NOVERLAP,
            window=STFT_WINDOW,
        )
        f_viz = f_viz[:NUM_FREQ_BINS]
        Zxx_viz = Zxx_viz[:NUM_FREQ_BINS]

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
        plt.title(f"MDD Spectrogram ({MODEL_FS} Hz, Shape: {spec_shape[0]}x{spec_shape[1]})")
        plt.ylim([0, FREQ_CUTOFF_HZ])
        plt.tight_layout()
        plt.savefig("spectrogram_mdd.png", dpi=150, bbox_inches="tight")
        logger.info("Spectrogram saved successfully!")
        plt.close()
    else:
        logger.error("No spectrograms were created!")
else:
    logger.error("No chunks were created!")
