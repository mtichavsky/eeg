"""
API configuration management using Pydantic settings.

This module provides centralized configuration with environment variable support.
"""

import logging
from pathlib import Path
from typing import Literal

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class APIConfig(BaseSettings):
    """
    API configuration with environment variable support.

    All settings can be overridden via environment variables.
    Example: MODEL_DIR=/custom/path
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model paths
    model_dir: Path = Path("models")
    model_in_ear_binary: str = "model_inear_binary.pth"
    model_in_ear_4class: str = "model_inear_4class.pth"
    model_8channel_binary: str = "model_8channel_binary.pth"
    model_8channel_4class: str = "model_8channel_4class.pth"

    # Device configuration
    device: Literal["auto", "cuda", "cpu"] = "auto"

    # Model loading strategy: "startup" loads all models on startup,
    # "on_demand" loads models on first request and caches them
    model_loading: Literal["startup", "on_demand"] = "startup"

    # File upload limits
    max_file_size_mb: int = 100

    # Rate limiting - 10 requests per 10 seconds
    rate_limit_times: int = 10
    rate_limit_seconds: int = 10

    # Logging
    log_level: str = "INFO"
    json_pretty_print: bool = False

    # STFT parameters (must match training configuration)
    # TODO: calculate based on input FS to normalize spectrogram size
    stft_nperseg: int = 256
    stft_noverlap: int = 192
    stft_fs: int = 250  # Default sampling frequency for spectrograms

    # EEG preprocessing parameters
    # TODO some shared config with the rest of the app - this very depends on the training
    chunk_duration_sec: int = 10
    bandpass_low: float = 1.0
    bandpass_high: float = 70.0
    notch_freq: float = 50.0

    def get_device(self) -> torch.device:
        """
        Get the PyTorch device based on configuration.

        :return: PyTorch device (cuda or cpu).
        :rtype: torch.device
        """
        if self.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Auto-detected device: {device}")
            return device
        elif self.device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return torch.device("cpu")
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def get_model_path(self, electrode_setup: str, classification_task: str) -> Path:
        """
        Get the model checkpoint path for given parameters.

        :param str electrode_setup: Electrode setup ("in-ear" or "8channel").
        :param str classification_task: Classification task ("2class" or "4class").
        :return: Full path to model checkpoint.
        :rtype: Path
        """
        safe_setup = electrode_setup.replace("-", "_")
        model_key = f"model_{safe_setup}_{classification_task}"
        model_filename = getattr(self, model_key)
        return self.model_dir / model_filename

    @property
    def max_file_size_bytes(self) -> int:
        """
        Get maximum file size in bytes.

        :return: Maximum file size in bytes.
        :rtype: int
        """
        return self.max_file_size_mb * 1024 * 1024


# Global config instance
config = APIConfig()
