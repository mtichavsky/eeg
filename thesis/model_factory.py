"""
Model creation factory for EEG classification models.

Provides a single `create_model()` function used by both CLI and API
to instantiate, optionally load pretrained weights, and freeze layers.
"""

import logging
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from thesis.model import MODEL_REGISTRY, RAW_EEG_MODELS

logger = logging.getLogger(__name__)


@dataclass
class DeformerConfig:
    """
    Hyperparameters for the Deformer architecture (EEG-Deformer, J-BHI 2024).

    All values default to the paper's settings for 250 Hz EEG.
    ``num_time`` and ``temporal_kernel`` depend on the sampling rate and segment length;
    the defaults match 10-second chunks at 250 Hz with kernel = Odd(0.1 × fs).
    """

    num_time: int = 2500
    """Number of time samples per chunk (10 s × 250 Hz = 2500)."""

    temporal_kernel: int = 25
    """Temporal CNN kernel size. Must be odd. Paper formula: Odd(0.1 × fs)."""

    num_kernel: int = 64
    """Number of filters in the shallow CNN encoder."""

    depth: int = 4
    """Number of hierarchical transformer layers."""

    heads: int = 16
    """Number of multi-head attention heads."""

    mlp_dim: int = 16
    """Hidden dimension of the FeedForward block inside each transformer layer."""

    dim_head: int = 16
    """Dimension per attention head."""


@dataclass
class DeformerSConfig:
    """
    Hyperparameters for the smaller Deformer variant (``DeformerS``).

    Reduces the original Deformer (~1.78 M params) to ~588 k (~1/3) by:
    - ``num_kernel`` 64 → 48  (narrower CNN encoder and transformer token width)
    - ``depth``      4  → 3   (one fewer hierarchical level; final sequence is 156 samples
                               instead of 78, i.e. ~0.6 s vs ~0.3 s temporal resolution)
    - ``heads``      16 → 4   (inner attention dim 256 → 64; biggest single saving)

    All other fields match ``DeformerConfig``.
    """

    num_time: int = 2500
    """Number of time samples per chunk (10 s × 250 Hz = 2500)."""

    temporal_kernel: int = 25
    """Temporal CNN kernel size. Must be odd. Paper formula: Odd(0.1 × fs)."""

    num_kernel: int = 48
    """Number of filters in the shallow CNN encoder (reduced from 64)."""

    depth: int = 3
    """Number of hierarchical transformer layers (reduced from 4)."""

    heads: int = 4
    """Number of multi-head attention heads (reduced from 16)."""

    mlp_dim: int = 16
    """Hidden dimension of the FeedForward block inside each transformer layer."""

    dim_head: int = 16
    """Dimension per attention head."""


# Maps each raw-EEG model name to its default config class.
# Add an entry here whenever a new Deformer variant is registered in MODEL_REGISTRY.
_DEFORMER_DEFAULT_CONFIGS: dict[str, type[DeformerConfig]] = {
    "Deformer": DeformerConfig,
    "DeformerS": DeformerSConfig,
}


def create_model(
    model_name: str,
    spec_shape: tuple[int, int],
    dropout: float,
    num_classes: int,
    device: torch.device,
    in_channels: int = 1,
    pretrained_checkpoint: str | None = None,
    freeze_cnn: bool = False,
    freeze_lstm: bool = False,
    deformer_config: DeformerConfig | None = None,
) -> nn.Module:
    """
    Create and initialize a model, optionally loading pretrained weights.

    :param str model_name: Model architecture name from MODEL_REGISTRY.
    :param tuple spec_shape: Input spectrogram shape (height, width). Ignored for raw EEG models.
    :param float dropout: Dropout rate.
    :param int num_classes: Number of output classes.
    :param torch.device device: Device to place model on.
    :param int in_channels: Number of input channels (1 for single-channel, 8 for multi-channel).
    :param str | None pretrained_checkpoint: Path to pretrained checkpoint for transfer learning.
    :param bool freeze_cnn: If True, freeze CNN layers (conv1, conv2) during training.
    :param bool freeze_lstm: If True, freeze LSTM layer during training.
    :param DeformerConfig | None deformer_config: Hyperparameters for Deformer. Required when
        model_name is in RAW_EEG_MODELS; ignored otherwise. Defaults are applied if None is
        passed for a raw-EEG model.
    :return: Initialized model.
    :rtype: nn.Module
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_class, rnn_hidden = MODEL_REGISTRY[model_name]

    if model_name in RAW_EEG_MODELS:
        # Raw EEG models (e.g. Deformer) take (batch, channels, time) directly
        cfg = deformer_config if deformer_config is not None else _DEFORMER_DEFAULT_CONFIGS[model_name]()
        model = model_class(
            num_chan=in_channels,
            num_classes=num_classes,
            dropout=dropout,
            **asdict(cfg),
        ).to(device)
    else:
        # Spectrogram models take (batch, channels, freq_bins, time_frames)
        model = model_class(
            input_shape=spec_shape,
            in_channels=in_channels,
            rnn_hidden=rnn_hidden,
            dropout=dropout,
            num_classes=num_classes,
        ).to(device)

    # Transfer learning: load pretrained weights if provided
    if pretrained_checkpoint:
        logger.info(f"Loading pretrained weights from: {pretrained_checkpoint}")
        saved = torch.load(pretrained_checkpoint, weights_only=False, map_location=device)

        # Load state dict (strict=False allows for num_classes mismatch in final layer)
        model.load_state_dict(saved["model_state_dict"], strict=False)
        logger.info("Pretrained weights loaded successfully")

        if freeze_cnn:
            # Freeze CNN layers (conv1, conv2 and their dropout layers)
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.conv2.parameters():
                param.requires_grad = False
            for param in model.dropout2d_1.parameters():
                param.requires_grad = False
            for param in model.dropout2d_2.parameters():
                param.requires_grad = False

        # TODO: attention doesn't have to be freezed?
        if freeze_lstm:
            # Freeze RNN layer (guard against attention models that have no .rnn)
            if hasattr(model, "rnn"):
                for param in model.rnn.parameters():
                    param.requires_grad = False
            else:
                logger.warning(
                    "freeze_lstm requested but model has no .rnn attribute (attention model?)"
                )

        # Log which layers are frozen
        if freeze_cnn or freeze_lstm:
            all_params, trainable_params = model.count_parameters()
            frozen_parts = []
            if freeze_cnn:
                frozen_parts.append("CNN")
            if freeze_lstm:
                frozen_parts.append("LSTM")
            logger.info(
                f"{'+'.join(frozen_parts)} layers frozen. "
                f"Trainable params: {trainable_params}/{all_params}"
            )

    return model
