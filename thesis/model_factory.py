"""
Model creation factory for EEG classification models.

Provides a single `create_model()` function used by both CLI and API
to instantiate, optionally load pretrained weights, and freeze layers.
"""

import argparse
import logging
from dataclasses import asdict, dataclass
from dataclasses import replace as dataclasses_replace
from typing import Any

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


@dataclass
class LGGNetConfig:
    """
    Hyperparameters for the LGGNet architecture (TNNLS 2023).

    Pool size scaled from the paper's 16 at 128 Hz to 32 at 250 Hz.
    Graph type 'hemisphere' matches the paper's default ``hem`` topology.
    """

    num_time: int = 2500
    """Number of time samples per chunk (10 s × 250 Hz = 2500)."""

    sampling_rate: int = 250
    """EEG sampling rate in Hz."""

    num_T: int = 64
    """Number of temporal filters per branch."""

    out_graph: int = 32
    """Output dimension of the GCN layer."""

    pool: int = 32
    """PowerLayer pooling window. Paper uses 16 at 128 Hz → scaled to 32 at 250 Hz."""

    pool_step_rate: float = 0.25
    """Pool stride as a fraction of pool size (stride = pool × pool_step_rate)."""

    graph_type: str = "hemisphere"
    """Graph topology: 'frontal' (4 regions) or 'hemisphere' (3 regions)."""


@dataclass
class LGGNetSConfig:
    """
    Smaller LGGNet variant (~585 K params), matching DeformerS in parameter count.

    Reduces num_T from 64 to 32 relative to LGGNetConfig; all other fields identical.
    """

    num_time: int = 2500
    sampling_rate: int = 250
    num_T: int = 32
    """Temporal filters per branch (reduced from 64 to halve the feature dimension)."""
    out_graph: int = 32
    pool: int = 32
    pool_step_rate: float = 0.25
    graph_type: str = "hemisphere"


@dataclass
class TSceptionConfig:
    """
    Hyperparameters for the TSception architecture (IJCNN 2020).

    Defaults match the paper's ``Train.py`` configuration (hidden=128) adapted to
    250 Hz EEG with 10-second chunks.  Temporal kernel sizes are computed as
    ``int(fraction × sampling_rate)`` for fractions [0.5, 0.25, 0.125].
    """

    num_T: int = 9
    """Temporal inception filter count (paper default)."""

    num_S: int = 6
    """Spatial filter count per branch (paper default)."""

    hidden: int = 128
    """FC hidden layer nodes (paper default for training on DEAP / MAHNOB-HCI)."""

    sampling_rate: int = 250
    """EEG sampling rate in Hz — determines temporal kernel sizes."""

    num_time: int = 2500
    """Number of time samples per chunk (10 s × 250 Hz = 2500)."""


@dataclass
class TSceptionSConfig:
    """
    Smaller TSception variant (~264 K params), matching SmallerAll in parameter count.

    Uses hidden=32, following the paper's own cross-dataset recommendation:
    'We also suggest T=S=15 and hidden_node=32 when applying TSception to other datasets.'
    All other fields are identical to ``TSceptionConfig``.
    """

    num_T: int = 9
    num_S: int = 6
    hidden: int = 32
    """FC hidden layer nodes (paper's cross-dataset recommendation)."""
    sampling_rate: int = 250
    num_time: int = 2500


# Maps each raw-EEG model name to its default config class.
# Add an entry here whenever a new raw-EEG model is registered in MODEL_REGISTRY.
RAW_EEG_DEFAULT_CONFIGS: dict[str, type] = {
    "Deformer": DeformerConfig,
    "DeformerS": DeformerSConfig,
    "LGGNet": LGGNetConfig,
    "LGGNetS": LGGNetSConfig,
    "TSception": TSceptionConfig,
    "TSceptionS": TSceptionSConfig,
}


# Per-family mapping from CLI arg name → config dataclass field name.
# When adding a new CLI flag for a raw-EEG model, register it here.
_CLI_OVERRIDES: dict[str, dict[str, str]] = {
    "Deformer": {
        "deformer_depth": "depth",
        "deformer_heads": "heads",
        "deformer_num_kernel": "num_kernel",
        "deformer_mlp_dim": "mlp_dim",
        "deformer_dim_head": "dim_head",
        "deformer_temporal_kernel": "temporal_kernel",
    },
    "LGGNet": {
        "lggnet_num_t": "num_T",
        "lggnet_out_graph": "out_graph",
        "lggnet_pool": "pool",
        "lggnet_pool_step_rate": "pool_step_rate",
        "lggnet_graph_type": "graph_type",
    },
    "TSception": {
        "tsception_num_t": "num_T",
        "tsception_num_s": "num_S",
        "tsception_hidden": "hidden",
        "tsception_sampling_rate": "sampling_rate",
    },
}


def _model_family(model_name: str) -> str:
    """Return the raw-EEG family ("Deformer", "LGGNet", "TSception") for a model name."""
    for family in _CLI_OVERRIDES:
        if family in model_name:
            return family
    raise ValueError(f"{model_name} is not a raw-EEG model")


def build_raw_eeg_config(
    model_name: str, args: argparse.Namespace
) -> DeformerConfig | DeformerSConfig | LGGNetConfig | LGGNetSConfig | TSceptionConfig | TSceptionSConfig | None:
    """
    Build the raw-EEG config for ``model_name`` from defaults plus non-None CLI overrides.

    Returns ``None`` for spectrogram models. ``args.chunk_duration`` (seconds) is converted to
    ``num_time`` at 250 Hz. Family-specific CLI flags listed in ``_CLI_OVERRIDES`` are applied
    only when the user passed a non-None value.

    :param str model_name: Model name from MODEL_REGISTRY.
    :param argparse.Namespace args: Parsed CLI arguments.
    :return: Resolved config dataclass instance, or None for spectrogram models.
    """
    if model_name not in RAW_EEG_MODELS:
        return None

    base = RAW_EEG_DEFAULT_CONFIGS[model_name]()
    overrides: dict[str, Any] = {"num_time": int(args.chunk_duration * 250)}
    for arg_name, field_name in _CLI_OVERRIDES[_model_family(model_name)].items():
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[field_name] = value
    return dataclasses_replace(base, **overrides)


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
    raw_eeg_config: DeformerConfig | LGGNetConfig | TSceptionConfig | TSceptionSConfig | None = None,
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
    :param DeformerConfig | LGGNetConfig | None raw_eeg_config: Hyperparameters for raw-EEG
        models (Deformer, LGGNet variants). Ignored for spectrogram models. Defaults from
        ``RAW_EEG_DEFAULT_CONFIGS`` are applied when None is passed for a raw-EEG model.
    :return: Initialized model.
    :rtype: nn.Module
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_class, rnn_hidden = MODEL_REGISTRY[model_name]

    if model_name in RAW_EEG_MODELS:
        # Raw EEG models take (batch, channels, time) directly — bypass spectrogram pipeline
        cfg = raw_eeg_config if raw_eeg_config is not None else RAW_EEG_DEFAULT_CONFIGS[model_name]()
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
