"""
Model creation factory for EEG classification models.

Provides a single `create_model()` function used by both CLI and API
to instantiate, optionally load pretrained weights, and freeze layers.
"""

import logging

import torch
import torch.nn as nn

from thesis.model import MODEL_REGISTRY

logger = logging.getLogger(__name__)


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
) -> nn.Module:
    """
    Create and initialize a model, optionally loading pretrained weights.

    :param str model_name: Model architecture ("CNN_LSTM_DepCap" or "Smaller").
    :param tuple spec_shape: Input spectrogram shape (height, width).
    :param float dropout: Dropout rate.
    :param int num_classes: Number of output classes.
    :param torch.device device: Device to place model on.
    :param int in_channels: Number of input channels (1 for single-channel, 8 for multi-channel).
    :param str | None pretrained_checkpoint: Path to pretrained checkpoint for transfer learning.
    :param bool freeze_cnn: If True, freeze CNN layers (conv1, conv2) during training.
    :param bool freeze_lstm: If True, freeze LSTM layer during training.
    :return: Initialized model.
    :rtype: nn.Module
    """
    # Get model class and default parameters from registry
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_class, rnn_hidden = MODEL_REGISTRY[model_name]

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
