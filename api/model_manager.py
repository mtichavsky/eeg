"""
Model loading and selection logic for API inference.

This module manages loading models and selecting the appropriate model
based on electrode_setup and classification_task parameters.

Supports two loading strategies (configurable via MODEL_LOADING env var):
- ``startup``: Load all 4 models on startup (default)
- ``on_demand``: Load models on first request, cache in memory
"""

from typing import Literal

import torch.nn as nn

from api.config import config
from api.logging_config import get_logger
from thesis.data_preparation import EXPECTED_SPECTROGRAM_SHAPE
from thesis.model_factory import create_model

logger = get_logger(__name__)


class ModelManager:
    """
    Manages loading and selection of EEG classification models.

    Models available:
    - inear_2class: In-ear single-channel, binary classification
    - inear_4class: In-ear single-channel, 4-class classification
    - 8channel_2class: 8-channel (all), binary classification
    - 8channel_4class: 8-channel (all), 4-class classification
    """

    # Model selection table
    MODEL_CONFIG = {
        ("in-ear", "2class"): {
            "model_name": "Smaller",
            "num_classes": 2,
            "in_channels": 1,
            "key": "inear_2class",
        },
        ("in-ear", "4class"): {
            "model_name": "Smaller",
            "num_classes": 4,
            "in_channels": 1,
            "key": "inear_4class",
        },
        ("8channel", "2class"): {
            "model_name": "SmallerAllV2Attn",
            "num_classes": 2,
            "in_channels": 8,
            "key": "8channel_2class",
        },
        ("8channel", "4class"): {
            "model_name": "SmallerAllV2Attn",
            "num_classes": 4,
            "in_channels": 8,
            "key": "8channel_4class",
        },
    }

    def __init__(self) -> None:
        """
        Initialize ModelManager.

        Loading strategy is controlled by ``config.model_loading``:
        - ``"startup"``: loads all 4 models immediately (default)
        - ``"on_demand"``: defers loading until first request per model
        """
        self.device = config.get_device()
        self.models: dict[str, nn.Module | None] = {}
        self.spec_shape = EXPECTED_SPECTROGRAM_SHAPE

        logger.info(f"Initializing ModelManager on device: {self.device}")
        logger.info(f"Expected spectrogram shape: {self.spec_shape}")
        logger.info(f"Model loading strategy: {config.model_loading}")

        if config.model_loading == "startup":
            self._load_all_models()
        else:
            # Mark all models as not yet loaded
            for model_config in self.MODEL_CONFIG.values():
                self.models[model_config["key"]] = None

    def _load_all_models(self) -> None:
        """Load all 4 models from disk."""
        for (electrode_setup, classification_task), model_config in self.MODEL_CONFIG.items():
            self._load_single_model(electrode_setup, classification_task, model_config)

        loaded_count = sum(1 for m in self.models.values() if m is not None)
        logger.info(
            f"ModelManager initialized: {loaded_count}/{len(self.MODEL_CONFIG)} models loaded"
        )

    def _load_single_model(
        self,
        electrode_setup: str,
        classification_task: str,
        model_config: dict,
    ) -> nn.Module | None:
        """
        Load a single model from disk.

        :param str electrode_setup: Electrode setup key.
        :param str classification_task: Classification task key.
        :param dict model_config: Model configuration from MODEL_CONFIG.
        :return: Loaded model or None if loading failed.
        :rtype: nn.Module | None
        """
        key = model_config["key"]
        try:
            model_path = config.get_model_path(electrode_setup, classification_task)
            logger.info(f"Loading {key} from {model_path}")

            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                self.models[key] = None
                return None

            model = create_model(
                model_name=model_config["model_name"],
                spec_shape=self.spec_shape,
                dropout=0.0,  # model.eval() disables dropout during inference
                num_classes=model_config["num_classes"],
                device=self.device,
                in_channels=model_config["in_channels"],
                pretrained_checkpoint=str(model_path),
            )

            model.eval()
            self.models[key] = model
            logger.info(f"Successfully loaded {key}")
            return model

        except Exception as e:
            logger.error(f"Failed to load {key}: {e}", exc_info=True)
            self.models[key] = None
            return None

    def _get_model_config(
        self,
        electrode_setup: str,
        classification_task: str,
    ) -> dict:
        """
        Look up model config, raising ValueError for unknown combinations.

        :param str electrode_setup: Electrode configuration.
        :param str classification_task: Classification task.
        :return: Model config dict from MODEL_CONFIG.
        :rtype: dict
        :raises ValueError: If the combination is not in MODEL_CONFIG.
        """
        model_config = self.MODEL_CONFIG.get((electrode_setup, classification_task))
        if not model_config:
            raise ValueError(
                f"Invalid combination: electrode_setup={electrode_setup}, "
                f"classification_task={classification_task}"
            )
        return model_config

    def select_model(
        self,
        electrode_setup: Literal["in-ear", "8channel"],
        classification_task: Literal["2class", "4class"],
    ) -> nn.Module:
        """
        Select and return the appropriate model based on parameters.

        In on_demand mode, loads the model on first request if not yet loaded.

        :param Literal["in-ear", "8channel"] electrode_setup: Electrode configuration.
        :param Literal["2class", "4class"] classification_task: Classification task.
        :return: Loaded PyTorch model in eval mode.
        :rtype: nn.Module
        :raises ValueError: If model is not available.
        """
        model_config = self._get_model_config(electrode_setup, classification_task)

        key = model_config["key"]
        model = self.models.get(key)

        # On-demand loading: try to load if not yet loaded
        if model is None and config.model_loading == "on_demand":
            logger.info(f"On-demand loading model: {key}")
            model = self._load_single_model(electrode_setup, classification_task, model_config)

        if model is None:
            raise ValueError(f"Model {key} is not available (failed to load)")

        logger.info(f"Selected model: {key}")
        return model

    def get_model_status(self) -> dict[str, bool]:
        """
        Get availability status for all models.

        :return: Dictionary mapping model keys to availability (True/False).
        :rtype: dict[str, bool]
        """
        return {key: (model is not None) for key, model in self.models.items()}

    @staticmethod
    def get_channel_name(electrode_setup: Literal["in-ear", "8channel"]) -> str:
        """
        Map electrode_setup to channel parameter for preprocessing.

        :param Literal["in-ear", "8channel"] electrode_setup: Electrode configuration.
        :return: Channel name for preprocessing ("in-ear" or "all").
        :rtype: str
        """
        return (
            "in-ear" if electrode_setup == "in-ear" else "all"
        )  # "all" = channel name for preprocessing

    def get_num_classes(
        self,
        electrode_setup: Literal["in-ear", "8channel"],
        classification_task: Literal["2class", "4class"],
    ) -> int:
        """
        Get number of classes for given configuration.

        :param Literal["in-ear", "8channel"] electrode_setup: Electrode configuration.
        :param Literal["2class", "4class"] classification_task: Classification task.
        :return: Number of output classes (2 or 4).
        :rtype: int
        """
        return self._get_model_config(electrode_setup, classification_task)["num_classes"]
