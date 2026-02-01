"""
Centralized label definitions for EEG classification.

This module provides the single source of truth for all label-related constants,
mappings, and conversions across the codebase.

Label System Architecture:
==========================

CANONICAL LABELS (4-class convention):
--------------------------------------
Used when combining datasets or training in 4-class mode:
- 0 = HEALTHY (healthy control)
- 1 = ANXIETY_ONLY
- 2 = DEPRESSION_ONLY
- 3 = COMORBID (anxiety + depression)

BINARY LABELS (2-class mode):
-----------------------------
Used for binary classification (healthy vs pathological):
- 0 = HEALTHY
- 1 = PATHOLOGICAL (any disorder: anxiety, depression, or comorbid)

Dataset Labels (returned by dataset classes):
---------------------------------------------
All datasets now return CanonicalLabel instances directly:
- MDDDataset: HEALTHY=0, DEPRESSION_ONLY=2
- CANEDataset/IDUNDataset: HEALTHY=0, ANXIETY_ONLY=1, DEPRESSION_ONLY=2, COMORBID=3
- SADDataset: HEALTHY=0, ANXIETY_ONLY=1

Label Remapping Flow:
====================
1. Dataset loads file → assigns CanonicalLabel instance
2. FlattenedSpectrogramDataset → applies CanonicalLabel → BinaryLabel if 2-class mode
"""

from enum import IntEnum
from typing import Optional


class CanonicalLabel(IntEnum):
    """
    Canonical 4-class labels used across all datasets.

    This is the standard convention when combining datasets or training in 4-class mode.
    All datasets return these values directly.
    """

    HEALTHY = 0  # Healthy control
    ANXIETY_ONLY = 1  # Anxiety disorder without depression
    DEPRESSION_ONLY = 2  # Depression (MDD) without anxiety
    COMORBID = 3  # Both anxiety and depression


class BinaryLabel(IntEnum):
    """
    Binary labels for 2-class classification.

    Used when training in binary mode (healthy vs any pathological condition).
    """

    HEALTHY = 0  # Healthy control
    PATHOLOGICAL = 1  # Any disorder (anxiety, depression, or comorbid)


class LabelMapping:
    """
    Static methods for label mapping operations.

    Handles conversions between different label systems (MDD raw → canonical,
    canonical → binary, dataset-specific logic).
    """

    @staticmethod
    def get_mdd_to_canonical() -> dict[int, int]:
        """
        Get mapping to convert MDD raw labels to canonical 4-class labels.

        MDD uses binary labels (0=healthy, 1=depression).
        In 4-class mode, these must be remapped:
        - 0 (healthy) → 0 (HEALTHY)
        - 1 (depression) → 2 (DEPRESSION_ONLY)

        This prevents MDD depression from being mislabeled as anxiety (class 1).

        :return: {raw_label: canonical_label} mapping
        :rtype: dict[int, int]
        """
        return {
            0: CanonicalLabel.HEALTHY,  # MDD healthy → Canonical HEALTHY
            1: CanonicalLabel.DEPRESSION_ONLY,  # MDD depression → Canonical DEPRESSION_ONLY
        }

    @staticmethod
    def get_canonical_to_binary() -> dict[int, int]:
        """
        Get mapping to convert canonical 4-class labels to binary labels.

        Collapses all pathological classes (anxiety, depression, comorbid) to 1.

        :return: {canonical_label: binary_label} mapping
        :rtype: dict[int, int]
        """
        return {
            CanonicalLabel.HEALTHY: BinaryLabel.HEALTHY,
            CanonicalLabel.ANXIETY_ONLY: BinaryLabel.PATHOLOGICAL,
            CanonicalLabel.DEPRESSION_ONLY: BinaryLabel.PATHOLOGICAL,
            CanonicalLabel.COMORBID: BinaryLabel.PATHOLOGICAL,
        }

    @staticmethod
    def get_for_dataset(
        dataset_name: str,
        num_classes: int,
        explicit_mapping: Optional[dict[int, int]] = None,
    ) -> Optional[dict[int, int]]:
        """
        Get the appropriate label mapping for a dataset and classification mode.

        Since datasets now return CanonicalLabel instances directly, this function
        only handles binary mode mapping (CanonicalLabel → BinaryLabel).

        :param str dataset_name: Name of dataset ("mdd", "cane", "idun", "sad", "ax_malik")
        :param int num_classes: Number of classes (2 or 4)
        :param Optional[dict[int, int]] explicit_mapping: Optional explicit mapping
            (overrides all defaults)
        :return: Label mapping, or None if no remapping needed
        :rtype: Optional[dict[int, int]]
        """
        # Explicit mapping takes precedence
        if explicit_mapping is not None:
            return explicit_mapping

        # Binary mode: collapse CanonicalLabel (4-class) to BinaryLabel (2-class)
        if num_classes == 2:
            return LabelMapping.get_canonical_to_binary()

        # 4-class mode: datasets return CanonicalLabel directly, no mapping needed
        return None


class LabelUtils:
    """
    Static utility methods for label operations.

    Provides human-readable names and validation for label values.
    """

    @staticmethod
    def get_canonical_name(label: int) -> str:
        """
        Get human-readable name for a canonical label.

        :param int label: Canonical label (0-3)
        :return: Label name (e.g., "Healthy", "Anxiety Only", "Depression Only", "Comorbid")
        :rtype: str
        """
        try:
            return CanonicalLabel(label).name.replace("_", " ").title()
        except ValueError:
            return f"Unknown({label})"

    @staticmethod
    def get_binary_name(label: int) -> str:
        """
        Get human-readable name for a binary label.

        :param int label: Binary label (0 or 1)
        :return: Label name ("Healthy" or "Pathological")
        :rtype: str
        """
        try:
            return BinaryLabel(label).name.title()
        except ValueError:
            return f"Unknown({label})"

    @staticmethod
    def validate_for_dataset(
        labels: set[int],
        dataset_name: str,
        num_classes: int,
    ) -> None:
        """
        Validate that labels match expected values for a dataset.

        This is useful for testing and debugging to ensure label remapping
        is working correctly.

        :param set[int] labels: Set of labels found in dataset
        :param str dataset_name: Name of dataset ("mdd", "cane", etc.)
        :param int num_classes: Number of classes (2 or 4)
        :raises ValueError: If labels don't match expected values
        :rtype: None
        """
        if num_classes == 2:
            expected_int = {int(BinaryLabel.HEALTHY), int(BinaryLabel.PATHOLOGICAL)}
            if not labels.issubset(expected_int):
                raise ValueError(
                    f"{dataset_name} in 2-class mode should only have labels {expected_int}, "
                    f"but found: {labels}"
                )
        elif num_classes == 4:
            if dataset_name == "mdd":
                expected_int = {int(CanonicalLabel.HEALTHY), int(CanonicalLabel.DEPRESSION_ONLY)}
                forbidden_int = {int(CanonicalLabel.ANXIETY_ONLY), int(CanonicalLabel.COMORBID)}
                found_forbidden = labels & forbidden_int
                if found_forbidden:
                    raise ValueError(
                        f"MDD in 4-class mode has forbidden labels {found_forbidden}. "
                        f"Expected only {expected_int}. "
                        f"This indicates depression chunks are mislabeled as anxiety/comorbid."
                    )
            elif dataset_name == "cane":
                expected_int = {int(label) for label in CanonicalLabel}
                if not labels.issubset(expected_int):
                    raise ValueError(
                        f"CANE in 4-class mode should only have labels {expected_int}, "
                        f"but found: {labels}"
                    )
            elif dataset_name == "sad":
                expected_int = {int(CanonicalLabel.HEALTHY), int(CanonicalLabel.ANXIETY_ONLY)}
                if not labels.issubset(expected_int):
                    raise ValueError(
                        f"SAD should only have labels {expected_int}, but found: {labels}"
                    )
            elif dataset_name == "ax_malik":
                expected_int = {int(CanonicalLabel.ANXIETY_ONLY)}
                if not labels.issubset(expected_int):
                    raise ValueError(
                        f"AX_MALIK should only have label {expected_int}, but found: {labels}"
                    )


# Display names for API/CLI output and metrics
BINARY_DISPLAY_NAMES: dict[int, str] = {
    BinaryLabel.HEALTHY: "Healthy",
    BinaryLabel.PATHOLOGICAL: "Pathological",
}

CANONICAL_DISPLAY_NAMES: dict[int, str] = {
    CanonicalLabel.HEALTHY: "Healthy",
    CanonicalLabel.ANXIETY_ONLY: "Anxiety",
    CanonicalLabel.DEPRESSION_ONLY: "Depression",
    CanonicalLabel.COMORBID: "Comorbid",
}


def get_display_names(num_classes: int) -> dict[int, str]:
    """
    Return the appropriate display name dict for 2 or 4 classes.

    :param int num_classes: Number of classes (2 or 4).
    :return: Mapping from class index to human-readable display name.
    :rtype: dict[int, str]
    """
    if num_classes == 2:
        return dict(BINARY_DISPLAY_NAMES)
    return dict(CANONICAL_DISPLAY_NAMES)
