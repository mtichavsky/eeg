"""Data preparation utilities for cross-validation training."""

import logging
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

from thesis.dataset import (
    CANEDataset,
    FlattenedSpectrogramDataset,
    MDDDataset,
    SpectrogramDataset,
)

logger = logging.getLogger(__name__)

EXPECTED_SPECTROGRAM_SHAPE = (129, 41)  # Expected spectrogram shape from training


def prepare_mdd_dataset(
    conditions: list[Literal["EC", "EO", "TASK"]],
    skip_ica: bool,
    channel: str | None,
    rng: np.random.RandomState,
    augmentation: Callable | None = None,
) -> tuple[
    ConcatDataset | FlattenedSpectrogramDataset,
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """
    Prepare MDD dataset with spectrograms and split subjects by class.

    :param list[str] conditions: EEG conditions to load (e.g., ["EC", "EO"]).
    :param bool skip_ica: Whether to skip ICA preprocessing.
    :param str | None channel: Single channel to use (e.g., "Fp1").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :param Callable | None augmentation: Optional augmentation to apply to raw EEG.
    :return: Tuple of (flat_dataset, normal_subjects, depressed_subjects, anxious_subjects).
    :rtype: tuple
    """
    all_flat_datasets = []
    all_subjects = []

    for condition in conditions:
        mdd_dataset = MDDDataset(condition=condition, skip_ica=skip_ica, channel=channel)
        mdd_spec_dataset = SpectrogramDataset(
            mdd_dataset, fs=MDDDataset.FS, augmentation=augmentation
        )
        mdd_flat_dataset = FlattenedSpectrogramDataset(mdd_spec_dataset)

        spec_shape = mdd_flat_dataset[0][0].shape[1:]
        assert spec_shape == torch.Size(list(EXPECTED_SPECTROGRAM_SHAPE)), (
            f"Expected (129, 41), got {spec_shape}. "
            f"The neural net was designed using this assumption."
        )

        all_flat_datasets.append(mdd_flat_dataset)
        all_subjects.extend(mdd_dataset.get_sorted_subjects())

    # Combine datasets if multiple conditions
    if len(all_flat_datasets) == 1:
        combined_flat_dataset = all_flat_datasets[0]
    else:
        combined_flat_dataset = ConcatDataset(all_flat_datasets)

    mdd_normal, mdd_depressed, mdd_anxious = split_subjects_into_classes(all_subjects, "mdd", rng)
    return combined_flat_dataset, mdd_normal, mdd_depressed, mdd_anxious


def prepare_cane_dataset(
    conditions: list[Literal["ec", "eo"]],
    channel: str | None,
    rng: np.random.RandomState,
    remap_labels: bool = False,
    skip_artifact_removal: bool = False,
    augmentation: Callable | None = None,
) -> tuple[
    ConcatDataset | FlattenedSpectrogramDataset,
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    """
    Prepare CANE dataset with spectrograms and split subjects by class.

    :param list[str] conditions: EEG conditions to load (e.g., ["ec", "eo"]).
    :param str | None channel: Single channel to use (e.g., "Fp1").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :param bool remap_labels: If True, remap labels {0->0, 2->1} for binary classification.
    :param bool skip_artifact_removal: If True, skip artifact interpolation and clipping.
    :param Callable | None augmentation: Optional augmentation to apply to raw EEG.
    :return: Tuple of (flat_dataset, normal_subjects, depressed_subjects, anxious_subjects).
    :rtype: tuple
    """
    all_flat_datasets = []
    all_subjects = []

    for condition in conditions:
        cane_dataset = CANEDataset(
            condition=condition,
            channel=channel,
            skip_extreme_artifacts=True,
            skip_artifact_removal=skip_artifact_removal,
        )
        cane_spec_dataset = SpectrogramDataset(
            cane_dataset,
            fs=CANEDataset.FS,
            nperseg=CANEDataset.STFT_NPERSEG,
            noverlap=CANEDataset.STFT_NOVERLAP,
            augmentation=augmentation,
        )

        # Apply label remapping if training on CANE alone (binary classification)
        label_mapping = {0: 0, 2: 1} if remap_labels else None
        cane_flat_dataset = FlattenedSpectrogramDataset(
            cane_spec_dataset, label_mapping=label_mapping
        )

        spec_shape = cane_flat_dataset[0][0].shape[1:]
        assert spec_shape == torch.Size(list(EXPECTED_SPECTROGRAM_SHAPE)), (
            f"Expected (129, 41), got {spec_shape}. "
            f"The neural net was designed using this assumption."
        )

        all_flat_datasets.append(cane_flat_dataset)
        all_subjects.extend(cane_dataset.get_sorted_subjects())

    # Combine datasets if multiple conditions
    if len(all_flat_datasets) == 1:
        cane_combined_flat = all_flat_datasets[0]
    else:
        cane_combined_flat = ConcatDataset(all_flat_datasets)

    cane_normal, cane_depressed, cane_anxious = split_subjects_into_classes(
        all_subjects, "cane", rng
    )
    return cane_combined_flat, cane_normal, cane_depressed, cane_anxious


def split_subjects_into_classes(
    subjects: list[str], dataset_label: str, rng: np.random.RandomState
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Split subjects into normal, depressed, and anxious classes.

    :param list[str] subjects: List of subject IDs.
    :param str dataset_label: Dataset label ("mdd" or "cane").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :return: Tuple of (normal, depressed, anxious) subject lists.
    :rtype: tuple
    """
    normal: list[tuple[str, str]] = []
    depressed: list[tuple[str, str]] = []
    anxious: list[tuple[str, str]] = []
    for subj in subjects:
        if subj.startswith("H "):
            normal.append((dataset_label, subj))
        elif subj.startswith("MDD "):
            depressed.append((dataset_label, subj))
        elif subj.startswith("AX "):
            anxious.append((dataset_label, subj))

    if len(normal) > 0:
        rng.shuffle(normal)
    if len(depressed) > 0:
        rng.shuffle(depressed)
    if len(anxious) > 0:
        rng.shuffle(anxious)
    return normal, depressed, anxious


def split_into_folds(subjects: list[tuple[str, str]], n_folds: int) -> list[list[tuple[str, str]]]:
    """
    Split subjects into n folds, keeping EC/EO recordings together.

    :param list subjects: List of (dataset_label, subject_id) tuples.
    :param int n_folds: Number of folds.
    :return: List of folds, each containing subject tuples.
    :rtype: list
    """
    mapping: dict[str, list[tuple[str, str]]] = {}
    for dataset, subject in subjects:
        subject_parts = subject.split()
        base = " ".join(subject_parts[:-1])
        if base not in mapping:
            mapping[base] = [(dataset, subject)]
        else:
            mapping[base].append((dataset, subject))

    # Use unique base subjects only (no duplicates)
    base_subjects = list(mapping.keys())
    folds = np.array_split(base_subjects, n_folds)
    folds_with_conditions = [[s for base in split for s in mapping[base]] for split in folds]
    return folds_with_conditions


def create_balanced_folds(
    mdd_normal: list[tuple[str, str]],
    mdd_depressed: list[tuple[str, str]],
    cane_normal: list[tuple[str, str]],
    cane_depressed: list[tuple[str, str]],
    cane_anxious: list[tuple[str, str]],
    n_folds: int,
) -> tuple[list[list[tuple[str, str]]], list[list[tuple[str, str]]], list[list[tuple[str, str]]]]:
    """
    Create balanced folds ensuring both datasets appear in each fold.

    Splits each dataset separately into n_folds, then merges corresponding folds.
    Groups subjects by base ID to keep EC/EO recordings together.

    :param list mdd_normal: List of tuples (dataset_label, subject_id) from MDD normal
        subjects.
    :param list mdd_depressed: List of tuples (dataset_label, subject_id) from MDD
        depressed subjects.
    :param list cane_normal: List of tuples (dataset_label, subject_id) from CANE normal
        subjects.
    :param list cane_depressed: List of tuples (dataset_label, subject_id) from CANE
        depressed subjects.
    :param list cane_anxious: List of tuples (dataset_label, subject_id) from CANE anxious
        subjects.
    :param int n_folds: Number of folds to create.
    :return: Tuple of (normal_folds, anxious_folds, mdd_folds).
    :rtype: tuple
    """
    empty_folds: list[list[tuple[str, str]]] = [[] for _ in range(n_folds)]

    mdd_normal_folds = split_into_folds(mdd_normal, n_folds)
    mdd_mdd_folds = split_into_folds(mdd_depressed, n_folds)
    cane_normal_folds = split_into_folds(cane_normal, n_folds)
    cane_anxious_folds = split_into_folds(cane_anxious, n_folds)
    cane_mdd_folds = split_into_folds(cane_depressed, n_folds) if cane_depressed else empty_folds

    normal_folds: list[list[tuple[str, str]]] = []
    anxious_folds = cane_anxious_folds
    mdd_folds: list[list[tuple[str, str]]] = []
    for mn, cn, mm, cm in zip(mdd_normal_folds, cane_normal_folds, mdd_mdd_folds, cane_mdd_folds):
        normal_folds.append(mn + cn)
        if len(cm) > 0:
            mdd_folds.append(mm + cm)
        else:
            mdd_folds.append(mm)

    return normal_folds, anxious_folds, mdd_folds


def get_indices_from_concat_dataset(
    concat_dataset: ConcatDataset, subject_list: list[str]
) -> list[int]:
    """
    Get indices for subjects from a ConcatDataset.

    :param ConcatDataset concat_dataset: ConcatDataset containing multiple
        FlattenedSpectrogramDatasets.
    :param list[str] subject_list: List of subject IDs.
    :return: List of indices in the concatenated dataset.
    :rtype: list[int]
    """
    all_indices: list[int] = []
    offset = 0
    for dataset in concat_dataset.datasets:
        dataset_indices = dataset.get_indices_for_subjects(subject_list)
        # Adjust indices by offset in concatenated dataset
        all_indices.extend([idx + offset for idx in dataset_indices])
        offset += len(dataset)
    return all_indices


def get_datasets_for_fold(
    fold: int,
    normal_folds: list[list[tuple[str, str]]],
    mdd_folds: list[list[tuple[str, str]]],
    anxious_folds: list[list[tuple[str, str]]],
    dataset_type: str,
    mdd_flat_dataset: ConcatDataset | FlattenedSpectrogramDataset | None,
    cane_flat_dataset: ConcatDataset | FlattenedSpectrogramDataset | None,
    flat_dataset: ConcatDataset | FlattenedSpectrogramDataset | None,
) -> tuple[ConcatDataset | Subset, ConcatDataset | Subset]:
    """
    Get train and validation datasets for a specific fold.

    :param int fold: Fold index.
    :param list normal_folds: Normal subject folds.
    :param list mdd_folds: MDD subject folds.
    :param list anxious_folds: Anxious subject folds.
    :param str dataset_type: Dataset type ("mdd", "cane", or "both").
    :param mdd_flat_dataset: MDD flattened dataset (for "both" mode).
    :param cane_flat_dataset: CANE flattened dataset (for "both" mode).
    :param flat_dataset: Single dataset (for "mdd" or "cane" mode).
    :return: Tuple of (train_dataset, val_dataset).
    :rtype: tuple
    """
    # Determine train/val subjects for this fold from all classes
    val_subjects_with_dataset: list[tuple[str, str]] = []
    train_subjects_with_dataset: list[tuple[str, str]] = []

    # Collect validation subjects from all classes
    val_subjects_with_dataset.extend(normal_folds[fold])
    val_subjects_with_dataset.extend(mdd_folds[fold])
    val_subjects_with_dataset.extend(anxious_folds[fold])

    # Collect training subjects from all other folds
    for i in range(len(normal_folds)):
        if i != fold:
            train_subjects_with_dataset.extend(normal_folds[i])
            train_subjects_with_dataset.extend(mdd_folds[i])
            train_subjects_with_dataset.extend(anxious_folds[i])

    # Log subject distribution
    val_subjects_clean = [f"{ds}:{subj}" for ds, subj in val_subjects_with_dataset]
    train_subjects_clean = [f"{ds}:{subj}" for ds, subj in train_subjects_with_dataset]
    logger.info(f"Train subjects ({len(train_subjects_clean)}): {train_subjects_clean[:100]}...")
    logger.info(f"Val subjects ({len(val_subjects_clean)}): {val_subjects_clean}")

    # Get indices for train/val based on subjects (chunk-level indices)
    # Need to handle combined dataset differently
    if dataset_type == "both":
        assert mdd_flat_dataset is not None
        assert cane_flat_dataset is not None
        # Split by dataset source
        mdd_train_subjects = [subj for ds, subj in train_subjects_with_dataset if ds == "mdd"]
        mdd_val_subjects = [subj for ds, subj in val_subjects_with_dataset if ds == "mdd"]
        cane_train_subjects = [subj for ds, subj in train_subjects_with_dataset if ds == "cane"]
        cane_val_subjects = [subj for ds, subj in val_subjects_with_dataset if ds == "cane"]

        # Handle case where datasets might be ConcatDataset (multiple conditions)
        if isinstance(mdd_flat_dataset, ConcatDataset):
            mdd_train_indices = get_indices_from_concat_dataset(
                mdd_flat_dataset, mdd_train_subjects
            )
            mdd_val_indices = get_indices_from_concat_dataset(mdd_flat_dataset, mdd_val_subjects)
        else:
            mdd_train_indices = mdd_flat_dataset.get_indices_for_subjects(mdd_train_subjects)
            mdd_val_indices = mdd_flat_dataset.get_indices_for_subjects(mdd_val_subjects)

        if isinstance(cane_flat_dataset, ConcatDataset):
            cane_train_indices = get_indices_from_concat_dataset(
                cane_flat_dataset, cane_train_subjects
            )
            cane_val_indices = get_indices_from_concat_dataset(cane_flat_dataset, cane_val_subjects)
        else:
            cane_train_indices = cane_flat_dataset.get_indices_for_subjects(cane_train_subjects)
            cane_val_indices = cane_flat_dataset.get_indices_for_subjects(cane_val_subjects)

        train_dataset = ConcatDataset(
            [
                Subset(mdd_flat_dataset, mdd_train_indices),
                Subset(cane_flat_dataset, cane_train_indices),
            ]
        )
        val_dataset = ConcatDataset(
            [
                Subset(mdd_flat_dataset, mdd_val_indices),
                Subset(cane_flat_dataset, cane_val_indices),
            ]
        )

        logger.info(
            f"Training chunks: MDD={len(mdd_train_indices)}, CANE={len(cane_train_indices)}, "
            f"Total={len(train_dataset)}"
        )
        logger.info(
            f"Validation chunks: MDD={len(mdd_val_indices)}, CANE={len(cane_val_indices)}, "
            f"Total={len(val_dataset)}"
        )
    else:
        assert flat_dataset is not None
        # Single dataset: extract just the subject names
        train_subjects = [subj for ds, subj in train_subjects_with_dataset]
        val_subjects = [subj for ds, subj in val_subjects_with_dataset]

        # Handle case where dataset might be ConcatDataset (multiple conditions)
        if isinstance(flat_dataset, ConcatDataset):
            train_indices = get_indices_from_concat_dataset(flat_dataset, train_subjects)
            val_indices = get_indices_from_concat_dataset(flat_dataset, val_subjects)
        else:
            train_indices = flat_dataset.get_indices_for_subjects(train_subjects)
            val_indices = flat_dataset.get_indices_for_subjects(val_subjects)

        logger.info(f"Training chunks: {len(train_indices)}")
        logger.info(f"Validation chunks: {len(val_indices)}")

        # Create Subset datasets (reusing precomputed spectrograms!)
        train_dataset = Subset(flat_dataset, train_indices)
        val_dataset = Subset(flat_dataset, val_indices)

    return train_dataset, val_dataset
