"""Data preparation utilities for cross-validation training."""

import logging
from collections.abc import Callable
from typing import Literal, NamedTuple, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

from thesis.dataset import (
    AX_MALIKDataset,
    CANEDataset,
    FlattenedSpectrogramDataset,
    MDDDataset,
    SpectrogramDataset,
)

logger = logging.getLogger(__name__)

EXPECTED_SPECTROGRAM_SHAPE = (129, 41)  # Expected spectrogram shape from training

SubjectList = list[tuple[str, str]]  # Single dataset: (dataset_label, subject_id) tuples
MultiDatasetSubjectList = list[SubjectList]  # Multiple datasets: list of subject lists


class SubjectClasses(NamedTuple):
    """Container for subjects split by diagnostic class."""

    normal: SubjectList
    anxiety: SubjectList
    depression: SubjectList
    anxiety_depression: SubjectList


def determine_num_classes(class_mode: str) -> tuple[int, Optional[dict[int, int]]]:
    """
    Determine number of classes and label mapping based on dataset and class_mode.

    :param class_mode: Class mode (2", or "4")
    :return: Tuple of (num_classes, label_mapping)
        - label_mapping is None for 4-class (use original labels)
        - label_mapping remaps for 2-class: {0:0, 1:1, 2:1, 3:1}
    :rtype: tuple[int, Optional[dict[int, int]]]
    """
    if class_mode == "2":
        # Force binary: healthy vs any-pathological
        return 2, {0: 0, 1: 1, 2: 1, 3: 1}
    elif class_mode == "4":
        return 4, None
    else:
        raise ValueError(f"Invalid class_mode: {class_mode}")


def _prepare_dataset_generic(
    dataset_class: type[MDDDataset] | type[AX_MALIKDataset],
    dataset_label: str,
    conditions: list[str],
    channel: str,
    rng: np.random.RandomState,
    fs: float,
    augmentation: Callable | None = None,
    test_mode: bool = False,
) -> tuple[ConcatDataset | FlattenedSpectrogramDataset, SubjectClasses]:
    """
    Generic dataset preparation for datasets following the MDD pattern.

    Used by prepare_mdd_dataset and prepare_ax_malik_dataset to reduce code duplication.

    :param type dataset_class: Dataset class to instantiate (MDDDataset or AX_MALIKDataset).
    :param str dataset_label: Label for dataset ("mdd" or "ax_malik").
    :param list[str] conditions: EEG conditions to load.
    :param str channel: Channel to use (e.g., "Fp1" or "all").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :param float fs: Sampling frequency for STFT.
    :param Callable | None augmentation: Optional augmentation to apply to raw EEG.
    :param bool test_mode: If True, load only one file per class for debugging.
    :return: Tuple of (flat_dataset, SubjectClasses).
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset, SubjectClasses]
    """
    all_flat_datasets = []
    all_subjects = []

    for condition in conditions:
        # Instantiate dataset
        dataset = dataset_class(
            condition=condition,
            channel=channel,
            test_mode=test_mode,
        )

        # Create spectrogram dataset
        spec_dataset = SpectrogramDataset(dataset, fs=fs, augmentation=augmentation)
        flat_dataset = FlattenedSpectrogramDataset(spec_dataset)

        # Validate shape
        spec_shape = flat_dataset[0][0].shape[1:]
        assert spec_shape == torch.Size(list(EXPECTED_SPECTROGRAM_SHAPE)), (
            f"Expected (129, 41), got {spec_shape}. "
            f"The neural net was designed using this assumption."
        )

        all_flat_datasets.append(flat_dataset)
        all_subjects.extend(dataset.get_sorted_subjects())

    # Combine datasets if multiple conditions
    if len(all_flat_datasets) == 1:
        combined_flat_dataset = all_flat_datasets[0]
    else:
        combined_flat_dataset = ConcatDataset(all_flat_datasets)

    subject_classes = split_subjects_into_classes(all_subjects, dataset_label, rng)
    return combined_flat_dataset, subject_classes


def prepare_mdd_dataset(
    conditions: list[Literal["EC", "EO", "TASK"]],
    channel: str,
    rng: np.random.RandomState,
    augmentation: Callable | None = None,
    test_mode: bool = False,
) -> tuple[ConcatDataset | FlattenedSpectrogramDataset, SubjectClasses]:
    """
    Prepare MDD dataset with spectrograms and split subjects by class.

    :param list[str] conditions: EEG conditions to load (e.g., ["EC", "EO"]).
    :param str channel: Channel to use (e.g., "Fp1" or "all").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :param Callable | None augmentation: Optional augmentation to apply to raw EEG.
    :param bool test_mode: If True, load only one file per class for debugging.
    :return: Tuple of (flat_dataset, SubjectClasses).
             Note: anxiety and anxiety_depression are empty for MDD dataset.
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset, SubjectClasses]
    """
    return _prepare_dataset_generic(
        dataset_class=MDDDataset,
        dataset_label="mdd",
        conditions=conditions,
        channel=channel,
        rng=rng,
        fs=MDDDataset.FS,
        augmentation=augmentation,
        test_mode=test_mode,
    )


def prepare_cane_dataset(
    conditions: list[Literal["ec", "eo"]],
    channel: str,
    rng: np.random.RandomState,
    label_mapping: Optional[dict[int, int]] = None,
    skip_artifact_removal: bool = False,
    augmentation: Callable | None = None,
    test_mode: bool = False,
) -> tuple[ConcatDataset | FlattenedSpectrogramDataset, SubjectClasses]:
    """
    Prepare CANE dataset with spectrograms and split subjects by class.

    :param list[str] conditions: EEG conditions to load (e.g., ["ec", "eo"]).
    :param str: channel: Channel to use (e.g., "Fp1" or "all").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :param Optional[dict[int, int]] label_mapping: Optional label remapping
           (e.g., {0:0, 1:1, 2:1, 3:1}).
    :param bool skip_artifact_removal: If True, skip artifact interpolation and clipping.
    :param Callable | None augmentation: Optional augmentation to apply to raw EEG.
    :param bool test_mode: If True, load only one file per class for debugging.
    :return: Tuple of (flat_dataset, SubjectClasses).
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset, SubjectClasses]
    """
    all_flat_datasets = []
    all_subjects = []

    for condition in conditions:
        cane_dataset = CANEDataset(
            condition=condition,
            channel=channel,
            skip_extreme_artifacts=True,
            skip_artifact_removal=skip_artifact_removal,
            test_mode=test_mode,
        )
        cane_spec_dataset = SpectrogramDataset(
            cane_dataset,
            fs=CANEDataset.FS,
            nperseg=CANEDataset.STFT_NPERSEG,
            noverlap=CANEDataset.STFT_NOVERLAP,
            augmentation=augmentation,
        )

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

    subject_classes = split_subjects_into_classes(all_subjects, "cane", rng)
    return cane_combined_flat, subject_classes


def prepare_ax_malik_dataset(
    conditions: list[Literal["EC", "EO"]],
    channel: str,
    rng: np.random.RandomState,
    augmentation: Callable | None = None,
    test_mode: bool = False,
) -> tuple[ConcatDataset | FlattenedSpectrogramDataset, SubjectClasses]:
    """
    Prepare AX_MALIK dataset with spectrograms and split subjects by class.

    :param list[str] conditions: EEG conditions to load (e.g., ["EC", "EO"]).
    :param str channel: Channel to use (e.g., "Fp1" or "all").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :param Callable | None augmentation: Optional augmentation to apply to raw EEG.
    :param bool test_mode: If True, load only one file per class for debugging.
    :return: Tuple of (flat_dataset, SubjectClasses).
             Note: All subjects are anxiety class; normal, depression, and
             anxiety_depression will be empty.
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset, SubjectClasses]
    """
    return _prepare_dataset_generic(
        dataset_class=AX_MALIKDataset,
        dataset_label="ax_malik",
        conditions=conditions,
        channel=channel,
        rng=rng,
        fs=AX_MALIKDataset.FS,
        augmentation=augmentation,
        test_mode=test_mode,
    )


def split_subjects_into_classes(
    subjects: list[str], dataset_label: str, rng: np.random.RandomState
) -> SubjectClasses:
    """
    Split subjects into normal, anxiety, depression, and anxiety+depression classes.

    :param list[str] subjects: List of subject IDs.
    :param str dataset_label: Dataset label ("mdd" or "cane").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :return: SubjectClasses containing subjects grouped by diagnostic class.
    :rtype: SubjectClasses
    """
    normal: SubjectList = []
    anxiety: SubjectList = []
    depression: SubjectList = []
    anxiety_depression: SubjectList = []

    for subj in subjects:
        if subj.startswith("H "):
            normal.append((dataset_label, subj))
        elif subj.startswith("AX "):
            anxiety.append((dataset_label, subj))
        elif subj.startswith("DEP "):
            depression.append((dataset_label, subj))
        elif subj.startswith("AXDEP "):
            anxiety_depression.append((dataset_label, subj))
        elif subj.startswith("MDD "):
            depression.append((dataset_label, subj))

    # Shuffle each class independently
    for class_list in [normal, anxiety, depression, anxiety_depression]:
        if len(class_list) > 0:
            rng.shuffle(class_list)

    return SubjectClasses(
        normal=normal,
        anxiety=anxiety,
        depression=depression,
        anxiety_depression=anxiety_depression,
    )


def split_into_folds(subjects: SubjectList, n_folds: int) -> list[SubjectList]:
    """
    Split subjects into n folds, keeping EC/EO recordings together.

    :param list subjects: List of (dataset_label, subject_id) tuples.
    :param int n_folds: Number of folds.
    :return: List of folds, each containing subject tuples.
    :rtype: list
    """
    mapping: dict[str, SubjectList] = {}
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


def _merge_dataset_folds(
    multi_dataset_subjects: MultiDatasetSubjectList, n_folds: int
) -> list[SubjectList]:
    """
    Split subjects from multiple datasets into folds and merge corresponding folds.

    :param MultiDatasetSubjectList multi_dataset_subjects: List of subject lists from multiple
           datasets, where each inner list contains (dataset_label, subject_id) tuples.
    :param int n_folds: Number of folds to create.
    :return: List of merged folds, where each fold contains subjects from all datasets.
    :rtype: list[SubjectList]
    """
    merged_folds: list[SubjectList] = [[] for _ in range(n_folds)]
    for dataset_subjects in multi_dataset_subjects:
        if len(dataset_subjects) > 0:
            tmp_folds = split_into_folds(dataset_subjects, n_folds)
            for i, fold in enumerate(tmp_folds):
                merged_folds[i].extend(fold)
    return merged_folds


def create_balanced_folds(
    normal: MultiDatasetSubjectList,
    anxiety: MultiDatasetSubjectList,
    depression: MultiDatasetSubjectList,
    anxiety_depression: MultiDatasetSubjectList,
    n_folds: int,
) -> tuple[
    list[SubjectList],  # normal_folds
    list[SubjectList],  # anxiety_folds
    list[SubjectList],  # depression_folds
    list[SubjectList],  # anxiety_depression_folds
]:
    """
    Create balanced folds ensuring both datasets appear in each fold.

    Splits each dataset separately into n_folds, then merges corresponding folds.
    Groups subjects by base ID to keep EC/EO recordings together.

    :param list normal: List of lists of tuples (dataset_label, subject_id) from all datasets'
           normal subjects. Each inner list represents one dataset.
    :param list anxiety: List of lists of tuples (dataset_label, subject_id) from all datasets'
           anxiety subjects. Each inner list represents one dataset.
    :param list depression: List of lists of tuples (dataset_label, subject_id) from all datasets'
           depression subjects. Each inner list represents one dataset.
    :param list anxiety_depression: List of lists of tuples (dataset_label, subject_id) from all
           datasets' anxiety+depression subjects. Each inner list represents one dataset.
    :param int n_folds: Number of folds to create.
    :return: Tuple of (normal_folds, anxiety_folds, depression_folds, anxiety_depression_folds).
    :rtype: tuple
    """
    # Create separate folds for each class from both datasets
    normal_folds = _merge_dataset_folds(normal, n_folds)
    anxiety_folds = _merge_dataset_folds(anxiety, n_folds)
    depression_folds = _merge_dataset_folds(depression, n_folds)
    anxiety_depression_folds = _merge_dataset_folds(anxiety_depression, n_folds)

    # Count total subjects for each class
    total_normal = sum(len(dataset_subjects) for dataset_subjects in normal)
    total_anxiety = sum(len(dataset_subjects) for dataset_subjects in anxiety)
    total_depression = sum(len(dataset_subjects) for dataset_subjects in depression)
    total_anxiety_depression = sum(len(dataset_subjects) for dataset_subjects in anxiety_depression)

    logger.info(f"Created {n_folds} balanced folds:")
    logger.info(f"  Normal subjects per fold: ~{total_normal / n_folds:.1f}")
    logger.info(f"  Anxiety subjects per fold: ~{total_anxiety / n_folds:.1f}")
    logger.info(f"  Depression subjects per fold: ~{total_depression / n_folds:.1f}")
    logger.info(
        f"  Anxiety+Depression subjects per fold: ~{total_anxiety_depression / n_folds:.1f}"
    )

    return normal_folds, anxiety_folds, depression_folds, anxiety_depression_folds


def get_indices_from_dataset(
    concat_dataset: ConcatDataset | FlattenedSpectrogramDataset, subject_list: list[str]
) -> list[int]:
    """
    Get indices for subjects from a ConcatDataset or FlattenedSpectrogramDataset.

    :param ConcatDataset | FlattenedSpectrogramDataset concat_dataset: Dataset containing
        EEG data, either concatenated or flattened.
    :param list[str] subject_list: List of subject IDs.
    :return: List of indices in the dataset.
    :rtype: list[int]
    """
    # Handle FlattenedSpectrogramDataset directly
    if isinstance(concat_dataset, FlattenedSpectrogramDataset):
        return concat_dataset.get_indices_for_subjects(subject_list)

    # Handle ConcatDataset by iterating through subdatasets
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
    normal_folds: list[SubjectList],
    anxiety_folds: list[SubjectList],
    depression_folds: list[SubjectList],
    anxiety_depression_folds: list[SubjectList],
    dataset_type: str,
    mdd_flat_dataset: ConcatDataset | FlattenedSpectrogramDataset | None,
    cane_flat_dataset: ConcatDataset | FlattenedSpectrogramDataset | None,
    ax_malik_flat_dataset: ConcatDataset | FlattenedSpectrogramDataset | None,
    flat_dataset: ConcatDataset | FlattenedSpectrogramDataset | None,
) -> tuple[ConcatDataset | Subset, ConcatDataset | Subset]:
    """
    Get train and validation datasets for a specific fold.

    :param int fold: Fold index.
    :param list normal_folds: Normal subject folds.
    :param list anxiety_folds: Anxiety subject folds.
    :param list depression_folds: Depression subject folds.
    :param list anxiety_depression_folds: Anxiety+depression subject folds.
    :param str dataset_type: Dataset type ("mdd", "cane", "ax_malik", or "all").
    :param mdd_flat_dataset: MDD flattened dataset (for "all" mode).
    :param cane_flat_dataset: CANE flattened dataset (for "all" mode).
    :param ax_malik_flat_dataset: AX_MALIK flattened dataset (for "all" mode).
    :param flat_dataset: Single dataset (for "mdd", "cane", or "ax_malik" mode).
    :return: Tuple of (train_dataset, val_dataset).
    :rtype: tuple
    """
    # Determine train/val subjects for this fold from all classes
    val_subjects_with_dataset: SubjectList = []
    train_subjects_with_dataset: SubjectList = []

    # Collect validation subjects from all 4 classes
    val_subjects_with_dataset.extend(normal_folds[fold])
    val_subjects_with_dataset.extend(anxiety_folds[fold])
    val_subjects_with_dataset.extend(depression_folds[fold])
    val_subjects_with_dataset.extend(anxiety_depression_folds[fold])

    # Collect training subjects from all other folds (all 4 classes)
    for i in range(len(normal_folds)):
        if i != fold:
            train_subjects_with_dataset.extend(normal_folds[i])
            train_subjects_with_dataset.extend(anxiety_folds[i])
            train_subjects_with_dataset.extend(depression_folds[i])
            train_subjects_with_dataset.extend(anxiety_depression_folds[i])

    # Log subject distribution
    val_subjects_clean = [f"{ds}:{subj}" for ds, subj in val_subjects_with_dataset]
    train_subjects_clean = [f"{ds}:{subj}" for ds, subj in train_subjects_with_dataset]
    logger.info(f"Train subjects ({len(train_subjects_clean)}): {train_subjects_clean[:100]}...")
    logger.info(f"Val subjects ({len(val_subjects_clean)}): {val_subjects_clean}")

    # Get indices for train/val based on subjects (chunk-level indices)
    # Need to handle combined dataset differently
    if dataset_type == "all":
        # Combine all available datasets
        train_subsets = []
        val_subsets = []
        train_chunk_counts = {}
        val_chunk_counts = {}

        # Process MDD dataset if available
        if mdd_flat_dataset is not None:
            mdd_train_subjects = [subj for ds, subj in train_subjects_with_dataset if ds == "mdd"]
            mdd_val_subjects = [subj for ds, subj in val_subjects_with_dataset if ds == "mdd"]

            mdd_train_indices = get_indices_from_dataset(
                mdd_flat_dataset, mdd_train_subjects
            )
            mdd_val_indices = get_indices_from_dataset(mdd_flat_dataset, mdd_val_subjects)

            train_subsets.append(Subset(mdd_flat_dataset, mdd_train_indices))
            val_subsets.append(Subset(mdd_flat_dataset, mdd_val_indices))
            train_chunk_counts["MDD"] = len(mdd_train_indices)
            val_chunk_counts["MDD"] = len(mdd_val_indices)

        # Process CANE dataset if available
        if cane_flat_dataset is not None:
            cane_train_subjects = [subj for ds, subj in train_subjects_with_dataset if ds == "cane"]
            cane_val_subjects = [subj for ds, subj in val_subjects_with_dataset if ds == "cane"]

            cane_train_indices = get_indices_from_dataset(
                cane_flat_dataset, cane_train_subjects
            )
            cane_val_indices = get_indices_from_dataset(cane_flat_dataset, cane_val_subjects)

            train_subsets.append(Subset(cane_flat_dataset, cane_train_indices))
            val_subsets.append(Subset(cane_flat_dataset, cane_val_indices))
            train_chunk_counts["CANE"] = len(cane_train_indices)
            val_chunk_counts["CANE"] = len(cane_val_indices)

        # Process AX_MALIK dataset if available
        if ax_malik_flat_dataset is not None:
            ax_malik_train_subjects = [
                subj for ds, subj in train_subjects_with_dataset if ds == "ax_malik"
            ]
            ax_malik_val_subjects = [
                subj for ds, subj in val_subjects_with_dataset if ds == "ax_malik"
            ]

            ax_malik_train_indices = get_indices_from_dataset(
                ax_malik_flat_dataset, ax_malik_train_subjects
            )
            ax_malik_val_indices = get_indices_from_dataset(
                ax_malik_flat_dataset, ax_malik_val_subjects
            )

            train_subsets.append(Subset(ax_malik_flat_dataset, ax_malik_train_indices))
            val_subsets.append(Subset(ax_malik_flat_dataset, ax_malik_val_indices))
            train_chunk_counts["AX_MALIK"] = len(ax_malik_train_indices)
            val_chunk_counts["AX_MALIK"] = len(ax_malik_val_indices)

        train_dataset = ConcatDataset(train_subsets)
        val_dataset = ConcatDataset(val_subsets)

        # Log chunk counts
        train_count_str = ", ".join([f"{k}={v}" for k, v in train_chunk_counts.items()])
        val_count_str = ", ".join([f"{k}={v}" for k, v in val_chunk_counts.items()])
        logger.info(f"Training chunks: {train_count_str}, Total={len(train_dataset)}")
        logger.info(f"Validation chunks: {val_count_str}, Total={len(val_dataset)}")
    else:
        assert flat_dataset is not None
        # Single dataset: extract just the subject names
        train_subjects = [subj for ds, subj in train_subjects_with_dataset]
        val_subjects = [subj for ds, subj in val_subjects_with_dataset]

        # Handle case where dataset might be ConcatDataset (multiple conditions)
        if isinstance(flat_dataset, ConcatDataset):
            train_indices = get_indices_from_dataset(flat_dataset, train_subjects)
            val_indices = get_indices_from_dataset(flat_dataset, val_subjects)
        else:
            train_indices = flat_dataset.get_indices_for_subjects(train_subjects)
            val_indices = flat_dataset.get_indices_for_subjects(val_subjects)

        logger.info(f"Training chunks: {len(train_indices)}")
        logger.info(f"Validation chunks: {len(val_indices)}")

        # Create Subset datasets (reusing precomputed spectrograms!)
        train_dataset = Subset(flat_dataset, train_indices)
        val_dataset = Subset(flat_dataset, val_indices)

    return train_dataset, val_dataset
