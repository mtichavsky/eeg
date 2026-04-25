"""Data preparation utilities for cross-validation training."""

import logging
from collections.abc import Callable
from typing import Literal, NamedTuple, Optional, Sequence, cast

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

from thesis.dataset import (
    CHUNK_DURATION_SEC,
    CANEDataset,
    FlattenedRawEEGDataset,
    FlattenedSpectrogramDataset,
    IDUNDataset,
    MDDDataset,
    SADDataset,
    SpectrogramDataset,
)
from thesis.labels import LabelMapping

# Deformer always expects 250 Hz input; all datasets are resampled/truncated to this rate.
# Use this constant for target_samples = int(chunk_duration * MODEL_FS).
MODEL_FS: int = 250

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
    Determine number of classes and label mapping based on class_mode.

    Note: Returns binary collapse mapping for 2-class mode (collapses all pathological
    classes to 1). Dataset-specific remapping (e.g., MDD depression → canonical position 2)
    is handled separately in prepare_*_dataset() functions.

    :param class_mode: Class mode ("2" or "4")
    :return: Tuple of (num_classes, label_mapping)
        - label_mapping is None for 4-class (use original labels)
        - label_mapping remaps for 2-class: {0:HEALTHY, 1:PATHOLOGICAL, 2:PATHOLOGICAL,
          3:PATHOLOGICAL}
    :rtype: tuple[int, Optional[dict[int, int]]]
    """
    if class_mode == "2":
        # Force binary: healthy vs any-pathological
        # Uses centralized mapping from thesis.labels module
        return 2, LabelMapping.get_canonical_to_binary()
    elif class_mode == "4":
        return 4, None
    else:
        raise ValueError(f"Invalid class_mode: {class_mode}")


def _prepare_dataset_generic(
    dataset_class: type[MDDDataset] | type[SADDataset],
    dataset_label: str,
    conditions: Sequence[str],
    channel: str | None,
    rng: np.random.RandomState,
    fs: float,
    augmentation: Callable | None = None,
    test_mode: bool = False,
    label_mapping: Optional[dict[int, int]] = None,
    use_raw_eeg: bool = False,
    chunk_duration: float = CHUNK_DURATION_SEC,
) -> tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset, SubjectClasses]:
    """
    Generic dataset preparation for datasets following the MDD pattern.

    Used by prepare_mdd_dataset and prepare_sad_dataset to reduce code duplication.

    :param type dataset_class: Dataset class to instantiate (MDDDataset or SADDataset).
    :param str dataset_label: Label for dataset ("mdd" or "sad").
    :param list[str] conditions: EEG conditions to load.
    :param str channel: Channel to use (e.g., "Fp1" or "all").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :param float fs: Sampling frequency for STFT. Ignored when use_raw_eeg=True.
    :param Callable | None augmentation: Optional augmentation to apply to raw EEG.
    :param bool test_mode: If True, load only one file per class for debugging.
    :param Optional[dict[int, int]] label_mapping: Optional label remapping dict
           (e.g., {0: 0, 1: 2} for MDD in 4-class).
    :param bool use_raw_eeg: If True, skip STFT and return FlattenedRawEEGDataset
           for use with raw-EEG models such as Deformer.
    :param float chunk_duration: EEG chunk duration in seconds. Applied only on the raw EEG
           path (use_raw_eeg=True); the spectrogram path always uses CHUNK_DURATION_SEC to
           preserve the expected (129, 41) STFT shape.
    :return: Tuple of (flat_dataset, SubjectClasses).
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset,
            SubjectClasses]
    """
    all_flat_datasets: list = []
    all_subjects = []

    for condition in conditions:
        # For spectrogram models, always use default 10-s chunks to preserve (129, 41) shape.
        effective_chunk_duration = chunk_duration if use_raw_eeg else CHUNK_DURATION_SEC
        # Instantiate dataset
        dataset = dataset_class(
            condition=cast(Literal["EC", "EO", "TASK"], condition),
            channel=channel,
            test_mode=test_mode,
            chunk_duration=effective_chunk_duration,
        )

        if use_raw_eeg:
            target_samples = int(chunk_duration * MODEL_FS)
            flat_dataset: FlattenedSpectrogramDataset | FlattenedRawEEGDataset = (
                FlattenedRawEEGDataset(
                    dataset, target_samples=target_samples, label_mapping=label_mapping
                )
            )
        else:
            # Create spectrogram dataset
            spec_dataset = SpectrogramDataset(
                dataset, fs=fs, augmentation=augmentation, channel=channel
            )
            flat_dataset = FlattenedSpectrogramDataset(spec_dataset, label_mapping=label_mapping)

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
    conditions: Sequence[str],
    channel: str | None,
    rng: np.random.RandomState,
    augmentation: Callable | None = None,
    test_mode: bool = False,
    num_classes: int = 2,
    label_mapping: Optional[dict[int, int]] = None,
    use_raw_eeg: bool = False,
    chunk_duration: float = CHUNK_DURATION_SEC,
) -> tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset, SubjectClasses]:
    """
    Prepare MDD dataset for classification.

    MDDDataset now returns CanonicalLabel instances directly (NORMAL=0, DEPRESSION_ONLY=2).
    No remapping needed for 4-class mode; binary mode applies CanonicalLabel → BinaryLabel.

    :param list[str] conditions: EEG conditions to load (e.g., ["EC", "EO"]).
    :param str channel: Channel to use (e.g., "Fp1" or "all").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :param Callable | None augmentation: Optional augmentation to apply to raw EEG.
    :param bool test_mode: If True, load only one file per class for debugging.
    :param int num_classes: Number of classes (2 or 4).
    :param Optional[dict[int, int]] label_mapping: Optional label remapping dict.
           If None, automatically applies CanonicalLabel → BinaryLabel mapping for 2-class mode.
    :param bool use_raw_eeg: If True, return FlattenedRawEEGDataset instead of spectrogram
           dataset. For use with raw-EEG models such as Deformer.
    :param float chunk_duration: EEG chunk duration in seconds (raw EEG path only).
    :return: Tuple of (flat_dataset, SubjectClasses).
             Note: anxiety and anxiety_depression are empty for MDD dataset.
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset,
            SubjectClasses]
    """
    # In binary mode, apply CanonicalLabel → BinaryLabel mapping
    effective_label_mapping = label_mapping
    if label_mapping is None and num_classes == 2:
        effective_label_mapping = LabelMapping.get_canonical_to_binary()

    return _prepare_dataset_generic(
        dataset_class=MDDDataset,
        dataset_label="mdd",
        conditions=conditions,
        channel=channel,
        rng=rng,
        fs=MDDDataset.FS,
        augmentation=augmentation,
        test_mode=test_mode,
        label_mapping=effective_label_mapping,
        use_raw_eeg=use_raw_eeg,
        chunk_duration=chunk_duration,
    )


def prepare_cane_dataset(
    conditions: Sequence[str],
    channel: str | None,
    rng: np.random.RandomState,
    label_mapping: Optional[dict[int, int]] = None,
    skip_artifact_removal: bool = False,
    augmentation: Callable | None = None,
    test_mode: bool = False,
    use_raw_eeg: bool = False,
    chunk_duration: float = CHUNK_DURATION_SEC,
) -> tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset, SubjectClasses]:
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
    :param bool use_raw_eeg: If True, return FlattenedRawEEGDataset (skips STFT).
           CANE is 500 Hz so native chunks are 2× the target_samples; FlattenedRawEEGDataset
           downsamples 2:1 automatically.
    :param float chunk_duration: EEG chunk duration in seconds (raw EEG path only).
    :return: Tuple of (flat_dataset, SubjectClasses).
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset,
            SubjectClasses]
    """
    all_flat_datasets: list = []
    all_subjects = []

    for condition in conditions:
        # For spectrogram models, always use default 10-s chunks to preserve (129, 41) shape.
        effective_chunk_duration = chunk_duration if use_raw_eeg else CHUNK_DURATION_SEC
        cane_dataset = CANEDataset(
            condition=cast(Literal["ec", "eo"], condition),
            channel=channel,
            skip_extreme_artifacts=True,
            skip_artifact_removal=skip_artifact_removal,
            test_mode=test_mode,
            chunk_duration=effective_chunk_duration,
        )

        if use_raw_eeg:
            target_samples = int(chunk_duration * MODEL_FS)
            cane_flat_dataset: FlattenedSpectrogramDataset | FlattenedRawEEGDataset = (
                FlattenedRawEEGDataset(
                    cane_dataset, target_samples=target_samples, label_mapping=label_mapping
                )
            )
        else:
            cane_spec_dataset = SpectrogramDataset(
                cane_dataset,
                fs=CANEDataset.FS,
                nperseg=CANEDataset.STFT_NPERSEG,
                noverlap=CANEDataset.STFT_NOVERLAP,
                augmentation=augmentation,
                channel=channel,
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


def prepare_sad_dataset(
    conditions: Sequence[str],
    channel: str | None,
    rng: np.random.RandomState,
    augmentation: Callable | None = None,
    test_mode: bool = False,
    num_classes: int = 2,
    label_mapping: Optional[dict[int, int]] = None,
    use_raw_eeg: bool = False,
    chunk_duration: float = CHUNK_DURATION_SEC,
) -> tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset, SubjectClasses]:
    """
    Prepare SAD dataset with spectrograms and split subjects by class.

    :param list[str] conditions: EEG conditions to load (e.g., ["EC", "EO"]).
    :param str channel: Channel to use (e.g., "Fp1" or "all").
    :param np.random.RandomState rng: Random number generator for shuffling.
    :param Callable | None augmentation: Optional augmentation to apply to raw EEG.
    :param bool test_mode: If True, load only one file per class for debugging.
    :param int num_classes: Number of classes (2 or 4).
    :param Optional[dict[int, int]] label_mapping: Optional label remapping dict.
    :param bool use_raw_eeg: If True, return FlattenedRawEEGDataset (skips STFT).
    :param float chunk_duration: EEG chunk duration in seconds (raw EEG path only).
    :return: Tuple of (flat_dataset, SubjectClasses).
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset,
            SubjectClasses]
    """
    return _prepare_dataset_generic(
        dataset_class=SADDataset,
        dataset_label="sad",
        conditions=conditions,
        channel=channel,
        rng=rng,
        fs=SADDataset.FS,
        augmentation=augmentation,
        test_mode=test_mode,
        label_mapping=label_mapping,
        use_raw_eeg=use_raw_eeg,
        chunk_duration=chunk_duration,
    )


def prepare_idun_dataset(
    conditions: Sequence[str],
    channel: str | None,
    rng: np.random.RandomState,
    label_mapping: Optional[dict[int, int]] = None,
    augmentation: Callable | None = None,
    test_mode: bool = False,
    quality_threshold: float = 30.0,
    use_raw_eeg: bool = False,
    chunk_duration: float = CHUNK_DURATION_SEC,
) -> tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset, "SubjectClasses"]:
    """
    Prepare IDUN real in-ear dataset with spectrograms and split subjects by class.

    Uses ``dataset_label="cane"`` so that ``get_datasets_for_fold()`` works unchanged
    (IDUN occupies the CANE slot in balanced folds).

    :param list[str] conditions: EEG conditions to load (e.g., ["ec", "eo"]).
    :param str channel: Channel to use (should be "in-ear" for IDUN).
    :param np.random.RandomState rng: Random number generator for shuffling.
    :param Optional[dict[int, int]] label_mapping: Optional label remapping dict.
    :param Callable | None augmentation: Optional augmentation to apply to raw EEG.
    :param bool test_mode: If True, load only one file per class for debugging.
    :param float quality_threshold: Reject IDUN chunks with quality at or below this value.
    :param bool use_raw_eeg: If True, return FlattenedRawEEGDataset (skips STFT).
    :param float chunk_duration: EEG chunk duration in seconds (raw EEG path only).
    :return: Tuple of (flat_dataset, SubjectClasses).
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset,
            SubjectClasses]
    """
    all_flat_datasets: list = []
    all_subjects = []

    for condition in conditions:
        # For spectrogram models, always use default 10-s chunks to preserve (129, 41) shape.
        effective_chunk_duration = chunk_duration if use_raw_eeg else CHUNK_DURATION_SEC
        idun_dataset = IDUNDataset(
            condition=cast(Literal["ec", "eo"], condition),
            quality_threshold=quality_threshold,
            test_mode=test_mode,
            chunk_duration=effective_chunk_duration,
        )

        if use_raw_eeg:
            target_samples = int(chunk_duration * MODEL_FS)
            idun_flat_dataset: FlattenedSpectrogramDataset | FlattenedRawEEGDataset = (
                FlattenedRawEEGDataset(
                    idun_dataset, target_samples=target_samples, label_mapping=label_mapping
                )
            )
        else:
            idun_spec_dataset = SpectrogramDataset(
                idun_dataset,
                fs=IDUNDataset.FS,
                nperseg=256,
                noverlap=192,
                augmentation=augmentation,
                channel=channel,
            )
            idun_flat_dataset = FlattenedSpectrogramDataset(
                idun_spec_dataset, label_mapping=label_mapping
            )
            spec_shape = idun_flat_dataset[0][0].shape[1:]
            assert spec_shape == torch.Size(list(EXPECTED_SPECTROGRAM_SHAPE)), (
                f"Expected (129, 41), got {spec_shape}. "
                f"The neural net was designed using this assumption."
            )

        all_flat_datasets.append(idun_flat_dataset)
        all_subjects.extend(idun_dataset.get_sorted_subjects())

    # Combine datasets if multiple conditions
    if len(all_flat_datasets) == 1:
        combined_flat = all_flat_datasets[0]
    else:
        combined_flat = ConcatDataset(all_flat_datasets)

    # Use dataset_label="cane" so IDUN occupies the CANE slot in balanced folds
    subject_classes = split_subjects_into_classes(all_subjects, "cane", rng)
    return combined_flat, subject_classes


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
    concat_dataset: ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset,
    subject_list: list[str],
) -> list[int]:
    """
    Get indices for subjects from a flat or concatenated dataset.

    Handles FlattenedSpectrogramDataset, FlattenedRawEEGDataset, and ConcatDataset
    (which may contain any of the above).

    :param concat_dataset: Dataset containing EEG data, either flat or concatenated.
    :param list[str] subject_list: List of subject IDs.
    :return: List of indices in the dataset.
    :rtype: list[int]
    """
    # Handle leaf flat datasets directly
    if isinstance(concat_dataset, (FlattenedSpectrogramDataset, FlattenedRawEEGDataset)):
        return concat_dataset.get_indices_for_subjects(subject_list)

    # Handle ConcatDataset by iterating through subdatasets
    all_indices: list[int] = []
    offset = 0
    for dataset in concat_dataset.datasets:
        flat_ds = cast(FlattenedSpectrogramDataset | FlattenedRawEEGDataset, dataset)
        dataset_indices = flat_ds.get_indices_for_subjects(subject_list)
        # Adjust indices by offset in concatenated dataset
        all_indices.extend([idx + offset for idx in dataset_indices])
        offset += len(flat_ds)
    return all_indices


def get_datasets_for_fold(
    fold: int,
    normal_folds: list[SubjectList],
    anxiety_folds: list[SubjectList],
    depression_folds: list[SubjectList],
    anxiety_depression_folds: list[SubjectList],
    dataset_type: str,
    mdd_flat_dataset: ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset | None,
    cane_flat_dataset: ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset | None,
    sad_flat_dataset: ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset | None,
    flat_dataset: ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset | None,
) -> tuple[ConcatDataset | Subset, ConcatDataset | Subset, dict[str, str]]:
    """
    Get train and validation datasets for a specific fold.

    :param int fold: Fold index.
    :param list normal_folds: Normal subject folds.
    :param list anxiety_folds: Anxiety subject folds.
    :param list depression_folds: Depression subject folds.
    :param list anxiety_depression_folds: Anxiety+depression subject folds.
    :param str dataset_type: Dataset type ("mdd", "cane", "sad", or "all").
    :param mdd_flat_dataset: MDD flattened dataset (for "all" mode).
    :param cane_flat_dataset: CANE flattened dataset (for "all" mode).
    :param sad_flat_dataset: SAD flattened dataset (for "all" mode).
    :param flat_dataset: Single dataset (for "mdd", "cane", or "sad" mode).
    :return: Tuple of (train_dataset, val_dataset, val_subject_dataset_map).
        val_subject_dataset_map maps subject IDs to their dataset label (e.g., "mdd", "cane").
    :rtype: tuple[ConcatDataset | Subset, ConcatDataset | Subset, dict[str, str]]
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

    # Build subject→dataset mapping for validation subjects
    val_subject_dataset_map: dict[str, str] = {subj: ds for ds, subj in val_subjects_with_dataset}

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

            mdd_train_indices = get_indices_from_dataset(mdd_flat_dataset, mdd_train_subjects)
            mdd_val_indices = get_indices_from_dataset(mdd_flat_dataset, mdd_val_subjects)

            train_subsets.append(Subset(mdd_flat_dataset, mdd_train_indices))
            val_subsets.append(Subset(mdd_flat_dataset, mdd_val_indices))
            train_chunk_counts["MDD"] = len(mdd_train_indices)
            val_chunk_counts["MDD"] = len(mdd_val_indices)

        # Process CANE dataset if available
        if cane_flat_dataset is not None:
            cane_train_subjects = [subj for ds, subj in train_subjects_with_dataset if ds == "cane"]
            cane_val_subjects = [subj for ds, subj in val_subjects_with_dataset if ds == "cane"]

            cane_train_indices = get_indices_from_dataset(cane_flat_dataset, cane_train_subjects)
            cane_val_indices = get_indices_from_dataset(cane_flat_dataset, cane_val_subjects)

            train_subsets.append(Subset(cane_flat_dataset, cane_train_indices))
            val_subsets.append(Subset(cane_flat_dataset, cane_val_indices))
            train_chunk_counts["CANE"] = len(cane_train_indices)
            val_chunk_counts["CANE"] = len(cane_val_indices)

        # Process SAD dataset if available
        if sad_flat_dataset is not None:
            sad_train_subjects = [subj for ds, subj in train_subjects_with_dataset if ds == "sad"]
            sad_val_subjects = [subj for ds, subj in val_subjects_with_dataset if ds == "sad"]

            sad_train_indices = get_indices_from_dataset(sad_flat_dataset, sad_train_subjects)
            sad_val_indices = get_indices_from_dataset(sad_flat_dataset, sad_val_subjects)

            train_subsets.append(Subset(sad_flat_dataset, sad_train_indices))
            val_subsets.append(Subset(sad_flat_dataset, sad_val_indices))
            train_chunk_counts["SAD"] = len(sad_train_indices)
            val_chunk_counts["SAD"] = len(sad_val_indices)

        train_dataset: ConcatDataset | Subset = ConcatDataset(train_subsets)
        val_dataset: ConcatDataset | Subset = ConcatDataset(val_subsets)

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

    return train_dataset, val_dataset, val_subject_dataset_map
