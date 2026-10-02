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
    bipolar_pair,
)
from thesis.labels import LabelMapping
from thesis.model import RAW_EEG_MODELS

# Canonical sampling rate and spectrogram geometry live in thesis.stft, the single source of
# truth shared by the training and inference paths. EXPECTED_SPECTROGRAM_SHAPE is re-exported
# here (explicit "as" alias so ruff doesn't prune it as unused) for api/model_manager.py, which
# still assumes the fixed 70 Hz default.
from thesis.stft import (
    EXPECTED_SPECTROGRAM_SHAPE as EXPECTED_SPECTROGRAM_SHAPE,
)
from thesis.stft import FREQ_CUTOFF_HZ, MODEL_FS, spectrogram_shape

logger = logging.getLogger(__name__)

#: Seed of the RandomState threaded through dataset preparation. Fold membership depends on it,
#: so training and the explainability harness must share this one value.
CV_SEED = 42

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
    freq_cutoff_hz: float = FREQ_CUTOFF_HZ,
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
           preserve the expected STFT shape.
    :param float freq_cutoff_hz: Highest spectrogram frequency kept, in Hz. Ignored when
           use_raw_eeg=True.
    :return: Tuple of (flat_dataset, SubjectClasses).
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset,
            SubjectClasses]
    """
    all_flat_datasets: list = []
    all_subjects = []

    for condition in conditions:
        # For spectrogram models, always use default 10-s chunks to preserve the expected shape.
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
                    dataset,
                    target_samples=target_samples,
                    label_mapping=label_mapping,
                    is_inear=bipolar_pair(channel) is not None,
                )
            )
        else:
            # Create spectrogram dataset
            spec_dataset = SpectrogramDataset(
                dataset,
                source_fs=fs,
                augmentation=augmentation,
                channel=channel,
                freq_cutoff_hz=freq_cutoff_hz,
            )
            flat_dataset = FlattenedSpectrogramDataset(spec_dataset, label_mapping=label_mapping)

            # Validate shape
            expected_shape = spectrogram_shape(freq_cutoff_hz)
            spec_shape = flat_dataset[0][0].shape[1:]
            assert spec_shape == torch.Size(list(expected_shape)), (
                f"Expected {expected_shape}, got {spec_shape}. "
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
    freq_cutoff_hz: float = FREQ_CUTOFF_HZ,
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
    :param float freq_cutoff_hz: Highest spectrogram frequency kept, in Hz. Ignored when
           use_raw_eeg=True.
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
        freq_cutoff_hz=freq_cutoff_hz,
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
    freq_cutoff_hz: float = FREQ_CUTOFF_HZ,
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
    :param float freq_cutoff_hz: Highest spectrogram frequency kept, in Hz. Ignored when
           use_raw_eeg=True.
    :return: Tuple of (flat_dataset, SubjectClasses).
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset,
            SubjectClasses]
    """
    all_flat_datasets: list = []
    all_subjects = []

    for condition in conditions:
        # For spectrogram models, always use default 10-s chunks to preserve the expected shape.
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
                    cane_dataset,
                    target_samples=target_samples,
                    label_mapping=label_mapping,
                    is_inear=bipolar_pair(channel) is not None,
                )
            )
        else:
            cane_spec_dataset = SpectrogramDataset(
                cane_dataset,
                source_fs=CANEDataset.FS,
                augmentation=augmentation,
                channel=channel,
                freq_cutoff_hz=freq_cutoff_hz,
            )
            cane_flat_dataset = FlattenedSpectrogramDataset(
                cane_spec_dataset, label_mapping=label_mapping
            )
            expected_shape = spectrogram_shape(freq_cutoff_hz)
            spec_shape = cane_flat_dataset[0][0].shape[1:]
            assert spec_shape == torch.Size(list(expected_shape)), (
                f"Expected {expected_shape}, got {spec_shape}. "
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
    freq_cutoff_hz: float = FREQ_CUTOFF_HZ,
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
    :param float freq_cutoff_hz: Highest spectrogram frequency kept, in Hz. Ignored when
           use_raw_eeg=True.
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
        freq_cutoff_hz=freq_cutoff_hz,
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
    freq_cutoff_hz: float = FREQ_CUTOFF_HZ,
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
    :param float freq_cutoff_hz: Highest spectrogram frequency kept, in Hz. Ignored when
           use_raw_eeg=True.
    :return: Tuple of (flat_dataset, SubjectClasses).
    :rtype: tuple[ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset,
            SubjectClasses]
    """
    all_flat_datasets: list = []
    all_subjects = []

    for condition in conditions:
        # For spectrogram models, always use default 10-s chunks to preserve the expected shape.
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
                    idun_dataset,
                    target_samples=target_samples,
                    label_mapping=label_mapping,
                    is_inear=True,
                )
            )
        else:
            idun_spec_dataset = SpectrogramDataset(
                idun_dataset,
                source_fs=IDUNDataset.FS,
                augmentation=augmentation,
                channel=channel,
                freq_cutoff_hz=freq_cutoff_hz,
            )
            idun_flat_dataset = FlattenedSpectrogramDataset(
                idun_spec_dataset, label_mapping=label_mapping
            )
            expected_shape = spectrogram_shape(freq_cutoff_hz)
            spec_shape = idun_flat_dataset[0][0].shape[1:]
            assert spec_shape == torch.Size(list(expected_shape)), (
                f"Expected {expected_shape}, got {spec_shape}. "
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


#: Any of the flattened dataset types the fold helpers accept.
FlatDataset = ConcatDataset | FlattenedSpectrogramDataset | FlattenedRawEEGDataset


class CVFolds(NamedTuple):
    """Everything the cross-validation loop needs before it starts iterating folds."""

    normal_folds: list[SubjectList]
    anxiety_folds: list[SubjectList]
    depression_folds: list[SubjectList]
    anxiety_depression_folds: list[SubjectList]
    mdd_flat_dataset: FlatDataset | None
    cane_flat_dataset: FlatDataset | None
    sad_flat_dataset: FlatDataset | None
    flat_dataset: FlatDataset | None
    num_classes: int
    use_raw_eeg: bool
    dataset_type: str

    def datasets_for_fold(
        self, fold: int
    ) -> tuple[ConcatDataset | Subset, ConcatDataset | Subset, dict[str, str]]:
        """
        Train and validation datasets for one fold.

        :param int fold: Zero-indexed fold number.
        :return: ``(train_dataset, val_dataset, val_subject_dataset_map)``.
        :rtype: tuple
        """
        return get_datasets_for_fold(
            fold,
            self.normal_folds,
            self.anxiety_folds,
            self.depression_folds,
            self.anxiety_depression_folds,
            self.dataset_type,
            self.mdd_flat_dataset,
            self.cane_flat_dataset,
            self.sad_flat_dataset,
            self.flat_dataset,
        )


def build_cv_folds(
    dataset_type: Literal["mdd", "cane", "sad", "all"],
    conditions: list[str],
    class_mode: str,
    n_folds: int,
    channel: str | None,
    model_name: str,
    *,
    skip_artifact_removal: bool = False,
    augmentation: Callable | None = None,
    test_mode: bool = False,
    chunk_duration: float = CHUNK_DURATION_SEC,
    freq_cutoff_hz: float = FREQ_CUTOFF_HZ,
    seed: int = CV_SEED,
) -> CVFolds:
    """
    Load the requested datasets and split their subjects into stratified folds.

    Shared by training (``main.train_cross_validation``) and the post-hoc explainability
    harness, so the harness reproduces exactly the folds a training run used. That matters
    because a single ``RandomState`` is threaded through every ``prepare_*`` call in a fixed
    order: the fold assignment depends on which datasets are loaded and in what order.

    :param dataset_type: Which dataset(s) to load.
    :param list[str] conditions: Upper-case conditions, e.g. ``["EC", "EO"]``.
    :param str class_mode: ``"2"`` or ``"4"``.
    :param int n_folds: Number of cross-validation folds.
    :param str channel: Channel spec (``all``, ``in-ear``, an electrode name, or a montage).
    :param str model_name: Architecture name; decides raw-EEG versus spectrogram pipeline.
    :param bool skip_artifact_removal: Skip CANE artifact interpolation/clipping.
    :param augmentation: Optional augmentation applied to training chunks.
    :param bool test_mode: Load a single file per class, for debugging.
    :param float chunk_duration: Chunk length in seconds.
    :param float freq_cutoff_hz: Highest spectrogram frequency kept, in Hz.
    :param int seed: Seed for the fold-assignment RandomState.
    :return: Fold lists and the backing datasets.
    :rtype: CVFolds
    :raises ValueError: If condition ``TASK`` is requested for the CANE dataset.
    """
    rng = np.random.RandomState(seed)
    if "TASK" in conditions and dataset_type == "cane":
        raise ValueError("condition TASK not possible for CANE dataset")

    # Determine num_classes and label_mapping based on class_mode
    num_classes, label_mapping = determine_num_classes(class_mode)
    logger.info(f"Classification mode: {num_classes}-class")
    if label_mapping:
        logger.info(f"Label mapping: {label_mapping} (collapsing to binary)")

    # Detect whether model needs raw EEG (Deformer) or spectrograms (CNN-LSTM family)
    use_raw_eeg = model_name in RAW_EEG_MODELS
    if use_raw_eeg:
        logger.info(f"Model {model_name} uses raw EEG pipeline (no STFT)")

    mdd_flat_dataset, cane_flat_dataset, sad_flat_dataset, flat_dataset = (
        None,
        None,
        None,
        None,
    )
    if dataset_type == "mdd":
        flat_dataset, subject_classes = prepare_mdd_dataset(
            conditions,
            channel,
            rng,
            augmentation=augmentation,
            test_mode=test_mode,
            num_classes=num_classes,
            label_mapping=label_mapping,
            use_raw_eeg=use_raw_eeg,
            chunk_duration=chunk_duration,
            freq_cutoff_hz=freq_cutoff_hz,
        )
        normal, anxiety, depression, anxiety_depression = (
            subject_classes.normal,
            subject_classes.anxiety,
            subject_classes.depression,
            subject_classes.anxiety_depression,
        )
        logger.info(f"MDD dataset: {len(normal)} normal, {len(depression)} depression subjects")
    elif dataset_type == "cane":
        # Convert to lowercase for CANE/IDUN
        cane_conditions = [c.lower() for c in conditions]
        if channel == "in-ear":
            logger.info("Using IDUN real in-ear dataset instead of CANE synthetic derivation")
            flat_dataset, subject_classes = prepare_idun_dataset(
                cane_conditions,
                channel,
                rng,
                label_mapping=label_mapping,
                augmentation=augmentation,
                test_mode=test_mode,
                use_raw_eeg=use_raw_eeg,
                chunk_duration=chunk_duration,
                freq_cutoff_hz=freq_cutoff_hz,
            )
        else:
            if bipolar_pair(channel) is not None:
                logger.info(
                    f"Bipolar channel '{channel}' with CANE: loading from the 8-channel "
                    "headset, not IDUN"
                )
            flat_dataset, subject_classes = prepare_cane_dataset(
                cane_conditions,
                channel,
                rng,
                label_mapping=label_mapping,
                skip_artifact_removal=skip_artifact_removal,
                augmentation=augmentation,
                test_mode=test_mode,
                use_raw_eeg=use_raw_eeg,
                chunk_duration=chunk_duration,
                freq_cutoff_hz=freq_cutoff_hz,
            )
        normal, anxiety, depression, anxiety_depression = (
            subject_classes.normal,
            subject_classes.anxiety,
            subject_classes.depression,
            subject_classes.anxiety_depression,
        )
        dataset_name = "IDUN" if channel == "in-ear" else "CANE"
        logger.info(
            f"{dataset_name} dataset: {len(normal)} normal, {len(anxiety)} anxiety, "
            f"{len(depression)} depression, {len(anxiety_depression)} anxiety+depression subjects"
        )
    elif dataset_type == "sad":
        flat_dataset, subject_classes = prepare_sad_dataset(
            conditions,
            channel,
            rng,
            augmentation=augmentation,
            test_mode=test_mode,
            num_classes=num_classes,
            label_mapping=label_mapping,
            use_raw_eeg=use_raw_eeg,
            chunk_duration=chunk_duration,
            freq_cutoff_hz=freq_cutoff_hz,
        )
        normal, anxiety, depression, anxiety_depression = (
            subject_classes.normal,
            subject_classes.anxiety,
            subject_classes.depression,
            subject_classes.anxiety_depression,
        )
        logger.info(f"SAD dataset: {len(normal)} normal, {len(anxiety)} anxiety subjects")
    elif dataset_type == "all":
        # Load MDD dataset
        # T3=T7 and T4=T8 for these purposes, otherwise I couldn't combine the datasets
        if channel in ["T7", "T8"]:
            channel = {"T7": "T3", "T8": "T4"}[channel]
        mdd_flat_dataset, mdd_subject_classes = prepare_mdd_dataset(
            conditions,
            channel,
            rng,
            augmentation=augmentation,
            test_mode=test_mode,
            num_classes=num_classes,
            label_mapping=label_mapping,
            use_raw_eeg=use_raw_eeg,
            chunk_duration=chunk_duration,
            freq_cutoff_hz=freq_cutoff_hz,
        )
        mdd_normal, mdd_anxiety, mdd_depression, mdd_anxiety_depression = (
            mdd_subject_classes.normal,
            mdd_subject_classes.anxiety,
            mdd_subject_classes.depression,
            mdd_subject_classes.anxiety_depression,
        )
        if channel in ["T3", "T4"]:
            channel = {"T3": "T7", "T4": "T8"}[channel]

        # Load CANE dataset (or IDUN real in-ear when channel == "in-ear")
        cane_conditions = [c.lower() for c in conditions]
        if channel == "in-ear":
            logger.info("Using IDUN real in-ear dataset instead of CANE synthetic derivation")
            cane_flat_dataset, cane_subject_classes = prepare_idun_dataset(
                cane_conditions,
                channel,
                rng,
                label_mapping=label_mapping,
                augmentation=augmentation,
                test_mode=test_mode,
                use_raw_eeg=use_raw_eeg,
                chunk_duration=chunk_duration,
                freq_cutoff_hz=freq_cutoff_hz,
            )
        else:
            if bipolar_pair(channel) is not None:
                logger.info(
                    f"Bipolar channel '{channel}' with CANE: loading from the 8-channel "
                    "headset, not IDUN"
                )
            cane_flat_dataset, cane_subject_classes = prepare_cane_dataset(
                cane_conditions,
                channel,
                rng,
                label_mapping=label_mapping,
                skip_artifact_removal=skip_artifact_removal,
                augmentation=augmentation,
                test_mode=test_mode,
                use_raw_eeg=use_raw_eeg,
                chunk_duration=chunk_duration,
                freq_cutoff_hz=freq_cutoff_hz,
            )
        cane_normal, cane_anxiety, cane_depression, cane_anxiety_depression = (
            cane_subject_classes.normal,
            cane_subject_classes.anxiety,
            cane_subject_classes.depression,
            cane_subject_classes.anxiety_depression,
        )

        # Load SAD dataset
        sad_flat_dataset, sad_subject_classes = prepare_sad_dataset(
            conditions,
            channel,
            rng,
            augmentation=augmentation,
            test_mode=test_mode,
            num_classes=num_classes,
            label_mapping=label_mapping,
            use_raw_eeg=use_raw_eeg,
            chunk_duration=chunk_duration,
            freq_cutoff_hz=freq_cutoff_hz,
        )
        sad_normal, sad_anxiety, sad_depression, sad_anxiety_depression = (
            sad_subject_classes.normal,
            sad_subject_classes.anxiety,
            sad_subject_classes.depression,
            sad_subject_classes.anxiety_depression,
        )

    # Create folds for each class separately (stratified)
    if dataset_type == "all":
        # For combined dataset, stratify each dataset separately then merge corresponding folds
        normal_folds, anxiety_folds, depression_folds, anxiety_depression_folds = (
            create_balanced_folds(
                [mdd_normal, cane_normal, sad_normal],
                [mdd_anxiety, cane_anxiety, sad_anxiety],
                [mdd_depression, cane_depression, sad_depression],
                [mdd_anxiety_depression, cane_anxiety_depression, sad_anxiety_depression],
                n_folds,
            )
        )
        logger.info(
            f"Created {n_folds} balanced folds with subjects from MDD, CANE, and SAD datasets"
        )
    else:
        # Single dataset: some classes may be empty
        empty_folds: list[SubjectList] = [[] for _ in range(n_folds)]
        normal_folds = split_into_folds(normal, n_folds)
        anxiety_folds = split_into_folds(anxiety, n_folds) if anxiety else empty_folds
        depression_folds = split_into_folds(depression, n_folds) if depression else empty_folds
        anxiety_depression_folds = (
            split_into_folds(anxiety_depression, n_folds) if anxiety_depression else empty_folds
        )

    return CVFolds(
        normal_folds=normal_folds,
        anxiety_folds=anxiety_folds,
        depression_folds=depression_folds,
        anxiety_depression_folds=anxiety_depression_folds,
        mdd_flat_dataset=mdd_flat_dataset,
        cane_flat_dataset=cane_flat_dataset,
        sad_flat_dataset=sad_flat_dataset,
        flat_dataset=flat_dataset,
        num_classes=num_classes,
        use_raw_eeg=use_raw_eeg,
        dataset_type=dataset_type,
    )
