import logging
import os
import re
import warnings
from collections import Counter
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Literal, Optional, cast

import mne
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, detrend, filtfilt, iirnotch, resample, stft
from scipy.stats import zscore
from torch.utils.data import Dataset

from thesis.labels import CanonicalLabel

DEFAULT_CACHE_SIZE: int = 100

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

BASE_DIR = Path(os.environ.get("EEG_DATA_DIR", "/home/milan/Documents/diplomka/"))
CANE_DIR = BASE_DIR / "CANE"
MDD_DIR = BASE_DIR / "MDD"
SAD_DIR = BASE_DIR / "SAD"
IDUN_DIR = BASE_DIR / "IDUN_IN_EAR"

CHUNK_DURATION_SEC = 10

# Canonical channel order for multi-channel loading (consistent across datasets)
# T7/T8 are new nomenclature, equivalent to T3/T4 in older systems
CANONICAL_CHANNEL_ORDER = ["Fp1", "Fp2", "C3", "Cz", "C4", "T7", "T8", "O2/Oz"]

# Dataset-specific channel mappings to canonical order
MDD_CHANNEL_ORDER = ["Fp1", "Fp2", "C3", "Cz", "C4", "T7", "T8", "O2"]  # T3→T7, T4→T8
CANE_CHANNEL_ORDER = ["Fp1", "Fp2", "C3", "Cz", "C4", "T7", "T8", "Oz"]
SAD_CHANNEL_ORDER = ["Fp1", "Fp2", "C3", "Cz", "C4", "T7", "T8", "O2"]

LABEL_INT_MAP: dict[str, int] = {
    "normals": CanonicalLabel.HEALTHY,
    "anxiety": CanonicalLabel.ANXIETY_ONLY,
    "depression": CanonicalLabel.DEPRESSION_ONLY,
    "comorbid": CanonicalLabel.COMORBID,
}


def load_and_preprocess_edf_file(
    file_path: Path,
    channel: str,
    fs: float,
    channel_mapping: dict[str, str],
    channel_order: list[str],
    *,
    chunk_samples: int = int(CHUNK_DURATION_SEC * 250),
) -> torch.Tensor:
    """
    Load an EDF file from disk, preprocess it, and return chunks.

    Pipeline: bandpass 1-70 Hz → notch 50 Hz → average reference → linear detrend
    → channel selection → chunking → per-chunk z-score.

    :param Path file_path: Path to the EDF file to preprocess.
    :param str channel: Channel to use: specific channel name (e.g., "Fp1") or
           "all" for all 8 channels.
    :param float fs: Sampling frequency in Hz (e.g., 250 for MDD, 256 for AX_MALIK).
    :param dict[str, str] channel_mapping: Mapping from raw channel names to canonical names
           (e.g., {"EEG Fp1-LE": "Fp1"}).
    :param list[str] channel_order: Canonical channel order for "all" channel mode
           (e.g., ["Fp1", "Fp2", "C3", "Cz", "C4", "T7", "T8", "O2"]).
    :param int chunk_samples: Number of samples per chunk (e.g. 2500 = 10 s × 250 Hz).
    :return: Tensor of preprocessed EEG chunks with shape (num_chunks, channels, samples).
    :rtype: torch.Tensor
    """

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw = raw.filter(l_freq=1, h_freq=70, method="iir", verbose=False)
    raw = raw.notch_filter(freqs=50, verbose=False)

    raw = raw.pick(list(channel_mapping.keys()))
    raw = raw.rename_channels(channel_mapping)

    # Drop reference electrode before computing average reference
    if "A2-A1" in raw.ch_names:
        raw = raw.drop_channels(["A2-A1"])

    if channel == "in-ear":
        # Bipolar derivation: only T7 and T8 are needed; CAR over two electrodes is meaningless.
        # Sign flip augmentation is applied later in SpectrogramDataset (per-epoch).
        raw = raw.pick(["T7", "T8"])
        pair_data = raw.get_data()  # (2, n_samples) — T7 at [0], T8 at [1]
        pair_data = detrend(pair_data, axis=1, type="linear")
        inear_signal = pair_data[1] - pair_data[0]  # T8 - T7
        data = inear_signal.reshape(1, -1)
    else:
        # Get all EEG channel data at once for shared preprocessing steps
        all_data = raw.get_data()  # (n_channels, n_samples)
        ch_index = {name: i for i, name in enumerate(raw.ch_names)}

        # Average reference: subtract mean across channels at each time point (matches CANE CAR)
        all_data -= all_data.mean(axis=0, keepdims=True)

        # Linear detrend per channel: removes slow DC drifts (matches CANE/IDUN)
        all_data = detrend(all_data, axis=1, type="linear")

        if channel == "all":
            data = np.stack([all_data[ch_index[ch]] for ch in channel_order])
        elif channel is not None:
            data = all_data[[ch_index[channel]]]
        else:
            raise RuntimeError("Must specify 'channel' parameter")

    n_samples = data.shape[1]
    chunks_list: list[np.ndarray] = []
    for i in range(0, n_samples - chunk_samples + 1, chunk_samples):
        chunk = data[:, i : i + chunk_samples]
        chunk = zscore(chunk, axis=1)
        chunks_list.append(chunk)

    chunks_tensor = torch.stack([torch.from_numpy(chunk).float() for chunk in chunks_list])
    return chunks_tensor


class MDDDataset(Dataset):
    """
    PyTorch Dataset for MDD EEG data.
    """

    # MDD channel mapping from raw EDF names to canonical names
    CHANNEL_MAPPING = {
        "EEG A2-A1": "A2-A1",
        "EEG C3-LE": "C3",
        "EEG C4-LE": "C4",
        "EEG Cz-LE": "Cz",
        "EEG Fp1-LE": "Fp1",
        "EEG Fp2-LE": "Fp2",
        "EEG O2-LE": "O2",
        # The reason T3->T7 and T4->T8 is so that I can merge multiple datasets
        "EEG T3-LE": "T7",
        "EEG T4-LE": "T8",
    }

    FS = 1000 / 4
    CHUNK_SAMPLES = int(CHUNK_DURATION_SEC * FS)

    @staticmethod
    def _log_file_discovery(
        files: list[dict[str, Any]], dataset_name: str, data_dir: Optional[Path] = None
    ) -> None:
        """
        Log discovery statistics for a list of discovered files.

        :param list[dict[str, Any]] files: List of file metadata dictionaries.
        :param str dataset_name: Name of the dataset for logging purposes.
        :param Optional[Path] data_dir: Directory that was searched, included in warning when empty.
        :rtype: None
        """
        if files:
            label_counts = Counter(f["label"] for f in files)
            condition_counts = Counter(f["condition"] for f in files)
            logger.info(
                f"Discovered {len(files)} {dataset_name} files: "
                f"{dict(label_counts)} labels, {dict(condition_counts)} conditions"
            )
        else:
            dir_info = f" in {data_dir}" if data_dir is not None else ""
            logger.warning(f"No {dataset_name} files discovered matching criteria{dir_info}")

    def __init__(
        self,
        data_dir: Path = MDD_DIR,
        condition: Optional[Literal["EC", "EO", "TASK"]] = None,
        subjects: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
        cache_size: Optional[int] = DEFAULT_CACHE_SIZE,
        transform: Optional[Callable] = None,
        channel: Optional[str] = "all",
        test_mode: bool = False,
        chunk_duration: float = CHUNK_DURATION_SEC,
    ):
        """
        Initialize the MDD EEG dataset.

        :param Path data_dir: Path to directory containing .edf files.
        :param Optional[Literal["EC", "EO", "TASK"]] condition: Filter by condition ("EC", "EO",
               "TASK") or None for all conditions.
        :param Optional[list[str]] subjects: List of subject IDs to include (e.g., ["H S1",
               "MDD S1"]). None = all.
        :param Optional[list[str]] labels: List of labels to include (e.g., ["H", "MDD"]).
               None = all.
        :param int cache_size: Number of preprocessed files to cache in memory. If None,
               cache is not used
        :param Optional[Callable] transform: Optional transform function to apply to EEG data.
        :param Optional[str] channel: Channel to use: specific channel name (e.g., "Fp1") or
               "all" for all 8 channels (excludes A2-A1 reference).
        :param bool test_mode: If True, load only one file per class for debugging.
        :param float chunk_duration: EEG chunk duration in seconds (default: CHUNK_DURATION_SEC).
               Affects the number of samples per chunk (chunk_samples = chunk_duration × FS).
        """
        self.data_dir = Path(data_dir)
        self.condition = condition
        self.transform = transform
        self.test_mode = test_mode
        self.chunk_samples = int(chunk_duration * self.FS)
        self.files = self._discover_files(
            cast(Literal["EC", "EO", "TASK"], condition.upper()) if condition is not None else None,
            subjects,
            labels,
        )

        # Handle channel selection
        self.channel = channel
        self.channel_names = (
            MDD_CHANNEL_ORDER
            if channel == "all"
            else ["in-ear"]
            if channel == "in-ear"
            else ([channel] if channel else [])
        )

        # Bind dataset-specific parameters to the preprocessing function
        preprocess_func = partial(
            load_and_preprocess_edf_file,
            channel_mapping=self.CHANNEL_MAPPING,
            channel_order=MDD_CHANNEL_ORDER,
            chunk_samples=self.chunk_samples,
        )

        if cache_size:
            self._load_and_preprocess_edf_raw_file_cached = lru_cache(maxsize=cache_size)(
                preprocess_func
            )
        else:
            self._load_and_preprocess_edf_raw_file_cached = preprocess_func

    def _discover_files(
        self,
        condition: Optional[Literal["EC", "EO", "TASK"]],
        subjects: Optional[list[str]],
        labels: Optional[list[str]],
    ) -> list[dict[str, Any]]:
        """
        Discover all .edf files matching the criteria.

        :param Optional[Literal["EC", "EO", "TASK"]] condition: Condition filter ("EC", "EO",
               "TASK") or None.
        :param Optional[list[str]] subjects: List of subject IDs to include or None for all.
        :param Optional[list[str]] labels: List of labels to include ("H", "MDD") or None for all.
        :return: List of dictionaries containing file metadata.
        :rtype: list[dict]
        """
        files = []
        if self.test_mode:
            pattern = re.compile(r"(H|MDD) S(1) (EC|EO|TASK)\.edf")
        else:
            pattern = re.compile(r"(H|MDD) S(\d+) (EC|EO|TASK)\.edf")

        for file_path in sorted(self.data_dir.glob("*.edf")):
            match = pattern.match(file_path.name)
            if not match:
                continue

            label, subject_num, cond = match.groups()
            subject_id = f"{label} S{subject_num} {cond}"

            # Filtering
            if condition and cond != condition:
                continue

            if subjects and subject_id not in subjects:
                continue

            if labels and label not in labels:
                continue

            files.append(
                {
                    "path": file_path,
                    "label": label,
                    "subject": subject_id,
                    "condition": cond,
                    "label_int": (
                        CanonicalLabel.HEALTHY if label == "H" else CanonicalLabel.DEPRESSION_ONLY
                    ),
                }
            )

        MDDDataset._log_file_discovery(files, "MDD", self.data_dir)
        return files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get file from the dataset preprocessed by self.transform method

        :param int idx: Index of the file to retrieve.
        :return: Dictionary containing 'eeg' (tensor of shape (num_chunks, channels, samples)),
                 'label' (integer: 0=Healthy, 1=MDD), 'subject' (subject ID string),
                 'condition' (condition string: EC/EO/TASK), and 'channels' (list of channel names
                 in the order they appear in the tensor).
        :rtype: dict
        """
        file_info = self.files[idx]

        chunks = self._load_and_preprocess_edf_raw_file_cached(
            file_info["path"], self.channel, self.FS
        )

        if self.transform:
            eeg_tensor = self.transform(chunks)
        else:
            eeg_tensor = chunks

        return {
            "eeg": eeg_tensor,
            "label": file_info["label_int"],
            "subject": file_info["subject"],
            "condition": file_info["condition"],
            "channels": self.channel_names,
        }

    def get_sorted_subjects(self) -> list[str]:
        """
        Get a list of unique subjects in the dataset.

        :return: Sorted list of unique subject IDs.
        :rtype: list[str]
        """
        return sorted(list(set(f["subject"] for f in self.files)))

    def get_statistics(self) -> dict[str, Any]:
        """
        Get dataset statistics.

        :return: Dictionary containing total files, healthy/MDD file counts,
                 conditions breakdown, and number of unique subjects.
        :rtype: dict[str, Any]
        """
        healthy_count = sum(1 for f in self.files if f["label"] == "H")
        mdd_count = sum(1 for f in self.files if f["label"] == "MDD")

        conditions: dict[str, int] = {}
        for f in self.files:
            conditions[f["condition"]] = conditions.get(f["condition"], 0) + 1

        return {
            "total_files": len(self.files),
            "healthy_files": healthy_count,
            "mdd_files": mdd_count,
            "conditions": conditions,
            "subjects": len(self.get_sorted_subjects()),
        }


class NaNValuesError(Exception):
    pass


class CANEDataset(Dataset):
    """
    PyTorch Dataset for CANE EEG data (anxiety detection).
    """

    FS = 500  # Hz (hardcoded, verified to be ~498-499 Hz in practice)
    SAMPLING_RATE_TOLERANCE = 10  # Hz (allow ±10 Hz deviation)

    # STFT Parameters for matching MDD spectrogram shape (129, 41):
    STFT_NPERSEG = 256
    STFT_NOVERLAP = 131

    NEIGHBOUR_ZSCORE_THRESHOLD = 2.5

    # Available EEG channels in raw CANE data
    CHANNELS = [f"d{i}" for i in range(1, 9)]

    CLASS_DIRECTORIES = ["normals", "anxiety", "depression", "comorbid"]
    LABEL_MAP = {"normals": "H", "anxiety": "AX", "depression": "DEP", "comorbid": "AXDEP"}
    CHANNEL_MAPPING = {
        "d1": "Fp1",
        "d2": "Fp2",
        "d3": "T7",
        "d4": "C3",
        "d5": "Cz",
        "d6": "C4",
        "d7": "T8",
        "d8": "Oz",
    }

    # Reverse mapping: MDD -> CANE (for when user specifies MDD channel name)
    REVERSE_CHANNEL_MAPPING = {v: k for k, v in CHANNEL_MAPPING.items()}

    def __init__(
        self,
        data_dir: Path = CANE_DIR,
        condition: Optional[Literal["ec", "eo"]] = None,
        subjects: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
        cache_size: Optional[int] = DEFAULT_CACHE_SIZE,
        transform: Optional[Callable] = None,
        channel: Optional[str] = "Fp1",
        artifact_method: str = "interpolation",
        artifact_threshold: float = 4.0,
        apply_car: bool = True,
        skip_extreme_artifacts: bool = False,
        skip_artifact_removal: bool = False,
        test_mode: bool = False,
        chunk_duration: float = CHUNK_DURATION_SEC,
    ):
        """
        Initialize the CANE EEG dataset.

        :param Path data_dir: Path to directory containing .csv files.
        :param Optional[Literal["EC", "EO"]] condition: Filter by condition
               or None for all conditions.
        :param Optional[list[str]] subjects: List of subject IDs to include. None = all.
        :param Optional[list[str]] labels: List of labels to include (e.g., ["normals", "anxiety"]).
               None = all.
        :param int cache_size: Number of preprocessed files to cache in memory. If None,
               cache is not used.
        :param Optional[Callable] transform: Optional transform function to apply to EEG data.
        :param Optional[str] channel: Channel to use: specific channel name (e.g., "Fp1", "T7")
               or "all" for all 8 channels. Accepts both MDD channel names (Fp1, Fp2, C3, C4, etc.)
               and CANE channel names (d1-d8).
        :param str artifact_method: Method for artifact removal ("interpolation", "clipping",
               or "both").
        :param float artifact_threshold: Z-score threshold for artifact detection.
        :param bool apply_car: Whether to apply Common Average Reference.
        :param bool skip_extreme_artifacts: If True, completely removes chunks with >10% artifacts.
        :param bool skip_artifact_removal: If True, skip artifact interpolation and clipping
               (let the neural network learn to handle artifacts).
        :param bool test_mode: If True, load only one file per class for debugging.
        :param float chunk_duration: EEG chunk duration in seconds (default: CHUNK_DURATION_SEC).
               Affects the number of samples per chunk at the native CANE sampling rate.
        """
        self.data_dir = Path(data_dir)
        self.condition = str(condition).lower()
        self.transform = transform
        self.channel = channel
        self.artifact_method = artifact_method
        self.artifact_threshold = artifact_threshold
        self.apply_car = apply_car
        self.skip_extreme_artifacts = skip_extreme_artifacts
        self.skip_artifact_removal = skip_artifact_removal
        self.test_mode = test_mode
        self.chunk_samples = int(chunk_duration * self.FS)

        # Set channel_names for verification
        if channel == "all":
            self.channel_names = CANE_CHANNEL_ORDER
        elif channel == "in-ear":
            self.channel_names = ["in-ear"]
        elif channel:
            self.channel_names = [channel]
        else:
            self.channel_names = []

        self.files = self._discover_files(condition, subjects, labels)

        self._load_and_preprocess_cane_raw_file: Any
        if cache_size:
            self._load_and_preprocess_cane_raw_file = lru_cache(maxsize=cache_size)(
                CANEDataset.load_and_preprocess_cane_raw_file
            )
        else:
            self._load_and_preprocess_cane_raw_file = CANEDataset.load_and_preprocess_cane_raw_file

    def _discover_files(
        self,
        condition: Optional[Literal["ec", "eo"]],
        subjects: Optional[list[str]],
        labels: Optional[list[str]],
    ) -> list[dict[str, Any]]:
        """
        Discover all .csv files matching the criteria.

        Files are organized in directory structure:
        {label}/{condition}/{subject_id}_{condition}.csv where label is "normals" or "anxiety",
        and condition is "ec" (eyes closed) or "eo" (eyes open).

        :param Optional[Literal["ec", "eo"]] condition: Condition filter ("ec", "eo") or None.
        :param Optional[list[str]] subjects: List of subject IDs to include or None for all.
        :param Optional[list[str]] labels: List of labels to include ("normals", "anxiety")
               or None for all.
        :return: List of dictionaries containing file metadata.
        :rtype: list[dict]
        """
        # Test mode: match specific subject IDs
        # (0012=normals, 0041=anxiety, 0016=depression, 0013=comorbid)
        if self.test_mode:
            pattern = re.compile(r"(0041|0013|0012|0016)_?(EC|EO|Ec|Eo|ec|eo)\.csv", re.IGNORECASE)
        else:
            # Pattern to match filenames like "0041_Ec.csv", "1005_EC.csv", or "1001EC.csv"
            # (with or without underscore)
            pattern = re.compile(r"(\d+)_?(EC|EO|Ec|Eo|ec|eo)\.csv", re.IGNORECASE)

        files = []
        # Traverse the directory structure: {label}/{condition}/*.csv
        condition_map = {"ec": "EC", "eo": "EO"}
        for label_dir in self.CLASS_DIRECTORIES:
            label_path = self.data_dir / label_dir
            if not label_path.exists():
                continue

            for condition_dir in ["ec", "eo"]:
                # Filter by condition if specified
                if condition and condition_dir != condition:
                    continue

                condition_path = label_path / condition_dir
                if not condition_path.exists():
                    continue

                for file_path in sorted(condition_path.glob("*.csv")):
                    match = pattern.match(file_path.name)
                    if not match:
                        logger.warning(f"Skipping file with unexpected name: {file_path.name}")
                        continue

                    subject_num, _ = match.groups()
                    label = self.LABEL_MAP.get(label_dir, label_dir)
                    subject_id = (
                        f"{label} S{subject_num} {condition_map.get(condition_dir, condition_dir)}"
                    )

                    # Filtering by subjects
                    if subjects and subject_id not in subjects:
                        continue

                    # Filtering by labels
                    if labels and label_dir not in labels:
                        continue

                    files.append(
                        {
                            "path": file_path,
                            "label": label_dir,
                            "subject": subject_id,
                            "condition": condition_dir,
                            "label_int": LABEL_INT_MAP[label_dir],
                        }
                    )

        MDDDataset._log_file_discovery(files, "CANE", self.data_dir)
        return files

    @staticmethod
    def verify_sampling_rate(df: pd.DataFrame) -> None:
        """
        Verify that detected sampling rate is within tolerance.

        :param pd.DataFrame df: DataFrame containing timestamp column.
        :raises ValueError: If sampling rate is outside tolerance.
        :rtype: None
        """
        duration = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) / 1000  # to seconds
        detected_fs = len(df) / duration
        logger.info(f"Detected sampling rate: {detected_fs:.2f} Hz")
        if abs(detected_fs - CANEDataset.FS) > CANEDataset.SAMPLING_RATE_TOLERANCE:
            raise ValueError(
                f"Detected sampling rate {detected_fs:.2f} Hz is outside tolerance "
                f"({CANEDataset.FS} ± {CANEDataset.SAMPLING_RATE_TOLERANCE} Hz)"
            )

    @staticmethod
    def _process_single_channel(
        signal: np.ndarray,
        channel_name: str,
        file_path: Path,
        artifact_method: str,
        artifact_threshold: float,
        skip_artifact_removal: bool,
    ) -> np.ndarray:
        """
        Process a single EEG channel through the preprocessing pipeline.

        :param np.ndarray signal: Raw channel signal.
        :param str channel_name: Name of the channel being processed.
        :param Path file_path: Path to the file being processed (for error reporting).
        :param str artifact_method: Method for artifact removal
               ("interpolation", "clipping", or "both").
        :param float artifact_threshold: Z-score threshold for artifact detection.
        :param bool skip_artifact_removal: If True, skip artifact removal.
        :return: Processed channel signal.
        :rtype: np.ndarray
        """
        # Check for NaN/inf values before detrending
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            raise NaNValuesError(
                f"Double check file {file_path}, seems like its standard deviation is 0."
            )

        # Step 1: Detrend to remove slow drifts (before filtering)
        signal = detrend(signal, type="linear")

        # Step 2: Artifact handling (before filtering to avoid spreading artifacts)
        # Skip if skip_artifact_removal is True (let neural network handle artifacts)
        if not skip_artifact_removal:
            if artifact_method in ["interpolation", "both"]:
                z_scores = np.abs(zscore(signal))
                artifact_indices = np.where(z_scores > artifact_threshold)[0]

                if len(artifact_indices) > 0:
                    artifact_pct = len(artifact_indices) / len(signal) * 100
                    logger.info(
                        f"Channel {channel_name}: Found {len(artifact_indices)} "
                        f"artifacts ({artifact_pct:.2f}%)"
                    )

                    # Interpolate artifacts
                    for idx in artifact_indices:
                        # Find clean neighbors within ±100 samples
                        window_start = max(0, idx - 100)
                        window_end = min(len(signal), idx + 100)

                        neighbors = signal[window_start:window_end]
                        neighbor_mask = (
                            np.abs(zscore(neighbors)) < CANEDataset.NEIGHBOUR_ZSCORE_THRESHOLD
                        )

                        if neighbor_mask.sum() > 10:  # Need at least 10 clean samples
                            signal[idx] = np.median(neighbors[neighbor_mask])
                        elif (
                            0 < idx < len(signal) - 1
                        ):  # If no clean neighbors, use linear interpolation
                            signal[idx] = (signal[idx - 1] + signal[idx + 1]) / 2

            if artifact_method in ["clipping", "both"]:
                # Percentile-based clipping as additional safety
                lower = np.percentile(signal, 0.5)
                upper = np.percentile(signal, 99.5)
                signal = np.clip(signal, lower, upper)
        else:
            logger.debug("Skipping artifact removal (--skip-artifact-removal enabled)")

        # Step 3: Bandpass filter (matching MDD: 1-70 Hz)
        nyq = 0.5 * CANEDataset.FS
        b_bp, a_bp = butter(4, [1.0 / nyq, 70.0 / nyq], btype="band")
        signal = filtfilt(b_bp, a_bp, signal)

        # Step 4: Notch filter at 50 Hz (European powerline)
        b_notch, a_notch = iirnotch(50.0, Q=30, fs=CANEDataset.FS)
        signal = filtfilt(b_notch, a_notch, signal)

        return signal

    @staticmethod
    def load_and_preprocess_cane_raw_file(
        file_path: Path,
        channel: str = "all",
        artifact_method: str = "interpolation",
        artifact_threshold: float = 4.0,
        apply_car: bool = True,
        skip_extreme_artifacts: bool = False,
        skip_artifact_removal: bool = False,
        chunk_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Load a CSV file from disk, preprocess it, and return chunks.

        This contains all the preprocessing logic for CANE data.
        Preprocessing pipeline: CAR → detrend → artifact removal → bandpass filter (1-70 Hz)
        → notch filter (50 Hz) → chunking → z-score normalization.

        :param Path file_path: Path to the CSV file to preprocess.
        :param str channel: Channel to use: specific channel name (e.g., "Fp1", "T7") or None.
        :param str artifact_method: Method for artifact removal ("interpolation", "clipping",
               or "both").
        :param float artifact_threshold: Z-score threshold for artifact detection.
        :param bool apply_car: Whether to apply Common Average Reference.
        :param bool skip_extreme_artifacts: If True, completely removes chunks with >10% artifacts.
        :param bool skip_artifact_removal: If True, skip artifact interpolation and clipping.
        :param Optional[int] chunk_samples: Number of samples per chunk at native CANE sampling
               rate. Defaults to ``int(CHUNK_DURATION_SEC * CANEDataset.FS)`` (5000 for 10 s).
        :return: Tensor of preprocessed EEG chunks with shape
                 (num_chunks, num_channels, chunk_samples) where num_channels is 1 for single
                 channel or 8 for all channels.
        :rtype: torch.Tensor
        """
        if chunk_samples is None:
            chunk_samples = int(CHUNK_DURATION_SEC * CANEDataset.FS)
        df = pd.read_csv(file_path)
        try:
            CANEDataset.verify_sampling_rate(df)
        except ValueError as exc:
            raise ValueError(f"Error while processing {file_path}") from exc

        # Step 1: Initial z-score normalization of raw ADC values
        for ch in CANEDataset.CHANNELS:
            if ch in df.columns:
                df[ch] = zscore(df[ch])

        # Step 2: Common Average Reference (before filtering to remove common noise)
        if apply_car and all(ch in df.columns for ch in CANEDataset.CHANNELS):
            car = df[CANEDataset.CHANNELS].mean(axis=1)
            for ch in CANEDataset.CHANNELS:
                df[ch] = df[ch] - car
            logger.info("Applied Common Average Reference")

        df.rename(columns=CANEDataset.CHANNEL_MAPPING, inplace=True)

        # Determine which channels to use
        if channel == "in-ear":
            # Bipolar derivation: process T7 and T8 individually (preserves filter integrity),
            # then subtract. Sign flip augmentation is applied in SpectrogramDataset (per-epoch).
            channels_to_use = ["T7", "T8"]
            logger.info("Computing in-ear bipolar derivation (T8 - T7)")
        elif channel == "all":
            # Use all 8 channels in canonical order
            channels_to_use = CANE_CHANNEL_ORDER
            logger.info(f"Loading all {len(channels_to_use)} channels in canonical order")
        elif channel is not None:
            channels_to_use = [channel]
            logger.info(f"Picking {channel} channel")
        else:
            raise RuntimeError("Must specify 'channel' parameter")

        # Process each channel through the preprocessing pipeline
        processed_channels = []
        for ch in channels_to_use:
            signal = cast(np.ndarray, df[ch].values.astype(float))
            processed_signal = CANEDataset._process_single_channel(
                signal,
                ch,
                file_path,
                artifact_method,
                artifact_threshold,
                skip_artifact_removal,
            )
            processed_channels.append(processed_signal)

        # For in-ear: subtract filtered channels to get bipolar derivation
        if channel == "in-ear":
            # processed_channels = [T7_filtered, T8_filtered]
            inear_signal = processed_channels[1] - processed_channels[0]  # T8 - T7
            processed_channels = [inear_signal]

        # Step 7: Chunk the signals at native sampling rate
        # Stack all channels together: (num_channels, n_samples)
        multi_channel_signal = np.stack(processed_channels, axis=0)
        num_channels = multi_channel_signal.shape[0]
        n_samples = multi_channel_signal.shape[1]

        chunks_list: list[torch.Tensor] = []
        logger.info(f"Using sampling rate: {CANEDataset.FS} Hz")
        logger.info(f"Chunk size: {chunk_samples} samples")

        for i in range(0, n_samples - chunk_samples + 1, chunk_samples):
            chunk = multi_channel_signal[:, i : i + chunk_samples]  # (num_channels, chunk_samples)

            # Optional: Skip chunks with too many residual artifacts (check all channels)
            if skip_extreme_artifacts:
                skip_chunk = False
                for ch_idx in range(num_channels):
                    chunk_z = np.abs(zscore(chunk[ch_idx]))
                    if (chunk_z > artifact_threshold * 0.95).sum() / len(chunk[ch_idx]) > 0.1:
                        logger.warning(
                            f"Skipping chunk {len(chunks_list)} due to excessive "
                            f"artifacts in channel {ch_idx}"
                        )
                        skip_chunk = True
                        break
                if skip_chunk:
                    continue

            # Final z-score normalization per chunk per channel (matching MDD)
            chunk_normalized = np.zeros_like(chunk)
            for ch_idx in range(num_channels):
                chunk_normalized[ch_idx] = zscore(chunk[ch_idx])

            # Shape is already (num_channels, samples) to match MDD format
            chunks_list.append(torch.from_numpy(chunk_normalized).float())

        # Stack into tensor (num_chunks, num_channels, samples)
        if chunks_list:
            chunks_tensor = torch.stack(chunks_list)
            logger.info(f"Created {len(chunks_list)} chunks of shape {chunks_tensor.shape}")
            return chunks_tensor
        else:
            logger.error("No valid chunks created!")
            return torch.empty(0, num_channels, chunk_samples)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get file from the dataset preprocessed by self.transform method.

        :param int idx: Index of the file to retrieve.
        :return: Dictionary containing 'eeg' (tensor of shape (num_chunks, num_channels, samples)
                 where num_channels is 1 for single channel or 8 for all channels),
                 'label' (integer: 0=normal, 1=anxiety, 2=depression, 3=anxiety+depression),
                 'subject' (subject ID string), 'condition' (condition string: ec/eo), and
                 'channels' (list of channel names in the order they appear in the tensor).
        :rtype: dict
        """
        file_info = self.files[idx]

        chunks = self._load_and_preprocess_cane_raw_file(
            file_info["path"],
            self.channel,
            self.artifact_method,
            self.artifact_threshold,
            self.apply_car,
            self.skip_extreme_artifacts,
            self.skip_artifact_removal,
            self.chunk_samples,
        )

        if self.transform:
            eeg_tensor = self.transform(chunks)
        else:
            eeg_tensor = chunks

        return {
            "eeg": eeg_tensor,
            "label": file_info["label_int"],
            "subject": file_info["subject"],
            "condition": file_info["condition"],
            "channels": self.channel_names,
        }

    def get_sorted_subjects(self) -> list[str]:
        """
        Get a list of unique subjects in the dataset.

        :return: Sorted list of unique subject IDs.
        :rtype: list[str]
        """
        return sorted(list(set(f["subject"] for f in self.files)))

    def get_statistics(self) -> dict[str, Any]:
        """
        Get dataset statistics.

        :return: Dictionary containing total files, normal/anxiety file counts,
                 conditions breakdown, and number of unique subjects.
        :rtype: dict[str, Any]
        """
        normal_count = sum(1 for f in self.files if f["label"] == "normals")
        anxiety_count = sum(1 for f in self.files if f["label"] == "anxiety")

        conditions: dict[str, int] = {}
        for f in self.files:
            conditions[f["condition"]] = conditions.get(f["condition"], 0) + 1

        return {
            "total_files": len(self.files),
            "normal_files": normal_count,
            "anxiety_files": anxiety_count,
            "conditions": conditions,
            "subjects": len(self.get_sorted_subjects()),
        }


class SADDataset(MDDDataset):
    """
    PyTorch Dataset for SAD EEG data (anxiety detection).

    Inherits preprocessing pipeline from MDDDataset. Supports both normal and anxious classes.
    Files are organized in: {normals|anxious}/{ec|eo}/C{N}.edf
    """

    FS = 256  # Hz (vs MDD's 250 Hz)
    CHUNK_SAMPLES = int(CHUNK_DURATION_SEC * FS)  # 2560 samples per chunk

    CLASS_DIRECTORIES = ["normals", "anxious"]
    LABEL_MAP = {"normals": "H", "anxious": "AX"}
    LABEL_INT_MAP_SAD = {
        "normals": CanonicalLabel.HEALTHY,
        "anxious": CanonicalLabel.ANXIETY_ONLY,
    }

    CHANNEL_MAPPING = {
        "Fp1": "Fp1",
        "Fp2": "Fp2",
        "C3": "C3",
        "Cz": "Cz",
        "C4": "C4",
        "T7": "T7",
        "T8": "T8",
        "O2": "O2",
    }

    def __init__(
        self,
        data_dir: Path = SAD_DIR,
        condition: Optional[Literal["EC", "EO", "TASK"]] = None,
        subjects: Optional[list[str]] = None,
        cache_size: Optional[int] = DEFAULT_CACHE_SIZE,
        transform: Optional[Callable] = None,
        channel: Optional[str] = "all",
        test_mode: bool = False,
        chunk_duration: float = CHUNK_DURATION_SEC,
    ):
        """
        Initialize the SAD EEG dataset.

        :param Path data_dir: Path to directory containing class subdirectories.
        :param Optional[Literal["EC", "EO", "TASK"]] condition: Filter by condition or None for all.
        :param Optional[list[str]] subjects: List of subject IDs to include. None = all.
        :param int cache_size: Number of preprocessed files to cache in memory.
        :param Optional[Callable] transform: Optional transform function to apply to EEG data.
        :param str channel: Channel to use: specific channel name or "all" for all 8 channels.
        :param bool test_mode: If True, load only C1.edf files (one per class) for debugging.
        :param float chunk_duration: EEG chunk duration in seconds (default: CHUNK_DURATION_SEC).
        """
        self.data_dir = Path(data_dir)
        self.condition = condition
        self.transform = transform
        self.test_mode = test_mode
        self.chunk_samples = int(chunk_duration * self.FS)
        self.files = self._discover_files(
            cast(Literal["EC", "EO", "TASK"], condition.upper()) if condition is not None else None,
            subjects,
            None,  # labels parameter ignored
        )

        # Handle channel selection
        self.channel = channel
        self.channel_names = (
            SAD_CHANNEL_ORDER
            if channel == "all"
            else ["in-ear"]
            if channel == "in-ear"
            else ([channel] if channel else [])
        )

        # Bind dataset-specific parameters to the preprocessing function
        preprocess_func = partial(
            load_and_preprocess_edf_file,
            channel_mapping=self.CHANNEL_MAPPING,
            channel_order=SAD_CHANNEL_ORDER,
            chunk_samples=self.chunk_samples,
        )

        # Set up caching (same pattern as MDDDataset)
        self._load_and_preprocess_edf_raw_file_cached: Any
        if cache_size:
            self._load_and_preprocess_edf_raw_file_cached = lru_cache(maxsize=cache_size)(
                preprocess_func
            )
        else:
            self._load_and_preprocess_edf_raw_file_cached = preprocess_func

    def _discover_files(
        self,
        condition: Optional[Literal["EC", "EO", "TASK"]],
        subjects: Optional[list[str]],
        labels: Optional[list[str]],  # Ignored - class determined by directory
    ) -> list[dict[str, Any]]:
        """
        Discover all .edf files matching the criteria.

        Files are organized in: {normals|anxious}/{ec|eo}/C{N}.edf

        :param Optional[Literal["EC", "EO"]] condition: Condition filter or None.
        :param Optional[list[str]] subjects: List of subject IDs to include or None for all.
        :param Optional[list[str]] labels: Ignored (class determined by directory).
        :return: List of dictionaries containing file metadata.
        :rtype: list[dict]
        """
        files = []
        if self.test_mode:
            pattern = re.compile(r"C(1)\.edf")
        else:
            pattern = re.compile(r"C(\d+)\.edf")

        for class_dir in self.CLASS_DIRECTORIES:
            class_path = self.data_dir / class_dir
            if not class_path.exists():
                continue

            label = self.LABEL_MAP[class_dir]
            label_int = self.LABEL_INT_MAP_SAD[class_dir]

            for condition_dir in ["ec", "eo"]:
                if condition and condition_dir.upper() != condition:
                    continue

                condition_path = class_path / condition_dir
                if not condition_path.exists():
                    continue

                for file_path in sorted(condition_path.glob("*.edf")):
                    match = pattern.match(file_path.name)
                    if not match:
                        logger.warning(f"Skipping file with unexpected name: {file_path.name}")
                        continue

                    subject_num = match.group(1)
                    subject_id = f"{label} S{subject_num} {condition_dir.upper()}"

                    # Filter by subjects if specified
                    if subjects and subject_id not in subjects:
                        continue

                    files.append(
                        {
                            "path": file_path,
                            "label": class_dir,
                            "subject": subject_id,
                            "condition": condition_dir.upper(),
                            "label_int": label_int,
                        }
                    )

        MDDDataset._log_file_discovery(files, "SAD", self.data_dir)
        return files

    def get_statistics(self) -> dict[str, Any]:
        """
        Get dataset statistics.

        :return: Dictionary containing total files, normal/anxiety counts,
                 conditions breakdown, and number of unique subjects.
        :rtype: dict[str, Any]
        """
        conditions: dict[str, int] = Counter(f["condition"] for f in self.files)
        normal_files = sum(1 for f in self.files if f["label"] == "normals")
        ax_files = sum(1 for f in self.files if f["label"] == "anxious")

        return {
            "total_files": len(self.files),
            "normal_files": normal_files,
            "ax_files": ax_files,
            "conditions": conditions,
            "subjects": len(self.get_sorted_subjects()),
        }


class IDUNDataset(Dataset):
    """
    PyTorch Dataset for IDUN in-ear EEG data.

    Loads real in-ear EEG recordings from the IDUN device. Single-channel CSV format.
    Designed as a drop-in replacement for CANEDataset when ``--channel in-ear`` is used,
    providing real in-ear recordings instead of synthetic bipolar derivations.
    """

    FS = 250  # Hz (verified from timestamp deltas)
    CHUNK_SAMPLES = int(CHUNK_DURATION_SEC * FS)  # 2500 samples per 10s chunk

    CLASS_DIRECTORIES = ["normals", "anxiety", "depression", "comorbid"]
    LABEL_MAP: dict[str, str] = {
        "normals": "H",
        "anxiety": "AX",
        "depression": "DEP",
        "comorbid": "AXDEP",
    }

    # Subjects with non-standard filenames (no condition suffix) — skip them
    SKIP_SUBJECTS: set[str] = {"1001", "1002"}

    def __init__(
        self,
        data_dir: Path = IDUN_DIR,
        condition: Optional[Literal["ec", "eo"]] = None,
        subjects: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
        cache_size: Optional[int] = DEFAULT_CACHE_SIZE,
        transform: Optional[Callable] = None,
        quality_threshold: float = 0.0,
        test_mode: bool = False,
        chunk_duration: float = CHUNK_DURATION_SEC,
    ):
        """
        Initialize the IDUN in-ear EEG dataset.

        :param Path data_dir: Path to IDUN dataset root directory.
        :param Optional[Literal["ec", "eo"]] condition: Filter by condition or None for all.
        :param Optional[list[str]] subjects: List of subject IDs to include. None = all.
        :param Optional[list[str]] labels: List of class labels to include
               (e.g., ["normals", "anxiety"]). None = all.
        :param Optional[int] cache_size: Number of preprocessed files to cache in memory.
        :param Optional[Callable] transform: Optional transform function to apply to EEG data.
        :param float quality_threshold: Reject chunks where quality drops below this value.
               Default 0.0 rejects only chunks with quality=0 (unmeasured).
        :param bool test_mode: If True, load only one file per class for debugging.
        :param float chunk_duration: EEG chunk duration in seconds (default: CHUNK_DURATION_SEC).
        """
        self.data_dir = Path(data_dir)
        self.condition = condition
        self.transform = transform
        self.quality_threshold = quality_threshold
        self.test_mode = test_mode
        self.chunk_samples = int(chunk_duration * self.FS)
        self.channel_names: list[str] = ["in-ear"]

        self.files = self._discover_files(condition, subjects, labels)

        preprocess_func = IDUNDataset.load_and_preprocess_idun_file
        if cache_size:
            self._load_cached = lru_cache(maxsize=cache_size)(preprocess_func)
        else:
            self._load_cached = preprocess_func

    def _discover_files(
        self,
        condition: Optional[Literal["ec", "eo"]],
        subjects: Optional[list[str]],
        labels: Optional[list[str]],
    ) -> list[dict[str, Any]]:
        """
        Discover all IDUN EEG CSV files matching the criteria.

        Files are organized as: ``{class_dir}/{subject_id}/eeg_{subject_id}{condition}.csv``
        with case-insensitive condition matching.

        :param Optional[Literal["ec", "eo"]] condition: Condition filter or None.
        :param Optional[list[str]] subjects: List of subject IDs to include or None for all.
        :param Optional[list[str]] labels: List of class labels to include or None for all.
        :return: List of dictionaries containing file metadata.
        :rtype: list[dict[str, Any]]
        """
        files: list[dict[str, Any]] = []
        condition_map = {"ec": "EC", "eo": "EO"}

        # Test mode subjects: one per class
        # normals=0012, anxiety=0041, depression=0016, comorbid=0013
        test_mode_subjects = {"0012", "0041", "0016", "0013"}

        for class_dir in self.CLASS_DIRECTORIES:
            class_path = self.data_dir / class_dir
            if not class_path.exists():
                continue

            # Filter by labels
            if labels and class_dir not in labels:
                continue

            label = self.LABEL_MAP[class_dir]
            label_int = LABEL_INT_MAP[class_dir]

            for subject_dir in sorted(class_path.iterdir()):
                if not subject_dir.is_dir():
                    continue

                subject_num = subject_dir.name

                # Skip subjects with non-standard filenames
                if subject_num in self.SKIP_SUBJECTS:
                    logger.warning(
                        f"Skipping IDUN subject {subject_num} in {class_dir}: "
                        f"non-standard filename format"
                    )
                    continue

                # Test mode: only load specific subjects
                if self.test_mode and subject_num not in test_mode_subjects:
                    continue

                # Find EEG files with case-insensitive condition matching
                eeg_pattern = re.compile(
                    rf"eeg_{re.escape(subject_num)}(ec|eo|erp)\.csv", re.IGNORECASE
                )

                for file_path in sorted(subject_dir.glob("eeg_*.csv")):
                    match = eeg_pattern.match(file_path.name)
                    if not match:
                        logger.warning(f"Skipping IDUN file with unexpected name: {file_path}")
                        continue

                    file_condition = match.group(1).lower()

                    # Skip erp condition
                    if file_condition == "erp":
                        continue

                    # Filter by condition
                    if condition and file_condition != condition:
                        continue

                    cond_upper = condition_map.get(file_condition, file_condition.upper())
                    subject_id = f"{label} S{subject_num} {cond_upper}"

                    # Filter by subjects
                    if subjects and subject_id not in subjects:
                        continue

                    # Look for quality file
                    quality_path = file_path.parent / file_path.name.replace("eeg_", "quality_")
                    if not quality_path.exists():
                        # Try case variations
                        for qf in file_path.parent.glob("quality_*"):
                            if qf.name.lower() == quality_path.name.lower():
                                quality_path = qf
                                break

                    files.append(
                        {
                            "path": file_path,
                            "quality_path": quality_path if quality_path.exists() else None,
                            "label": class_dir,
                            "subject": subject_id,
                            "condition": file_condition,
                            "label_int": label_int,
                        }
                    )

        MDDDataset._log_file_discovery(files, "IDUN", self.data_dir)
        return files

    @staticmethod
    def load_and_preprocess_idun_file(
        file_path: Path,
        quality_path: Optional[Path] = None,
        quality_threshold: float = 30.0,  # https://sdk-docs.idunguardian.com/data-analysis.html
        chunk_samples: int = 2500,
    ) -> torch.Tensor:
        """
        Load an IDUN CSV file, preprocess it, and return chunks.

        Preprocessing pipeline: z-score raw → detrend → bandpass 1-70 Hz → notch 50 Hz
        → chunk into N-sample segments → per-chunk z-score → quality-based rejection.

        :param Path file_path: Path to the EEG CSV file.
        :param Optional[Path] quality_path: Path to the corresponding quality CSV file.
        :param float quality_threshold: Reject chunks where quality is at or below this value.
               Default 0.0 rejects only unmeasured (quality=0) chunks.
        :param int chunk_samples: Number of samples per chunk (default: 2500 = 10 s × 250 Hz).
        :return: Tensor of preprocessed EEG chunks with shape (num_chunks, 1, chunk_samples).
        :rtype: torch.Tensor
        """
        df = pd.read_csv(file_path)
        signal = cast(np.ndarray, df["ch1"].values.astype(np.float64))
        timestamps = df["timestamp"].values

        # Verify sampling rate
        duration = timestamps[-1] - timestamps[0]
        detected_fs = (len(signal) - 1) / duration
        if abs(detected_fs - IDUNDataset.FS) > 5:
            logger.warning(
                f"IDUN file {file_path.name}: detected Fs={detected_fs:.1f} Hz, "
                f"expected {IDUNDataset.FS} Hz"
            )

        # Load quality data if available
        quality_timestamps: Optional[np.ndarray] = None
        quality_values: Optional[np.ndarray] = None
        if quality_path is not None:
            try:
                qdf = pd.read_csv(quality_path)
                quality_timestamps = cast(np.ndarray, qdf["timestamp"].values)
                quality_values = cast(np.ndarray, qdf["signalQuality"].values)
                logger.info(
                    f"Quality stats for {file_path.name}: "
                    f"mean={quality_values.mean():.1f}, "
                    f"min={quality_values.min():.1f}, "
                    f"max={quality_values.max():.1f}"
                )
            except Exception as e:
                logger.warning(f"Failed to load quality file {quality_path}: {e}")

        # Step 1: Z-score normalize raw values
        signal = zscore(signal)

        # Step 2: Detrend (linear)
        signal = detrend(signal, type="linear")

        # Step 3: Bandpass filter 1-70 Hz
        nyq = 0.5 * IDUNDataset.FS
        b_bp, a_bp = butter(4, [1.0 / nyq, 70.0 / nyq], btype="band")
        signal = filtfilt(b_bp, a_bp, signal)

        # Step 4: Notch filter 50 Hz
        b_notch, a_notch = iirnotch(50.0, Q=30, fs=IDUNDataset.FS)
        signal = filtfilt(b_notch, a_notch, signal)

        # Step 5: Chunk into segments
        n_samples = len(signal)
        chunks_list: list[torch.Tensor] = []
        skipped_quality = 0

        for i in range(0, n_samples - chunk_samples + 1, chunk_samples):
            chunk_start_time = timestamps[i]
            chunk_end_time = timestamps[min(i + chunk_samples - 1, n_samples - 1)]

            # Quality check: reject chunks with low quality
            if quality_timestamps is not None and quality_values is not None:
                # Find quality samples within this chunk's time range
                mask = (quality_timestamps >= chunk_start_time) & (
                    quality_timestamps <= chunk_end_time
                )
                chunk_quality = quality_values[mask]

                if len(chunk_quality) > 0:
                    min_quality = chunk_quality.min()
                    if min_quality <= quality_threshold:
                        skipped_quality += 1
                        logger.warning(
                            f"Skipping chunk {len(chunks_list) + skipped_quality} "
                            f"from {file_path.name}: quality={min_quality:.1f} "
                            f"<= threshold={quality_threshold}"
                        )
                        continue

            # Per-chunk z-score normalization
            chunk = signal[i : i + chunk_samples]
            chunk = zscore(chunk)

            # Shape: (1, chunk_samples) — single channel
            chunk_tensor = torch.from_numpy(chunk.reshape(1, -1)).float()
            chunks_list.append(chunk_tensor)

        if skipped_quality > 0:
            logger.info(
                f"Rejected {skipped_quality} chunks from {file_path.name} due to low quality"
            )

        if chunks_list:
            chunks_tensor = torch.stack(chunks_list)
            logger.info(
                f"IDUN {file_path.name}: created {len(chunks_list)} chunks "
                f"of shape {chunks_tensor.shape}"
            )
            return chunks_tensor
        else:
            logger.warning(f"No valid chunks created from {file_path.name}")
            return torch.empty(0, 1, chunk_samples)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get preprocessed EEG data for a single file.

        :param int idx: Index of the file to retrieve.
        :return: Dictionary containing 'eeg' (tensor), 'label' (int), 'subject' (str),
                 'condition' (str), and 'channels' (list[str]).
        :rtype: dict[str, Any]
        """
        file_info = self.files[idx]

        chunks = self._load_cached(
            file_info["path"],
            file_info.get("quality_path"),
            self.quality_threshold,
            self.chunk_samples,
        )

        if self.transform:
            eeg_tensor = self.transform(chunks)
        else:
            eeg_tensor = chunks

        return {
            "eeg": eeg_tensor,
            "label": file_info["label_int"],
            "subject": file_info["subject"],
            "condition": file_info["condition"],
            "channels": self.channel_names,
        }

    def get_sorted_subjects(self) -> list[str]:
        """
        Get a sorted list of unique subjects in the dataset.

        :return: Sorted list of unique subject IDs.
        :rtype: list[str]
        """
        return sorted(set(f["subject"] for f in self.files))

    def get_statistics(self) -> dict[str, Any]:
        """
        Get dataset statistics.

        :return: Dictionary containing total files, per-class file counts,
                 conditions breakdown, and number of unique subjects.
        :rtype: dict[str, Any]
        """
        class_counts: dict[str, int] = Counter(f["label"] for f in self.files)
        condition_counts: dict[str, int] = Counter(f["condition"] for f in self.files)

        return {
            "total_files": len(self.files),
            "class_counts": dict(class_counts),
            "conditions": dict(condition_counts),
            "subjects": len(self.get_sorted_subjects()),
        }


class SpectrogramDataset(Dataset):
    """
    Wrapper dataset that converts raw EEG data to spectrograms on-the-fly.

    Takes an MDDDataset and converts EEG chunks to spectrograms using Short-Time Fourier
    Transform (STFT). Currently, extracts only the first channel from multichannel EEG data.
    """

    def __init__(
        self,
        dataset: MDDDataset | CANEDataset | IDUNDataset,
        fs: float = MDDDataset.FS,
        nperseg: int = 256,
        noverlap: int = 192,
        window: str = "hamming",
        cache_size: int = DEFAULT_CACHE_SIZE,
        augmentation: Optional[Callable] = None,
        channel: Optional[str] = "all",
    ):
        """
        Initialize the SpectrogramDataset.

        :param dataset: Underlying dataset providing raw EEG data.
        :param float fs: Sampling frequency for STFT (default: 250 Hz).
        :param int nperseg: Length of each segment for STFT (default: 256).
        :param int noverlap: Number of points to overlap between segments (default: 192).
        :param str window: Window function for STFT (default: "hamming").
        :param int cache_size: Number of processed items to cache in memory.
        :param Optional[Callable] augmentation: Optional augmentation to apply to raw EEG
               before spectrogram conversion. When enabled, caching is disabled to ensure
               fresh augmentations on each access.
        :param Optional[str] channel: Channel mode. When "in-ear", applies 50%% sign flip
               augmentation per chunk to handle polarity ambiguity (always enabled regardless
               of augmentation).
        """
        self.dataset = dataset
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window
        self.augmentation = augmentation
        self.is_inear: bool = channel == "in-ear"

        # Disable caching only when augmentation is enabled (requires fresh randomness).
        if augmentation is None:
            self._get_item_cached = lru_cache(maxsize=cache_size)(self._get_item)
        else:
            self._get_item_cached = self._get_item

    @staticmethod
    def convert_to_spectrograms(
        tensor: torch.Tensor,
        nperseg: int,
        fs: float,
        noverlap: int,
        window: str,
        augmentation: Optional[Callable] = None,
        is_inear: bool = False,
    ) -> list[torch.Tensor]:
        """
        Convert EEG tensor to list of spectrograms using STFT.

        Expects input tensor of shape (num_chunks, num_channels, samples).
        Processes channels to create spectrograms.

        :param torch.Tensor tensor: EEG tensor with shape (num_chunks, num_channels, samples).
        :param int nperseg: Length of each segment for STFT.
        :param float fs: Sampling frequency for STFT.
        :param int noverlap: Number of points to overlap between segments.
        :param str window: Window function for STFT.
        :param Optional[Callable] augmentation: Optional augmentation to apply to raw EEG
               before STFT conversion.
        :param bool is_inear: If True, applies 50%% sign flip per chunk to handle in-ear
               polarity ambiguity (independent of augmentation flag).
        :return: List of spectrogram tensors, each with shape
                 (num_channels, freq_bins, time_frames).
        :rtype: list[torch.Tensor]
        """
        spectograms = []
        num_chunks = tensor.size(0)
        num_channels = tensor.size(1)

        for chunk_idx in range(num_chunks):
            chunk_data = tensor[chunk_idx]  # (num_channels, samples)

            channel_spectrograms = []
            for ch_idx in range(num_channels):
                channel_data = chunk_data[ch_idx].numpy()

                # Sign flip for in-ear: 50% probability per chunk, handles polarity ambiguity
                # if is_inear and np.random.random() < 0.5:
                #    channel_data = -channel_data

                # Apply augmentation to raw EEG before STFT
                if augmentation is not None:
                    channel_data = augmentation(channel_data)

                # Skip if too short for STFT
                if len(channel_data) < nperseg:
                    logger.warning(
                        f"Skipping chunk {chunk_idx} channel {ch_idx} due to "
                        f"insufficient length: {len(channel_data)}"
                    )
                    break

                # Compute STFT
                f, t, Zxx = stft(
                    channel_data,
                    fs=fs,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    window=window,
                )

                # Log magnitude spectrogram
                Zxx_mag = np.log1p(np.abs(Zxx))
                channel_spectrograms.append(Zxx_mag)

            # Only add if all channels processed successfully
            if len(channel_spectrograms) == num_channels:
                # Stack: (num_channels, H, W)
                spec_tensor = torch.tensor(np.stack(channel_spectrograms), dtype=torch.float32)
                spectograms.append(spec_tensor)

        if spectograms:
            logger.debug(
                f"Created {len(spectograms)} spectrograms with shape {spectograms[0].shape}"
            )
        else:
            logger.warning("No valid spectrograms created!")

        return spectograms

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_item(self, idx: int) -> tuple[list[torch.Tensor], int, str]:
        """
        Internal method to get a single item (used by LRU cache).

        :param int idx: Index of the sample.
        :return: Tuple of (list of spectrograms, label, subject).
        :rtype: tuple[list[torch.Tensor], int, str]
        """
        item = self.dataset.__getitem__(idx)
        spectograms = self.convert_to_spectrograms(
            item["eeg"],
            self.nperseg,
            self.fs,
            self.noverlap,
            self.window,
            self.augmentation,
            self.is_inear,
        )
        return spectograms, item["label"], item["subject"]

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], int, str]:
        """
        Get spectrograms-label-subject tuple for a given index.

        :param int idx: Index of the sample.
        :return: Tuple of (list of spectrograms, label, subject).
        :rtype: tuple[list[torch.Tensor], int, str]
        """
        return cast(tuple[list[torch.Tensor], int, str], self._get_item_cached(idx))

    def get_indices_for_subjects(self, subject_list: list[str]) -> list[int]:
        """
        Get indices of all samples belonging to specific subjects.

        :param list[str] subject_list: List of subject IDs (e.g., ["H S1", "MDD S2"]).
        :return: List of indices for samples belonging to those subjects.
        :rtype: list[int]
        """
        subject_set = set(subject_list)
        indices = []
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            if item["subject"] in subject_set:
                indices.append(i)
        return indices


class FlattenedSpectrogramDataset(Dataset):
    """
    Wrapper that flattens SpectrogramDataset to operate on individual chunks.

    SpectrogramDataset returns (list[spectrograms], label, subject) where each item
    represents all chunks from a file. This wrapper flattens it so each index
    corresponds to a single chunk/spectrogram.
    """

    def __init__(
        self,
        spectrogram_dataset: SpectrogramDataset,
        label_mapping: Optional[dict[int, int]] = None,
    ):
        """
        Initialize the flattened dataset.

        :param SpectrogramDataset spectrogram_dataset: The underlying spectrogram dataset.
        :param Optional[dict[int, int]] label_mapping: Optional mapping to remap labels.
               E.g., {0: 0, 2: 1} remaps label 2 to 1 for binary classification.
        """
        self.dataset = spectrogram_dataset
        self.label_mapping = label_mapping
        # Build index mapping: (file_idx, chunk_idx) for each chunk
        self.index: list[tuple[int, int]] = []

        logger.info("Building chunk index for FlattenedSpectrogramDataset...")
        for file_idx in range(len(spectrogram_dataset)):
            spec_list, _, _ = spectrogram_dataset[file_idx]
            for chunk_idx in range(len(spec_list)):
                self.index.append((file_idx, chunk_idx))
        logger.info(f"Built index with {len(self.index)} total chunks")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """
        Get a single chunk/spectrogram.

        :param int idx: Index of the chunk to retrieve.
        :return: Tuple of (spectrogram, label, subject).
        :rtype: tuple[torch.Tensor, int, str]
        """
        file_idx, chunk_idx = self.index[idx]
        spec_list, label, subject = self.dataset[file_idx]

        # Apply label remapping if configured
        if self.label_mapping is not None:
            label = self.label_mapping.get(label, label)

        return spec_list[chunk_idx], label, subject

    def get_indices_for_subjects(self, subject_list: list[str]) -> list[int]:
        """
        Get indices of all chunks belonging to specific subjects.

        :param list[str] subject_list: List of subject IDs (e.g., ["H S1", "MDD S2"]).
        :return: List of indices for chunks belonging to those subjects.
        :rtype: list[int]
        """
        # Get file-level indices from underlying dataset
        file_indices = set(self.dataset.get_indices_for_subjects(subject_list))

        # Find all chunk indices that belong to those files
        chunk_indices = []
        for chunk_idx, (file_idx, _) in enumerate(self.index):
            if file_idx in file_indices:
                chunk_indices.append(chunk_idx)

        return chunk_indices


def collate_spectrograms(
    batch: list[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Custom collate function to handle potentially variable-sized spectrograms.

    Each item in the batch is a tuple of (spectrogram, label, subject).
    This function stacks them into proper batches.

    :param list batch: List of tuples (spectrogram, label, subject) from
           FlattenedSpectrogramDataset.__getitem__()
    :return: Tuple of (stacked spectrograms, labels, subjects).
    :rtype: tuple[torch.Tensor, torch.Tensor, list[str]]
    """
    spectrograms = []
    labels = []
    subjects = []

    for spectrogram, label, subject in batch:
        spectrograms.append(spectrogram)
        labels.append(label)
        subjects.append(subject)

    shapes = [s.shape for s in spectrograms]
    assert len(set(shapes)) == 1, f"Inconsistent shapes: {set(shapes)}"

    # Stack all spectrograms into a single tensor
    spectrograms_tensor = torch.stack(spectrograms)  # Shape: (batch_size, 1, H, W)
    labels_tensor = torch.tensor(labels)

    return spectrograms_tensor, labels_tensor, subjects


class FlattenedRawEEGDataset(Dataset):
    """
    Flat view of raw EEG chunks for time-series models such as Deformer.

    Wraps an underlying EEG dataset (MDDDataset, CANEDataset, IDUNDataset, SADDataset)
    directly — without the STFT step — and flattens its file-level items into individual
    chunks. Applies per-chunk z-score normalization per channel and resamples to a common
    target sample count when the source rate differs.

    Resampling strategy (target_samples=2500, i.e. 10 s @ 250 Hz):
      - 2500 samples (MDD, IDUN): used as-is.
      - 5000 samples (CANE, 500 Hz): downsampled 2:1 via scipy.signal.resample.
      - 2560 samples (AX_MALIK, 256 Hz): truncated to 2500 (last 60 samples dropped).
      - Any other length > target_samples * 1.5: resampled.
      - Any other length > target_samples: truncated.
    """

    def __init__(
        self,
        dataset: MDDDataset | CANEDataset | IDUNDataset,
        target_samples: int = 2500,
        label_mapping: Optional[dict[int, int]] = None,
        is_inear: bool = False,
    ) -> None:
        """
        Initialize the dataset and build the flat chunk index.

        :param dataset: Underlying EEG dataset providing raw chunks via __getitem__.
        :param int target_samples: Target number of time samples per chunk (default 2500).
        :param Optional[dict[int, int]] label_mapping: Optional label remapping dict,
               e.g. {0: 0, 1: 1, 2: 1, 3: 1} for binary collapse.
        :param bool is_inear: If True, applies 50%% sign flip per chunk to handle in-ear
               polarity ambiguity (electrode orientation is arbitrary).
        """
        self.dataset = dataset
        self.target_samples = target_samples
        self.label_mapping = label_mapping
        self.is_inear = is_inear

        # Build flat index: (file_idx, chunk_idx), and map file_idx → subject
        self.index: list[tuple[int, int]] = []
        self.file_subjects: dict[int, str] = {}

        logger.info("Building chunk index for FlattenedRawEEGDataset...")
        for file_idx in range(len(dataset)):
            item = dataset[file_idx]
            eeg: torch.Tensor = item["eeg"]  # (num_chunks, num_channels, samples)
            subject: str = item["subject"]
            self.file_subjects[file_idx] = subject
            for chunk_idx in range(eeg.shape[0]):
                self.index.append((file_idx, chunk_idx))
        logger.info(f"Built index with {len(self.index)} total chunks")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """
        Return a single normalized raw EEG chunk.

        :param int idx: Flat chunk index.
        :return: Tuple of (raw_chunk, label, subject) where raw_chunk has shape
                 (num_channels, target_samples).
        :rtype: tuple[torch.Tensor, int, str]
        """
        file_idx, chunk_idx = self.index[idx]
        item = self.dataset[file_idx]
        eeg: torch.Tensor = item["eeg"]  # (num_chunks, num_channels, samples)
        label: int = item["label"]
        subject: str = item["subject"]

        chunk = eeg[chunk_idx]  # (num_channels, samples)

        # Resample / truncate to target_samples if needed
        n_samples = chunk.shape[-1]
        if n_samples != self.target_samples:
            if n_samples > self.target_samples * 1.5:
                # Large ratio difference (e.g. CANE 5000 → 2500): resample
                chunk_np = resample(chunk.numpy(), self.target_samples, axis=-1)
                chunk = torch.from_numpy(chunk_np).float()
            elif n_samples > self.target_samples:
                # Small excess (e.g. AX_MALIK 2560 → 2500): truncate
                chunk = chunk[..., : self.target_samples]
            else:
                # chunk shorter than target — pad with zeros (edge case)
                pad = torch.zeros(
                    chunk.shape[0], self.target_samples - n_samples, dtype=chunk.dtype
                )
                chunk = torch.cat([chunk, pad], dim=-1)

        # Per-channel z-score normalization
        mean = chunk.mean(dim=-1, keepdim=True)
        std = chunk.std(dim=-1, keepdim=True).clamp(min=1e-8)
        chunk = (chunk - mean) / std

        # 50% sign flip for in-ear: electrode polarity is arbitrary, so train invariance
        if self.is_inear and np.random.random() < 0.5:
            chunk = -chunk

        if self.label_mapping is not None:
            label = self.label_mapping.get(label, label)

        return chunk, label, subject

    def get_indices_for_subjects(self, subject_list: list[str]) -> list[int]:
        """
        Return flat chunk indices belonging to the given subjects.

        :param list[str] subject_list: Subject IDs to select (e.g. ["H S1 EC", "MDD S3 EC"]).
        :return: List of flat chunk indices for those subjects.
        :rtype: list[int]
        """
        subject_set = set(subject_list)
        return [
            chunk_idx
            for chunk_idx, (file_idx, _) in enumerate(self.index)
            if self.file_subjects.get(file_idx) in subject_set
        ]
