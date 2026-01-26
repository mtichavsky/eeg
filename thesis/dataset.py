import logging
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import mne
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, detrend, filtfilt, iirnotch, stft
from scipy.stats import zscore
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

CANE_DIR = Path("/home/milan/Documents/diplomka/CANE/")
MDD_DIR = Path("/home/milan/Documents/diplomka/MDD/")

CHUNK_DURATION_SEC = 10

# Canonical channel order for multi-channel loading (consistent across datasets)
# T7/T8 are new nomenclature, equivalent to T3/T4 in older systems
CANONICAL_CHANNEL_ORDER = ["Fp1", "Fp2", "C3", "Cz", "C4", "T7", "T8", "O2/Oz"]

# Dataset-specific channel mappings to canonical order
MDD_CHANNEL_ORDER = ["Fp1", "Fp2", "C3", "Cz", "C4", "T7", "T8", "O2"]  # T3→T7, T4→T8
CANE_CHANNEL_ORDER = ["Fp1", "Fp2", "C3", "Cz", "C4", "T7", "T8", "Oz"]


class MDDDataset(Dataset):
    """
    PyTorch Dataset for MDD EEG data.
    """

    FS = 1000 / 4
    CHUNK_SAMPLES = int(CHUNK_DURATION_SEC * FS)

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

    def __init__(
        self,
        data_dir: Path = MDD_DIR,
        condition: Optional[Literal["EC", "EO", "TASK"]] = None,
        subjects: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
        cache_size: Optional[int] = 100,
        transform: Optional[Callable] = None,
        skip_ica: bool = True,
        channel: str = "all",
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
        :param bool skip_ica: If True, skip ICA artifact removal during preprocessing.
        :param Optional[str] channel: Channel to use: specific channel name (e.g., "Fp1") or
               "all" for all 8 channels (excludes A2-A1 reference).
        """
        self.data_dir = Path(data_dir)
        self.condition = condition
        self.transform = transform
        self.skip_ica = skip_ica
        self.files = self._discover_files(
            condition.upper() if condition is not None else None, subjects, labels
        )

        # Handle channel selection
        self.channel = channel
        self.channel_names = (
            MDD_CHANNEL_ORDER if channel == "all" else ([channel] if channel else [])
        )

        if cache_size:
            self._load_and_preprocess_mdd_raw_file_cached = lru_cache(maxsize=cache_size)(
                MDDDataset.load_and_preprocess_mdd_raw_file
            )
        else:
            self._load_and_preprocess_mdd_raw_file_cached = (
                MDDDataset.load_and_preprocess_mdd_raw_file
            )

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
                    "label_int": 0 if label == "H" else 1,  # H=0 (healthy), MDD=1
                }
            )
        return files

    @staticmethod
    def load_and_preprocess_mdd_raw_file(
        file_path: Path,
        channel: str,
    ) -> torch.Tensor:
        """
        Load an EDF file from disk, preprocess it, and return chunks. This contains all the
        preprocessing logic in this class.

        Applies the full preprocessing pipeline: filtering, artifact removal (no ICA), and chunking.

        :param Path file_path: Path to the EDF file to preprocess.
        :param str channel: Channel to use: specific channel name (e.g., "Fp1") or
               "all" for all 8 channels.
        :return: Tensor of preprocessed EEG chunks with shape (num_chunks, channels, samples).
        :rtype: torch.Tensor
        """
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw = raw.filter(l_freq=1, h_freq=70, method="iir", verbose=False)
        raw = raw.notch_filter(freqs=50, verbose=False)

        raw = raw.pick(list(MDDDataset.CHANNEL_MAPPING.keys()))
        raw = raw.rename_channels(MDDDataset.CHANNEL_MAPPING)

        if channel == "all":
            # Pick channels in canonical order (excludes A2-A1 reference)
            # This ensures consistent channel ordering across datasets
            raw = raw.pick(MDD_CHANNEL_ORDER)
        elif channel is not None:
            raw = raw.pick([channel])
        else:
            raise RuntimeError("Must specify 'channel' parameter")

        data = raw.get_data()
        n_samples = data.shape[1]
        chunks_list: list[np.ndarray] = []
        for i in range(0, n_samples - MDDDataset.CHUNK_SAMPLES + 1, MDDDataset.CHUNK_SAMPLES):
            chunk = data[:, i : i + MDDDataset.CHUNK_SAMPLES]
            chunk = zscore(chunk, axis=1)
            chunks_list.append(chunk)

        chunks_tensor = torch.stack([torch.from_numpy(chunk).float() for chunk in chunks_list])
        return chunks_tensor

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

        chunks = self._load_and_preprocess_mdd_raw_file_cached(file_info["path"], self.channel)

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

    CLASS_DIRECTORIES = ["normal", "anxiety", "depression", "anxiety-depression"]
    LABEL_MAP = {"normal": "H", "anxiety": "AX", "depression": "DEP", "anxiety-depression": "AXDEP"}
    LABEL_INT_MAP = {"normal": 0, "anxiety": 1, "depression": 2, "anxiety-depression": 3}
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
        cache_size: Optional[int] = 100,
        transform: Optional[Callable] = None,
        channel: Optional[str] = "Fp1",
        artifact_method: str = "interpolation",
        artifact_threshold: float = 4.0,
        apply_car: bool = True,
        skip_extreme_artifacts: bool = False,
        skip_artifact_removal: bool = False,
    ):
        """
        Initialize the CANE EEG dataset.

        :param Path data_dir: Path to directory containing .csv files.
        :param Optional[Literal["EC", "EO"]] condition: Filter by condition
               or None for all conditions.
        :param Optional[list[str]] subjects: List of subject IDs to include. None = all.
        :param Optional[list[str]] labels: List of labels to include (e.g., ["normal", "anxiety"]).
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

        # Set channel_names for verification
        if channel == "all":
            self.channel_names = CANE_CHANNEL_ORDER
        elif channel:
            self.channel_names = [channel]
        else:
            self.channel_names = []

        self.files = self._discover_files(condition, subjects, labels)

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
        {label}/{condition}/{subject_id}_{condition}.csv where label is "normal" or "anxiety",
        and condition is "ec" (eyes closed) or "eo" (eyes open).

        :param Optional[Literal["ec", "eo"]] condition: Condition filter ("ec", "eo") or None.
        :param Optional[list[str]] subjects: List of subject IDs to include or None for all.
        :param Optional[list[str]] labels: List of labels to include ("normal", "anxiety")
               or None for all.
        :return: List of dictionaries containing file metadata.
        :rtype: list[dict]
        """
        files = []
        # Pattern to match filenames like "0041_Ec.csv", "1005_EC.csv", or "1001EC.csv"
        # (with or without underscore)
        pattern = re.compile(r"(\d+)_?(EC|EO|Ec|Eo|ec|eo)\.csv", re.IGNORECASE)

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
                            "label_int": self.LABEL_INT_MAP[label_dir],
                        }
                    )

        logger.info(f"Discovered {len(files)} CANE files")
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
        :return: Tensor of preprocessed EEG chunks with shape
                 (num_chunks, num_channels, chunk_samples) where num_channels is 1 for single
                 channel or 8 for all channels.
        :rtype: torch.Tensor
        """
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
        if channel == "all":
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
            signal = df[ch].values.astype(float)
            processed_signal = CANEDataset._process_single_channel(
                signal,
                ch,
                file_path,
                artifact_method,
                artifact_threshold,
                skip_artifact_removal,
            )
            processed_channels.append(processed_signal)

        # Step 7: Chunk the signals at native sampling rate
        # Stack all channels together: (num_channels, n_samples)
        multi_channel_signal = np.stack(processed_channels, axis=0)
        num_channels = multi_channel_signal.shape[0]
        n_samples = multi_channel_signal.shape[1]

        chunks_list: list[torch.Tensor] = []
        chunk_samples = int(CHUNK_DURATION_SEC * CANEDataset.FS)
        logger.info(f"Using sampling rate: {CANEDataset.FS} Hz")
        logger.info(f"Chunk size: {chunk_samples} samples ({CHUNK_DURATION_SEC}s)")

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
        normal_count = sum(1 for f in self.files if f["label"] == "normal")
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


class SpectrogramDataset(Dataset):
    """
    Wrapper dataset that converts raw EEG data to spectrograms on-the-fly.

    Takes an MDDDataset and converts EEG chunks to spectrograms using Short-Time Fourier
    Transform (STFT). Currently, extracts only the first channel from multichannel EEG data.
    """

    def __init__(
        self,
        dataset: MDDDataset | CANEDataset,
        fs: float = MDDDataset.FS,
        nperseg: int = 256,
        noverlap: int = 192,
        window: str = "hamming",
        cache_size: int = 100,
        augmentation: Optional[Callable] = None,
    ):
        """
        Initialize the SpectrogramDataset.

        :param MDDDataset dataset: Underlying MDDDataset providing raw EEG data.
        :param float fs: Sampling frequency for STFT (default: 250 Hz).
        :param int nperseg: Length of each segment for STFT (default: 256).
        :param int noverlap: Number of points to overlap between segments (default: 192).
        :param str window: Window function for STFT (default: "hamming").
        :param int cache_size: Number of processed items to cache in memory.
        :param Optional[Callable] augmentation: Optional augmentation to apply to raw EEG
               before spectrogram conversion. When enabled, caching is disabled to ensure
               fresh augmentations on each access.
        """
        self.dataset = dataset
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window
        self.augmentation = augmentation

        # Disable caching when augmentation is enabled - each access should return
        # a fresh augmentation. Otherwise, use LRU cache for performance.
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
            logger.info(
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
        )
        return spectograms, item["label"], item["subject"]

    def __getitem__(self, idx: int) -> tuple[list[torch.Tensor], int, str]:
        """
        Get spectrograms-label-subject tuple for a given index.

        :param int idx: Index of the sample.
        :return: Tuple of (list of spectrograms, label, subject).
        :rtype: tuple[list[torch.Tensor], int, str]
        """
        return self._get_item_cached(idx)

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
