import logging
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import mne
import numpy as np
import torch
from mne.preprocessing import ICA
from mne_icalabel import label_components
from scipy.signal import stft
from scipy.stats import zscore
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

CANE_DIR = Path("../CANE/")
MDD_DIR = Path("/home/milan/Documents/diplomka/MDD/")

SEGMENT_LENGTH = 10
SFREQ = 1000 / 4
CHUNK_SAMPLES = int(SEGMENT_LENGTH * SFREQ)

logger = logging.getLogger(__name__)


class MDDDataset(Dataset):
    """
    PyTorch Dataset for MDD EEG data.
    """

    CHANNEL_MAPPING = {
        "EEG A2-A1": "A2-A1",
        "EEG C3-LE": "C3",
        "EEG C4-LE": "C4",
        "EEG Cz-LE": "Cz",
        "EEG Fp1-LE": "Fp1",
        "EEG Fp2-LE": "Fp2",
        "EEG O2-LE": "O2",
        "EEG T3-LE": "T3",
        "EEG T4-LE": "T4",
    }

    def __init__(
        self,
        data_dir: Path = MDD_DIR,
        condition: Optional[Literal["EC", "EO", "TASK"]] = None,
        subjects: Optional[list[str]] = None,
        labels: Optional[list[str]] = None,
        cache_size: Optional[int] = 100,
        transform: Optional[Callable] = None,
        skip_ica: bool = False,
        channel: Optional[str] = None,
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
        :param bool skip_ica: If True, skip ICA artifact removal (faster but less clean data).
        """
        self.data_dir = Path(data_dir)
        self.condition = condition
        self.transform = transform
        self.skip_ica = skip_ica
        self.files = self._discover_files(condition, subjects, labels)
        self.channel = channel
        if channel and not skip_ica:
            raise RuntimeError("When channel is chosen, skip ICA has to be set to True.")

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
            subject_id = f"{label} S{subject_num}"

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
        file_path: Path, channel: Optional[str] = None, skip_ica: bool = False
    ) -> torch.Tensor:
        """
        Load an EDF file from disk, preprocess it, and return chunks. This contains all the
        preprocessing logic in this class.

        Applies the full preprocessing pipeline: filtering, optional ICA, artifact removal,
        and chunking.

        :param Path file_path: Path to the EDF file to preprocess.
        :param bool skip_ica: If True, skip ICA artifact removal for faster processing.
        :return: Tensor of preprocessed EEG chunks with shape (num_chunks, channels, samples).
        :rtype: torch.Tensor
        """
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw = raw.filter(l_freq=1, h_freq=70, method="iir", verbose=False)
        raw = raw.notch_filter(freqs=50, verbose=False)

        raw = raw.pick(list(MDDDataset.CHANNEL_MAPPING.keys()))
        raw = raw.rename_channels(MDDDataset.CHANNEL_MAPPING)

        # Pick the channel if specified
        if channel is not None:
            raw = raw.pick([channel])
        else:
            raise RuntimeError("Not implemented properly yet")

        # Apply ICA if not skipped
        if not skip_ica:
            filt_raw = raw.set_eeg_reference("average", verbose=False)
            ica = ICA(
                max_iter="auto",
                method="infomax",
                random_state=97,
                fit_params=dict(extended=True),
            )
            ica.fit(filt_raw, verbose=False)

            montage = mne.channels.make_standard_montage("standard_1020")
            filt_raw = filt_raw.set_montage(montage, match_case=False, verbose=False)
            ic_labels = label_components(filt_raw, ica, method="iclabel")

            labels = ic_labels["labels"]
            exclude_idx = [
                idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
            ]

            preprocessed = filt_raw.copy()
            ica.apply(preprocessed, exclude=exclude_idx, verbose=False)
        else:
            preprocessed = raw

        data = preprocessed.get_data()
        n_samples = data.shape[1]
        chunks = []
        for i in range(0, n_samples - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
            chunk = data[:, i : i + CHUNK_SAMPLES]
            chunk = zscore(chunk, axis=1)
            chunks.append(chunk)

        chunks = torch.stack([torch.from_numpy(chunk).float() for chunk in chunks])
        return chunks

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get file from the dataset preprocessed by self.transform method

        :param int idx: Index of the file to retrieve.
        :return: Dictionary containing 'eeg' (tensor of shape (num_chunks, channels, samples)),
                 'label' (integer: 0=Healthy, 1=MDD), 'subject' (subject ID string),
                 and 'condition' (condition string: EC/EO/TASK).
        :rtype: dict
        """
        file_info = self.files[idx]

        chunks = self._load_and_preprocess_mdd_raw_file_cached(
            file_info["path"], self.channel, self.skip_ica
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


class SpectrogramDataset(Dataset):
    """
    Wrapper dataset that converts raw EEG data to spectrograms on-the-fly.

    Takes an MDDDataset and converts EEG chunks to spectrograms using Short-Time Fourier
    Transform (STFT). Currently, extracts only the first channel from multichannel EEG data.
    """

    def __init__(
        self,
        dataset: MDDDataset,
        fs: float = SFREQ,
        nperseg: int = 256,
        noverlap: int = 192,
        window: str = "hamming",
        cache_size: int = 100,
    ):
        """
        Initialize the SpectrogramDataset.

        :param MDDDataset dataset: Underlying MDDDataset providing raw EEG data.
        :param float fs: Sampling frequency for STFT (default: 250 Hz).
        :param int nperseg: Length of each segment for STFT (default: 256).
        :param int noverlap: Number of points to overlap between segments (default: 192).
        :param str window: Window function for STFT (default: "hamming").
        :param int cache_size: Number of processed items to cache in memory.
        """
        self.dataset = dataset
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.window = window

        # Create a bound method cache for this instance
        self._get_item_cached = lru_cache(maxsize=cache_size)(self._get_item)

    @staticmethod
    def convert_to_spectrograms(
        tensor: torch.Tensor, nperseg: int, fs: float, noverlap: int, window: str
    ) -> list[torch.Tensor]:
        """
        Convert EEG tensor to list of spectrograms using STFT.

        Expects input tensor of shape (num_chunks, channels, samples).
        Extracts only the first channel for spectrogram generation.

        :param torch.Tensor tensor: EEG tensor with shape (num_chunks, channels, samples).
        :param int nperseg: Length of each segment for STFT.
        :param float fs: Sampling frequency for STFT.
        :param int noverlap: Number of points to overlap between segments.
        :param str window: Window function for STFT.
        :return: List of spectrogram tensors, each with shape (1, freq_bins, time_frames).
        :rtype: list[torch.Tensor]
        """
        spectograms = []
        for chunk_idx in range(tensor.size(0)):
            # Extract the first channel for spectrogram generation
            channel_data = tensor[chunk_idx, 0, :].numpy()

            # Skip if too short for STFT
            if len(channel_data) < nperseg:
                logger.warning(f"Skipping sample due to insufficient length: {len(channel_data)}")
                continue

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

            # Convert to tensor: (1, H, W) for single-channel spectrogram
            spec_tensor = torch.tensor(Zxx_mag, dtype=torch.float32).unsqueeze(0)

            spectograms.append(spec_tensor)

        logger.info(f"Created {len(spectograms)} spectrograms")
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
            item["eeg"], self.nperseg, self.fs, self.noverlap, self.window
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

    def __init__(self, spectrogram_dataset: SpectrogramDataset):
        """
        Initialize the flattened dataset.

        :param SpectrogramDataset spectrogram_dataset: The underlying spectrogram dataset.
        """
        self.dataset = spectrogram_dataset
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


import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, iirnotch, detrend, stft
from scipy.stats import zscore
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# CANE dataset constants
CANE_SAMPLING_RATE = 500  # Hz (hardcoded, verified to be ~498-499 Hz in practice)
CANE_SAMPLING_RATE_TOLERANCE = 10  # Hz (allow ±10 Hz deviation)

# CANE STFT Parameters for matching MDD spectrogram shape (129, 41):
# - Sampling rate: 500 Hz (hardcoded)
# - nperseg: 256 (gives 129 frequency bins: 256//2 + 1)
# - noverlap: 131 (gives ~41 time frames from 5000 samples in 10 seconds)
# - This preserves frequencies up to ~250 Hz (vs. 128 Hz for downsampled MDD data)
CANE_STFT_NPERSEG = 256
CANE_STFT_NOVERLAP = 131  # (5000 - 256) / (256 - 131) + 1 ≈ 38.9 ≈ 39-40 frames


def load_and_preprocess_cane_raw_file(
        file_path: Path,
        channel: str = "d4",
        artifact_method: str = "interpolation",  # "interpolation", "clipping", or "both"
        artifact_threshold: float = 4.0,
        apply_car: bool = True,
        skip_extreme_artifacts: bool = False
) -> tuple[torch.Tensor, float]:
    """
    Load and preprocess CANE dataset CSV file at 500 Hz sampling rate.

    This function preprocesses CANE data at a hardcoded 500 Hz sampling rate,
    preserving frequency information up to the Nyquist limit (~250 Hz).
    The detected sampling rate is verified to be within tolerance of 500 Hz.

    Args:
        file_path: Path to CSV file containing CANE data
        channel: Which channel to use (d1-d8), default "d4"
        artifact_method: Method for artifact removal
        artifact_threshold: Z-score threshold for artifact detection
        apply_car: Whether to apply Common Average Reference
        skip_extreme_artifacts: If True, completely removes chunks with >10% artifacts

    Returns:
        tuple: (tensor, sampling_rate)
            - torch.Tensor: Shape (num_chunks, 1, chunk_samples) where chunk_samples
              is 5000 (10 seconds at 500 Hz)
            - float: The hardcoded sampling rate (500 Hz)
    """
    # Constants
    CHUNK_DURATION_SEC = 10.0  # 10 second chunks
    NEIGHBOUR_ZSCORE_TRESHOLD = 2.5

    # Load data
    df = pd.read_csv(file_path)

    # Calculate actual sampling rate and verify it's close to expected
    duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / 1000  # to seconds
    detected_fs = len(df) / duration
    logger.info(f"Detected sampling rate: {detected_fs:.2f} Hz")

    # Verify sampling rate is within tolerance
    if abs(detected_fs - CANE_SAMPLING_RATE) > CANE_SAMPLING_RATE_TOLERANCE:
        raise ValueError(
            f"Detected sampling rate {detected_fs:.2f} Hz is outside tolerance "
            f"({CANE_SAMPLING_RATE} ± {CANE_SAMPLING_RATE_TOLERANCE} Hz)"
        )

    # Use hardcoded sampling rate for consistency
    fs = CANE_SAMPLING_RATE
    chunk_samples = int(CHUNK_DURATION_SEC * fs)
    logger.info(f"Using hardcoded sampling rate: {fs} Hz")
    logger.info(f"Chunk size: {chunk_samples} samples ({CHUNK_DURATION_SEC}s)")

    # Step 1: Initial scaling (z-score normalization of raw ADC values)
    # This is critical for CANE data which has huge raw values
    eeg_channels = [f"d{i}" for i in range(1, 9)]
    for ch in eeg_channels:
        if ch in df.columns:
            df[ch] = zscore(df[ch])

    # Step 2: Common Average Reference (before filtering to remove common noise)
    if apply_car and all(ch in df.columns for ch in eeg_channels):
        car = df[eeg_channels].mean(axis=1)
        for ch in eeg_channels:
            df[ch] = df[ch] - car
        logger.info("Applied Common Average Reference")

    # Extract the specific channel
    signal = df[channel].values.astype(float)

    # Step 3: Detrend to remove slow drifts (before filtering)
    signal = detrend(signal, type='linear')

    # Step 4: Artifact handling (before filtering to avoid spreading artifacts)
    if artifact_method in ["interpolation", "both"]:
        z_scores = np.abs(zscore(signal))
        artifact_indices = np.where(z_scores > artifact_threshold)[0]

        if len(artifact_indices) > 0:
            artifact_pct = len(artifact_indices) / len(signal) * 100
            logger.info(f"Found {len(artifact_indices)} artifacts ({artifact_pct:.2f}%)")

            # Interpolate artifacts
            for idx in artifact_indices:
                # Find clean neighbors within ±100 samples
                window_start = max(0, idx - 100)
                window_end = min(len(signal), idx + 100)

                neighbors = signal[window_start:window_end]
                neighbor_mask = np.abs(zscore(neighbors)) < NEIGHBOUR_ZSCORE_TRESHOLD

                if neighbor_mask.sum() > 10:  # Need at least 10 clean samples
                    signal[idx] = np.median(neighbors[neighbor_mask])
                else:
                    # If no clean neighbors, use linear interpolation
                    if idx > 0 and idx < len(signal) - 1:
                        signal[idx] = (signal[idx - 1] + signal[idx + 1]) / 2

    if artifact_method in ["clipping", "both"]:
        # Percentile-based clipping as additional safety
        lower = np.percentile(signal, 0.5)
        upper = np.percentile(signal, 99.5)
        signal = np.clip(signal, lower, upper)

    # Step 5: Bandpass filter (matching MDD: 1-70 Hz)
    # Using 4th order Butterworth like in the context
    nyq = 0.5 * fs

    # High-pass at 1 Hz (removes DC and very slow drifts)
    b_hp, a_hp = butter(4, 1.0 / nyq, btype='high')
    signal = filtfilt(b_hp, a_hp, signal)

    # Low-pass at 70 Hz (removes high-frequency noise)
    b_lp, a_lp = butter(4, 70.0 / nyq, btype='low')
    signal = filtfilt(b_lp, a_lp, signal)

    # Step 6: Notch filter at 50 Hz (European powerline)
    b_notch, a_notch = iirnotch(50.0, Q=30, fs=fs)
    signal = filtfilt(b_notch, a_notch, signal)

    # Step 7: Chunk the signal at native sampling rate (NO resampling)
    n_samples = len(signal)
    chunks = []

    for i in range(0, n_samples - chunk_samples + 1, chunk_samples):
        chunk = signal[i:i + chunk_samples]

        # Optional: Skip chunks with too many residual artifacts
        if skip_extreme_artifacts:
            chunk_z = np.abs(zscore(chunk))
            if (chunk_z > 3).sum() / len(chunk) > 0.1:  # >10% artifacts
                logger.warning(f"Skipping chunk {len(chunks)} due to excessive artifacts")
                continue

        # Final z-score normalization per chunk (matching MDD)
        chunk = zscore(chunk)

        # Reshape to (1, samples) to match MDD format (channels, samples)
        chunk = chunk.reshape(1, -1)
        chunks.append(torch.from_numpy(chunk).float())

    # Stack into tensor (num_chunks, channels=1, samples)
    if chunks:
        chunks_tensor = torch.stack(chunks)
        logger.info(f"Created {len(chunks)} chunks of shape {chunks_tensor.shape}")
        return chunks_tensor, fs
    else:
        logger.error("No valid chunks created!")
        return torch.empty(0, 1, chunk_samples), fs


# Alternative: Process all channels at once
def load_and_preprocess_cane_all_channels(
        file_path: Path,
        channels: list[str] = None,
        **kwargs
) -> tuple[torch.Tensor, float]:
    """
    Process multiple channels and return best one or average.

    Args:
        file_path: Path to CSV file
        channels: List of channels to process (default: all d1-d8)
        **kwargs: Additional arguments for load_and_preprocess_cane_raw_file

    Returns:
        tuple: (tensor, sampling_rate)
            - torch.Tensor: Best channel or average of good channels
            - float: The detected sampling rate in Hz
    """
    if channels is None:
        channels = [f"d{i}" for i in range(1, 9)]

    all_chunks = []
    quality_scores = []
    fs = None

    for channel in channels:
        try:
            chunks, detected_fs = load_and_preprocess_cane_raw_file(
                file_path, channel=channel, **kwargs
            )

            if fs is None:
                fs = detected_fs

            if chunks.shape[0] > 0:
                # Calculate quality score (lower is better)
                # Based on: artifact percentage, high-frequency noise, signal variance
                signal_var = chunks.var().item()
                quality_score = abs(1.0 - signal_var)  # Variance should be ~1 after z-score

                all_chunks.append(chunks)
                quality_scores.append(quality_score)
                logger.info(f"Channel {channel} quality score: {quality_score:.3f}")
        except Exception as e:
            logger.warning(f"Failed to process channel {channel}: {e}")

    if not all_chunks:
        raise ValueError("No channels could be processed successfully")

    # Return best channel
    best_idx = np.argmin(quality_scores)
    logger.info(f"Selected channel {channels[best_idx]} as best quality")
    return all_chunks[best_idx], fs
