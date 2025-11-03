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
        "EEG Fp1-LE": "Fp1",
        "EEG Fp2-LE": "Fp2",
        "EEG C3-LE": "C3",
        "EEG C4-LE": "C4",
        "EEG O2-LE": "O2",
        "EEG Cz-LE": "Cz",
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

        raw = raw.pick(
            ["EEG Fp1-LE", "EEG Fp2-LE", "EEG C3-LE", "EEG C4-LE", "EEG O2-LE", "EEG Cz-LE"]
        )
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


def collate_spectrograms(
    batch: list[tuple[list[torch.Tensor], int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Custom collate function to handle variable-length lists of spectrograms from SpectrogramDataset.

    Each item in the batch is a tuple of (list of spectrograms, label, subject).
    This function flattens the lists and creates proper batches of individual spectrograms.

    :param list batch: List of tuples (list[spectrograms], label, subject) from
           SpectrogramDataset.__getitem__()
    :return: Tuple of (stacked spectrograms, repeated labels, repeated subjects).
    :rtype: tuple[torch.Tensor, torch.Tensor, list[str]]
    """
    all_spectrograms = []
    all_labels = []
    all_subjects = []

    for spectrograms, label, subject in batch:
        # Extend lists with all spectrograms from this file
        all_spectrograms.extend(spectrograms)
        # Repeat label and subject for each spectrogram
        all_labels.extend([label] * len(spectrograms))
        all_subjects.extend([subject] * len(spectrograms))

    # Stack all spectrograms into a single tensor
    spectrograms_tensor = torch.stack(all_spectrograms)  # Shape: (total_spectrograms, 1, H, W)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    return spectrograms_tensor, labels_tensor, all_subjects
