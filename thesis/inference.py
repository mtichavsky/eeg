"""
Shared inference pipeline for EEG classification.

Used by both the CLI (``main.py run``) and the API (``api/app.py /predict``)
to ensure identical preprocessing, inference, and aggregation logic.
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import mne
import torch
import torch.nn as nn

from thesis.dataset import (
    CANONICAL_CHANNEL_ORDER,
    MDD_CHANNEL_ORDER,
    SAD_CHANNEL_ORDER,
    CANEDataset,
    IDUNDataset,
    MDDDataset,
    SADDataset,
    SpectrogramDataset,
    load_and_preprocess_edf_file,
)
from thesis.labels import get_display_names
from thesis.model import RAW_EEG_MODELS
from thesis.stft import CHUNK_DURATION_SEC

logger = logging.getLogger(__name__)

# Known EDF channel configurations: (channel_mapping, channel_order)
# Tried in order; first whose keys are all present in the EDF wins.
_KNOWN_EDF_CONFIGS: list[tuple[dict[str, str], list[str]]] = [
    (MDDDataset.CHANNEL_MAPPING, MDD_CHANNEL_ORDER),
    (SADDataset.CHANNEL_MAPPING, SAD_CHANNEL_ORDER),
]


def _detect_edf_channel_config(file_path: Path) -> tuple[dict[str, str], list[str]]:
    """
    Detect the appropriate channel mapping and order for an EDF file.

    Tries each known configuration and returns the first whose mapping keys are all
    present in the file's channel list. Falls back to an identity mapping with
    ``CANONICAL_CHANNEL_ORDER`` if nothing matches.

    :param Path file_path: Path to the EDF file.
    :return: Tuple of (channel_mapping, channel_order).
    :rtype: tuple[dict[str, str], list[str]]
    """

    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
    edf_channels = set(raw.ch_names)

    for mapping, channel_order in _KNOWN_EDF_CONFIGS:
        if set(mapping.keys()).issubset(edf_channels):
            return mapping, channel_order

    logger.warning(
        "No known channel mapping matched %s (channels: %s). Using identity mapping.",
        file_path.name,
        sorted(edf_channels),
    )
    return {ch: ch for ch in edf_channels}, CANONICAL_CHANNEL_ORDER


def _detect_csv_format(file_path: Path) -> Literal["idun", "cane"]:
    """
    Detect CSV format by inspecting the header row.

    IDUN files have a ``ch1`` column (single in-ear channel).
    CANE files have EEG channel-named columns (``Fp1``, ``T7``, etc.).

    :param Path file_path: Path to the CSV file.
    :return: ``"idun"`` or ``"cane"``.
    :rtype: Literal["idun", "cane"]
    """
    with open(file_path, newline="") as f:
        header = f.readline()
    return "idun" if "ch1" in header else "cane"


def detect_sampling_rate(file_path: Path, file_format: str) -> float:
    """
    Auto-detect the sampling rate from an EEG file.

    - **EDF**: reads the EDF header via MNE.
    - **IDUN CSV** (``ch1``/``timestamp`` columns): returns :attr:`IDUNDataset.FS` (250 Hz).
    - **CANE CSV** (channel-named columns): returns :attr:`CANEDataset.FS` (500 Hz).

    :param Path file_path: Path to the EEG file.
    :param str file_format: File extension including dot (e.g. ``".edf"``, ``".csv"``).
    :return: Sampling rate in Hz.
    :rtype: float
    :raises ValueError: If the file format is unsupported.
    """
    if file_format == ".edf":
        import mne  # lazy import — only needed here

        raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)
        return float(round(cast(float, raw.info["sfreq"])))
    elif file_format == ".csv":
        fmt = _detect_csv_format(file_path)
        return round(IDUNDataset.FS if fmt == "idun" else CANEDataset.FS)
    else:
        raise ValueError(f"Cannot detect sampling rate from format: {file_format}")


@dataclass
class ChunkResult:
    """Result of inference on a single EEG chunk."""

    chunk_index: int
    predicted_class: int
    class_name: str


@dataclass
class InferenceResult:
    """Aggregated inference result over all chunks of a recording."""

    final_prediction: int
    final_class_name: str
    total_chunks: int
    class_distribution: dict[str, int]
    class_percentages: dict[str, float]
    chunk_results: list[ChunkResult] = field(default_factory=list)


def preprocess_file(
    file_path: Path,
    channel: str,
    file_format: str,
    sampling_rate: float | None = None,
    skip_artifact_removal: bool = True,
) -> torch.Tensor:
    """
    Preprocess an EEG file into chunks ready for spectrogram conversion.

    Automatically routes to the correct preprocessing pipeline:

    - ``.edf`` → :func:`load_and_preprocess_edf_file` (MDD / AX-MALIK)
    - ``.csv`` with ``ch1`` column → :meth:`IDUNDataset.load_and_preprocess_idun_file`
    - ``.csv`` with channel-named columns → :meth:`CANEDataset.load_and_preprocess_cane_raw_file`

    :param Path file_path: Path to the EEG file (.edf or .csv).
    :param str channel: Channel to use (e.g. ``"Fp1"``, ``"all"``, ``"in-ear"``).
    :param str file_format: File extension including dot (e.g. ``".edf"``, ``".csv"``).
    :param float | None sampling_rate: Sampling rate in Hz. Required for EDF; auto-detected
        from the CSV format if None.
    :param bool skip_artifact_removal: Skip artifact removal for CANE CSV files.
    :return: Tensor of shape ``(num_chunks, num_channels, samples)``.
    :rtype: torch.Tensor
    :raises ValueError: If the file format is unsupported.
    """
    if file_format == ".edf":
        fs = sampling_rate if sampling_rate is not None else detect_sampling_rate(file_path, ".edf")
        channel_mapping, channel_order = _detect_edf_channel_config(file_path)
        # Chunk at the file's own rate so a chunk is CHUNK_DURATION_SEC of wall-clock time.
        # Using a fixed sample count here would make a 256 Hz recording yield 9.77 s chunks,
        # which survive the STFT with one time frame too few instead of failing loudly.
        return load_and_preprocess_edf_file(
            file_path,
            channel=channel,
            fs=fs,
            channel_mapping=channel_mapping,
            channel_order=channel_order,
            chunk_samples=int(CHUNK_DURATION_SEC * fs),
        )
    elif file_format == ".csv":
        fmt = _detect_csv_format(file_path)
        if fmt == "idun":
            logger.info(f"Detected IDUN CSV format for {file_path.name}")
            return IDUNDataset.load_and_preprocess_idun_file(file_path)
        else:
            logger.info(f"Detected CANE CSV format for {file_path.name}")
            return CANEDataset.load_and_preprocess_cane_raw_file(
                file_path,
                channel=channel,
                skip_artifact_removal=skip_artifact_removal,
            )
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Supported: .edf, .csv")


def run_chunk_inference(
    model: nn.Module,
    inputs: list[torch.Tensor],
    device: torch.device,
    num_classes: int,
) -> list[ChunkResult]:
    """
    Run inference on a list of input tensors.

    Accepts both spectrogram tensors ``(num_channels, H, W)`` for CNN/LSTM models and raw EEG
    tensors ``(num_channels, T)`` for raw-EEG models. In both cases each tensor is unsqueezed to
    add a batch dimension before being passed to the model.

    :param nn.Module model: Trained model in eval mode.
    :param list[torch.Tensor] inputs: List of input tensors, each with shape
        ``(num_channels, H, W)`` (spectrogram) or ``(num_channels, T)`` (raw EEG).
    :param torch.device device: Device for inference.
    :param int num_classes: Number of output classes (2 or 4).
    :return: Per-chunk inference results.
    :rtype: list[ChunkResult]
    """
    class_names = get_display_names(num_classes)
    results: list[ChunkResult] = []

    model.eval()
    with torch.no_grad():
        for chunk_idx, inp in enumerate(inputs):
            batch = inp.unsqueeze(0).to(device)  # (1, C, H, W) or (1, C, T)
            logits = model(batch)
            pred = logits.argmax(dim=1).item()

            results.append(
                ChunkResult(
                    chunk_index=chunk_idx,
                    predicted_class=pred,
                    class_name=class_names[pred],
                )
            )

    return results


def aggregate_predictions(
    chunk_results: list[ChunkResult],
    num_classes: int,
) -> InferenceResult:
    """
    Aggregate per-chunk predictions via majority voting.

    :param list[ChunkResult] chunk_results: Per-chunk inference results.
    :param int num_classes: Number of output classes (2 or 4).
    :return: Aggregated inference result.
    :rtype: InferenceResult
    :raises ValueError: If chunk_results is empty.
    """
    if not chunk_results:
        raise ValueError("No chunk results to aggregate")

    class_names = get_display_names(num_classes)
    pred_list = [r.predicted_class for r in chunk_results]
    counts = Counter(pred_list)
    final_prediction = counts.most_common(1)[0][0]
    total = len(chunk_results)

    class_distribution = {name: counts.get(k, 0) for k, name in class_names.items()}
    class_percentages = {name: counts.get(k, 0) / total * 100 for k, name in class_names.items()}

    return InferenceResult(
        final_prediction=final_prediction,
        final_class_name=class_names[final_prediction],
        total_chunks=total,
        class_distribution=class_distribution,
        class_percentages=class_percentages,
        chunk_results=chunk_results,
    )


def preprocess_and_infer(
    file_path: Path,
    model: nn.Module,
    device: torch.device,
    channel: str,
    file_format: str,
    num_classes: int,
    sampling_rate: int | None = None,
    skip_artifact_removal: bool = True,
    model_name: str = "",
) -> InferenceResult:
    """
    High-level convenience: preprocess → (spectrograms or raw EEG) → inference → aggregation.

    Raw-EEG models (those in :data:`thesis.model.RAW_EEG_MODELS`) skip STFT conversion and
    receive the raw EEG chunks directly. All other models follow the spectrogram path.

    The sampling rate is inferred via :func:`detect_sampling_rate` when ``sampling_rate=None``.
    It is used only to resample the signal to :data:`thesis.stft.MODEL_FS`; the STFT parameters
    themselves are fixed in :mod:`thesis.stft` and shared with the training path, so a
    spectrogram produced here is identical to one the model saw during training.

    :param Path file_path: Path to the EEG file (.edf or .csv).
    :param nn.Module model: Trained model.
    :param torch.device device: Device for inference.
    :param str channel: Channel to use (e.g. "Fp1", "all", "in-ear").
    :param str file_format: File extension including dot (e.g. ".edf", ".csv").
    :param int num_classes: Number of output classes (2 or 4).
    :param float | None sampling_rate: Sampling rate in Hz. Auto-detected from the file if None.
    :param bool skip_artifact_removal: Skip artifact removal for CANE CSV files.
    :param str model_name: Model architecture name (e.g. ``"Deformer"``). Used to determine
        whether to skip STFT. Defaults to ``""`` (spectrogram path).
    :return: Aggregated inference result.
    :rtype: InferenceResult
    """
    fs = (
        sampling_rate if sampling_rate is not None else detect_sampling_rate(file_path, file_format)
    )
    logger.info(f"Using sampling rate: {fs} Hz")

    # Preprocess file into EEG chunks: (num_chunks, num_channels, samples)
    chunks = preprocess_file(file_path, channel, file_format, fs, skip_artifact_removal)
    logger.info(f"Extracted {len(chunks)} chunks from {file_path.name}")

    if len(chunks) == 0:
        raise RuntimeError(f"No valid chunks extracted from {file_path}")

    if model_name in RAW_EEG_MODELS:
        # Raw-EEG models consume (batch, channels, time) directly — skip STFT
        logger.info(f"Raw-EEG model '{model_name}': skipping spectrogram conversion")
        inputs: list[torch.Tensor] = [chunks[i] for i in range(len(chunks))]
    else:
        # Spectrogram models: convert chunks to log-magnitude STFT tensors. Resampling to
        # MODEL_FS and the STFT parameters themselves are handled inside, from thesis.stft.
        spectrograms = SpectrogramDataset.convert_to_spectrograms(chunks, source_fs=fs)

        if not spectrograms:
            raise RuntimeError("No valid spectrograms generated")

        logger.info(f"Generated {len(spectrograms)} spectrograms")
        inputs = spectrograms

    # Run inference
    chunk_results = run_chunk_inference(model, inputs, device, num_classes)
    logger.info(f"Inference complete: {len(chunk_results)} predictions")

    # Aggregate
    return aggregate_predictions(chunk_results, num_classes)
