"""Tests for the frequency-cutoff ablation (``--freq-cutoff``).

Verifies:
- ``num_freq_bins()`` / ``spectrogram_shape()`` match the row count the cutoff mask actually
  produces, for several cutoffs and source sampling rates.
- The 70 Hz default is unchanged from before this ablation was added.
- Crop consistency: a lower cutoff is a strict prefix of the 70 Hz spectrogram's rows, so a
  30 Hz run differs from baseline only by the removed rows, never by different values in the
  rows that remain.
- Spectrogram models build and run at the reduced (31, 41) shape, and at the geometric minimum
  (22, 41); one row below that (21, 41) fails to build (the CNN would need a negative dimension).
- ``main._validate_freq_cutoff`` enforces the ``[21, 70]`` range the CNN can build for.
- A ``SpectrogramDataset`` with a non-default ``freq_cutoff_hz`` survives pickling (needed
  because ``forkserver`` DataLoader workers re-import modules, so a module-level override
  would not reach them -- the cutoff must live on the picklable instance).

All synthetic; no dataset files needed.
"""

import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from main import _validate_freq_cutoff, train_one_fold
from thesis.dataset import MDDDataset, SpectrogramDataset
from thesis.model_factory import create_model
from thesis.stft import (
    EXPECTED_SPECTROGRAM_SHAPE,
    compute_log_spectrogram,
    num_freq_bins,
    spectrogram_shape,
)


@pytest.mark.parametrize("freq_cutoff_hz", [21, 30, 45.5, 70])
@pytest.mark.parametrize("source_fs", [250, 256, 500])
def test_bin_count_matches_mask(freq_cutoff_hz, source_fs):
    """The row count `num_freq_bins`/`spectrogram_shape` predict matches the actual mask."""
    signal = np.random.randn(10 * source_fs)
    spec = compute_log_spectrogram(signal, source_fs, freq_cutoff_hz)
    assert spec.shape == spectrogram_shape(freq_cutoff_hz)


def test_default_cutoff_unchanged():
    """The 70 Hz default geometry is untouched by adding the freq_cutoff_hz parameter."""
    assert spectrogram_shape() == (72, 41)
    assert EXPECTED_SPECTROGRAM_SHAPE == (72, 41)
    assert num_freq_bins() == 72

    signal = np.random.randn(2560)
    default_call = compute_log_spectrogram(signal, 256)
    explicit_call = compute_log_spectrogram(signal, 256, 70.0)
    assert np.array_equal(default_call, explicit_call)


def test_crop_consistency_is_a_prefix():
    """A 30 Hz spectrogram equals the first 31 rows of the 70 Hz spectrogram of the same signal.

    This is the key invariant: a lower-cutoff run differs from the 70 Hz baseline only by the
    rows removed, never by different values in the rows that remain.
    """
    signal = np.random.randn(2560)
    cropped = compute_log_spectrogram(signal, 256, 30)
    full = compute_log_spectrogram(signal, 256, 70)
    assert np.array_equal(cropped, full[:31])


def test_models_build_and_run_at_reduced_shape():
    """Spectrogram models build and produce correct-shaped output at the 30 Hz cutoff geometry."""
    device = torch.device("cpu")

    single_channel_model = create_model(
        model_name="CNNAttnS",
        spec_shape=(31, 41),
        dropout=0.0,
        num_classes=2,
        device=device,
        in_channels=1,
    )
    out = single_channel_model(torch.zeros(2, 1, 31, 41))
    assert out.shape == (2, 2)

    multi_channel_model = create_model(
        model_name="AllTransformerV4",
        spec_shape=(31, 41),
        dropout=0.0,
        num_classes=2,
        device=device,
        in_channels=8,
    )
    out = multi_channel_model(torch.zeros(2, 8, 31, 41))
    assert out.shape == (2, 2)


def test_geometric_minimum_shape_builds():
    """(22, 41) -- the minimum height the CNN can build layers for -- builds successfully."""
    device = torch.device("cpu")
    model = create_model(
        model_name="CNNAttnS",
        spec_shape=(22, 41),
        dropout=0.0,
        num_classes=2,
        device=device,
        in_channels=1,
    )
    out = model(torch.zeros(2, 1, 22, 41))
    assert out.shape == (2, 2)


def test_below_geometric_minimum_shape_raises():
    """(21, 41) is one row too few: the CNN would need a negative dimension and fails to build."""
    device = torch.device("cpu")
    with pytest.raises(RuntimeError):
        create_model(
            model_name="CNNAttnS",
            spec_shape=(21, 41),
            dropout=0.0,
            num_classes=2,
            device=device,
            in_channels=1,
        )


@pytest.mark.parametrize("freq_cutoff_hz", [20.0, 80.0])
def test_validate_freq_cutoff_rejects_out_of_range(freq_cutoff_hz):
    with pytest.raises(ValueError):
        _validate_freq_cutoff(freq_cutoff_hz)


@pytest.mark.parametrize("freq_cutoff_hz", [21.0, 30.0, 70.0])
def test_validate_freq_cutoff_accepts_in_range(freq_cutoff_hz):
    _validate_freq_cutoff(freq_cutoff_hz)  # must not raise


def test_spectrogram_dataset_with_freq_cutoff_survives_pickle(tmp_path):
    """A SpectrogramDataset with a non-default freq_cutoff_hz pickles with the value intact."""
    inner = MDDDataset(data_dir=tmp_path, cache_size=8, channel="all")
    spec = SpectrogramDataset(inner, source_fs=MDDDataset.FS, cache_size=8, freq_cutoff_hz=30.0)

    restored = pickle.loads(pickle.dumps(spec))

    assert restored.freq_cutoff_hz == 30.0
    assert callable(restored._get_item_cached)


class _TinyBinaryDataset(Dataset):
    """Trivially separable 2-class dataset: x=+1 -> label 0, x=-1 -> label 1."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        x = torch.tensor([1.0]) if idx % 2 == 0 else torch.tensor([-1.0])
        return x, idx % 2, f"subj{idx}"


def test_train_one_fold_omits_freq_cutoff_for_raw_eeg_models(tmp_path):
    """``train_cross_validation`` passes ``freq_cutoff_hz=None`` for RAW_EEG_MODELS.

    The cutoff only ever affects spectrogram geometry, so a raw-EEG checkpoint must not carry
    a ``freq_cutoff_hz`` key at all -- recording one would wrongly suggest the flag did
    something for a model that never builds a spectrogram.
    """
    model = nn.Linear(1, 2)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        model.bias.zero_()

    train_loader = DataLoader(_TinyBinaryDataset(), batch_size=2)
    val_loader = DataLoader(_TinyBinaryDataset(), batch_size=2)

    train_one_fold(
        fold=0,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        device=torch.device("cpu"),
        num_classes=2,
        num_epochs=1,
        save_every=1000,
        val_every=1,
        patience=1000,
        checkpoint_dir=tmp_path,
        freq_cutoff_hz=None,
    )

    checkpoint_path = tmp_path / "fold_1_best.pth"
    assert checkpoint_path.is_file(), "expected a best-model checkpoint to be saved"
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert "freq_cutoff_hz" not in checkpoint
