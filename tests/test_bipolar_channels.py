"""Tests for the bipolar surrogate ablation (Fp2-Fp1 / C4-C3 / T8-T7 channels).

Verifies:
- ``bipolar_pair()`` resolves the right electrode pairs (and ``in-ear``'s alias for T8-T7).
- The CLI accepts the three new ``--channel`` specs.
- ``T8-T7`` on MDD/SAD is bit-identical to ``in-ear`` (same electrodes, same "no CAR" path).
- A bipolar derivation equals the difference of its two electrodes preprocessed independently.
- CANE bipolar derivations produce sane, distinguishable output.

The equivalence/derivation/CANE tests need real recordings and are skipped when the
corresponding dataset directory is absent (e.g. in CI or a machine without the thesis data).
"""

import mne
import pytest
import torch
from scipy.signal import detrend
from scipy.stats import zscore

from thesis.cli import get_arg_parser
from thesis.dataset import (
    BIPOLAR_CHANNELS,
    CANE_DIR,
    MDD_CHANNEL_ORDER,
    MDD_DIR,
    SAD_CHANNEL_ORDER,
    SAD_DIR,
    CANEDataset,
    MDDDataset,
    SADDataset,
    bipolar_pair,
    load_and_preprocess_edf_file,
)
from thesis.stft import CHUNK_DURATION_SEC

mne.set_log_level("ERROR")


class TestBipolarPair:
    """``bipolar_pair()`` resolution."""

    @pytest.mark.parametrize(
        "channel,expected",
        [
            ("Fp2-Fp1", ("Fp2", "Fp1")),
            ("C4-C3", ("C4", "C3")),
            ("T8-T7", ("T8", "T7")),
            ("in-ear", ("T8", "T7")),
        ],
    )
    def test_resolves_bipolar_specs(self, channel, expected):
        assert bipolar_pair(channel) == expected

    @pytest.mark.parametrize("channel", ["all", "Fp1", None])
    def test_non_bipolar_specs_return_none(self, channel):
        assert bipolar_pair(channel) is None

    def test_whitelist_has_exactly_three_headset_montages(self):
        assert set(BIPOLAR_CHANNELS) == {"Fp2-Fp1", "C4-C3", "T8-T7"}


class TestCLIAcceptsBipolarChannels:
    """--channel parser accepts the new specs."""

    @pytest.mark.parametrize("channel", ["Fp2-Fp1", "C4-C3", "T8-T7"])
    def test_accepts_bipolar_channel(self, channel):
        parser = get_arg_parser()
        args = parser.parse_args(["train", "--model", "CNNLSTM", "--channel", channel])
        assert args.channel == channel

    def test_still_accepts_in_ear(self):
        parser = get_arg_parser()
        args = parser.parse_args(["train", "--model", "CNNLSTM", "--channel", "in-ear"])
        assert args.channel == "in-ear"


def _one_edf_file(data_dir, pattern):
    matches = sorted(data_dir.rglob(pattern))
    if not matches:
        pytest.skip(f"No files matching {pattern!r} found under {data_dir}")
    return matches[0]


def _preprocess_single_channel_reference(file_path, channel_mapping, canonical_name):
    """Independently preprocess one electrode: filter+notch on the full montage (matching
    ``load_and_preprocess_edf_file``), then narrow to a single channel and detrend.

    Filtering is applied identically whether or not other channels are also picked
    (each channel is filtered independently), so this exercises the same math as the
    bipolar path in ``load_and_preprocess_edf_file`` while indexing by name via a
    separate call, catching any positional-indexing bug.
    """
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw = raw.filter(l_freq=1, h_freq=70, method="iir", verbose=False)
    raw = raw.notch_filter(freqs=50, verbose=False)
    raw = raw.pick(list(channel_mapping.keys()))
    raw = raw.rename_channels(channel_mapping)
    if "A2-A1" in raw.ch_names:
        raw = raw.drop_channels(["A2-A1"])
    raw = raw.pick([canonical_name])
    signal = raw.get_data()[0]
    return detrend(signal, type="linear")


@pytest.mark.skipif(not MDD_DIR.exists(), reason="MDD dataset not available")
class TestMDDBipolarEquivalence:
    """T8-T7 must be bit-identical to in-ear on MDD, and match an independent reference."""

    def test_t8_t7_equals_in_ear(self):
        file_path = _one_edf_file(MDD_DIR, "*.edf")
        in_ear = load_and_preprocess_edf_file(
            file_path,
            channel="in-ear",
            fs=MDDDataset.FS,
            channel_mapping=MDDDataset.CHANNEL_MAPPING,
            channel_order=MDD_CHANNEL_ORDER,
        )
        t8_t7 = load_and_preprocess_edf_file(
            file_path,
            channel="T8-T7",
            fs=MDDDataset.FS,
            channel_mapping=MDDDataset.CHANNEL_MAPPING,
            channel_order=MDD_CHANNEL_ORDER,
        )
        assert torch.equal(in_ear, t8_t7)

    def test_c4_c3_matches_independent_reference(self):
        file_path = _one_edf_file(MDD_DIR, "*.edf")
        c4 = _preprocess_single_channel_reference(file_path, MDDDataset.CHANNEL_MAPPING, "C4")
        c3 = _preprocess_single_channel_reference(file_path, MDDDataset.CHANNEL_MAPPING, "C3")
        diff = c4 - c3

        chunk_samples = int(CHUNK_DURATION_SEC * MDDDataset.FS)
        expected_chunks = [
            zscore(diff[i : i + chunk_samples])
            for i in range(0, len(diff) - chunk_samples + 1, chunk_samples)
        ]
        expected = torch.stack(
            [torch.from_numpy(chunk).float().reshape(1, -1) for chunk in expected_chunks]
        )

        actual = load_and_preprocess_edf_file(
            file_path,
            channel="C4-C3",
            fs=MDDDataset.FS,
            channel_mapping=MDDDataset.CHANNEL_MAPPING,
            channel_order=MDD_CHANNEL_ORDER,
        )
        assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.skipif(not SAD_DIR.exists(), reason="SAD dataset not available")
class TestSADBipolarEquivalence:
    """T8-T7 must be bit-identical to in-ear on SAD too."""

    def test_t8_t7_equals_in_ear(self):
        file_path = _one_edf_file(SAD_DIR, "*.edf")
        in_ear = load_and_preprocess_edf_file(
            file_path,
            channel="in-ear",
            fs=SADDataset.FS,
            channel_mapping=SADDataset.CHANNEL_MAPPING,
            channel_order=SAD_CHANNEL_ORDER,
        )
        t8_t7 = load_and_preprocess_edf_file(
            file_path,
            channel="T8-T7",
            fs=SADDataset.FS,
            channel_mapping=SADDataset.CHANNEL_MAPPING,
            channel_order=SAD_CHANNEL_ORDER,
        )
        assert torch.equal(in_ear, t8_t7)


@pytest.mark.skipif(not CANE_DIR.exists(), reason="CANE dataset not available")
class TestCANEBipolarDerivations:
    """Bipolar derivations on CANE (headset source, not IDUN)."""

    def test_t8_t7_shape_and_finiteness(self):
        file_path = _one_edf_file(CANE_DIR, "*.csv")
        chunks = CANEDataset.load_and_preprocess_cane_raw_file(file_path, channel="T8-T7")
        assert chunks.shape[1:] == (1, 5000)
        assert torch.isfinite(chunks).all()

    def test_t8_t7_differs_from_fp2_fp1(self):
        file_path = _one_edf_file(CANE_DIR, "*.csv")
        t8_t7 = CANEDataset.load_and_preprocess_cane_raw_file(file_path, channel="T8-T7")
        fp2_fp1 = CANEDataset.load_and_preprocess_cane_raw_file(file_path, channel="Fp2-Fp1")
        n = min(len(t8_t7), len(fp2_fp1))
        assert not torch.equal(t8_t7[:n], fp2_fp1[:n])
