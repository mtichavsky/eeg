"""Tests for the bipolar surrogate ablation (Fp2-Fp1 / C4-C3 / T8-T7 channels).

Verifies:
- ``bipolar_pair()`` resolves the right electrode pairs (and ``in-ear``'s alias for T8-T7).
- The CLI accepts the three new ``--channel`` specs.
- ``T8-T7`` on MDD/SAD is bit-identical to ``in-ear`` (same electrodes, same "no CAR" path).
- A bipolar derivation equals the difference of its two electrodes preprocessed independently.
- CANE bipolar derivations produce sane, distinguishable output.
- CANE bipolar derivations subtract raw electrode voltages (no initial per-channel z-score,
  no CAR) so common-mode content cancels exactly even when the two electrodes have very
  different noise scales, and are unaffected by artifacts on unrelated channels -- using
  synthetic CANE-format CSVs, no real dataset required.

The equivalence/derivation/CANE tests need real recordings and are skipped when the
corresponding dataset directory is absent (e.g. in CI or a machine without the thesis data).
The synthetic-CSV cancellation tests need no dataset and always run.
"""

import mne
import numpy as np
import pandas as pd
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


def _write_cane_csv(tmp_path, filename: str, channels: dict):
    """Write a synthetic CANE-format CSV.

    :param tmp_path: pytest ``tmp_path`` fixture directory to write into.
    :param str filename: Name of the CSV file to create.
    :param dict channels: Maps raw CANE column names (``"d1"``..``"d8"``) to sample arrays.
           A ``"timestamp"`` column (milliseconds, exactly ``CANEDataset.FS`` Hz spacing) is
           added automatically, satisfying ``CANEDataset.verify_sampling_rate``.
    :return: Path to the written CSV file.
    """
    n_samples = len(next(iter(channels.values())))
    t_ms = np.arange(n_samples) * (1000.0 / CANEDataset.FS)
    data = {"timestamp": t_ms}
    data.update(channels)
    path = tmp_path / filename
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _base_cane_channels(n_samples: int, seed: int = 0) -> dict:
    """Baseline arrays (small independent Gaussian noise) for all 8 raw CANE channels.

    Tests overwrite specific channels (e.g. ``"d3"``/``"d7"`` for T7/T8) to build scenarios.

    :param int n_samples: Number of samples per channel.
    :param int seed: Seed for the noise RNG, for reproducibility.
    :return: Dict mapping ``"d1"``..``"d8"`` to noise arrays.
    """
    rng = np.random.RandomState(seed)
    return {f"d{i}": rng.normal(0, 1.0, n_samples) for i in range(1, 9)}


class TestCANEBipolarCancellation:
    """CANE bipolar derivations must subtract raw electrode voltages, not per-channel
    z-scored/CAR'd ones. Otherwise common-mode content does not cancel when the two
    electrodes have different noise scales, and unrelated channels can leak in via CAR.
    Uses synthetic CANE-format CSVs (``tmp_path``), so no real dataset is required.
    """

    N = 1000  # 2 s at CANEDataset.FS (500 Hz); short enough to keep the test fast.

    def _t7_t8_with_sigma_mismatch(self):
        """Common 10 Hz component shared by T7/T8 at equal raw amplitude, but with very
        different independent noise scales (std 1 vs std 100) -- so per-channel z-scoring
        (dividing by each electrode's own total sigma) would rescale the shared component
        very differently on each electrode before any subtraction.
        """
        t = np.arange(self.N)
        common = 50.0 * np.sin(2 * np.pi * 10 * t / CANEDataset.FS)
        t7_noise = np.random.RandomState(1).normal(0, 1.0, self.N)
        t8_noise = np.random.RandomState(2).normal(0, 100.0, self.N)
        return common + t7_noise, common + t8_noise

    def test_common_mode_cancels_exactly_despite_sigma_mismatch(self, tmp_path):
        t7, t8 = self._t7_t8_with_sigma_mismatch()

        channels = _base_cane_channels(self.N, seed=0)
        channels["d3"] = t7  # T7
        channels["d7"] = t8  # T8
        file_path = _write_cane_csv(tmp_path, "sigma_mismatch.csv", channels)
        bipolar = CANEDataset.load_and_preprocess_cane_raw_file(
            file_path, channel="T8-T7", chunk_samples=self.N
        )

        # A second file storing the raw difference directly as "T8" with "T7" zeroed. Since
        # the fixed pipeline subtracts raw voltages (T8 - T7) before any per-channel
        # rescaling or filtering, this is algebraically the same input signal as on the
        # first file: (t8 - t7) - 0 == t8 - t7. The two must therefore produce the same
        # output -- proof the common-mode content cancels exactly, independent of the
        # sigma mismatch.
        diff_channels = _base_cane_channels(self.N, seed=0)
        diff_channels["d3"] = np.zeros(self.N)
        diff_channels["d7"] = t8 - t7
        diff_file_path = _write_cane_csv(tmp_path, "diff_only.csv", diff_channels)
        bipolar_from_diff = CANEDataset.load_and_preprocess_cane_raw_file(
            diff_file_path, channel="T8-T7", chunk_samples=self.N
        )

        assert torch.allclose(bipolar, bipolar_from_diff, atol=1e-6)

    def test_old_per_channel_zscore_and_car_would_have_broken_cancellation(self, tmp_path):
        """Regression guard: reimplements the pre-fix pipeline (z-score every channel, then
        CAR, then process T7/T8 individually, then subtract) to show it does NOT reproduce
        the raw-difference reference that the fixed pipeline matches exactly above --
        confirming the old code was genuinely broken for this case, not just differently
        written.
        """
        t7, t8 = self._t7_t8_with_sigma_mismatch()
        channels = _base_cane_channels(self.N, seed=0)
        channels["d3"] = t7
        channels["d7"] = t8
        df = pd.DataFrame({"timestamp": np.arange(self.N) * (1000.0 / CANEDataset.FS), **channels})

        # Old (buggy) steps: per-channel z-score, then CAR, on raw ADC columns.
        for ch in CANEDataset.CHANNELS:
            df[ch] = zscore(df[ch])
        car = df[CANEDataset.CHANNELS].mean(axis=1)
        for ch in CANEDataset.CHANNELS:
            df[ch] = df[ch] - car
        df.rename(columns=CANEDataset.CHANNEL_MAPPING, inplace=True)

        dummy_path = tmp_path / "dummy.csv"
        t7_old = CANEDataset._process_single_channel(
            df["T7"].values.astype(float), "T7", dummy_path, "interpolation", 4.0, False
        )
        t8_old = CANEDataset._process_single_channel(
            df["T8"].values.astype(float), "T8", dummy_path, "interpolation", 4.0, False
        )
        old_bipolar = zscore(t8_old - t7_old)

        # Reference: raw difference run through the single-channel pipeline once (what the
        # fixed code does), then z-scored the same way the final chunk normalization does.
        reference = zscore(
            CANEDataset._process_single_channel(
                t8 - t7, "T8-T7", dummy_path, "interpolation", 4.0, False
            )
        )

        assert not np.allclose(old_bipolar, reference, atol=0.5)

    def test_unaffected_by_artifact_on_unrelated_channel(self, tmp_path):
        """Fp1 gets a large artifact in one file and none in the other; T8-T7 must be
        identical between the two since the fixed pipeline never touches Fp1 (no CAR, no
        shared z-score) when computing a bipolar derivation.
        """
        t7, t8 = self._t7_t8_with_sigma_mismatch()

        clean_channels = _base_cane_channels(self.N, seed=0)
        clean_channels["d3"] = t7
        clean_channels["d7"] = t8
        clean_path = _write_cane_csv(tmp_path, "clean_fp1.csv", clean_channels)

        artifact_channels = _base_cane_channels(self.N, seed=0)
        artifact_channels["d3"] = t7
        artifact_channels["d7"] = t8
        artifact_channels["d1"] = artifact_channels["d1"].copy()
        artifact_channels["d1"][self.N // 2] += 5000.0  # large spike on Fp1 only
        artifact_path = _write_cane_csv(tmp_path, "artifact_fp1.csv", artifact_channels)

        clean_bipolar = CANEDataset.load_and_preprocess_cane_raw_file(
            clean_path, channel="T8-T7", chunk_samples=self.N
        )
        artifact_bipolar = CANEDataset.load_and_preprocess_cane_raw_file(
            artifact_path, channel="T8-T7", chunk_samples=self.N
        )

        assert torch.equal(clean_bipolar, artifact_bipolar)

    def test_non_bipolar_paths_unaffected(self, tmp_path):
        """Sanity/regression check: 'all' and single-channel CANE paths are untouched by the
        bipolar fix (shape + finiteness only -- their code path did not change).
        """
        channels = _base_cane_channels(self.N, seed=0)
        file_path = _write_cane_csv(tmp_path, "non_bipolar.csv", channels)

        single = CANEDataset.load_and_preprocess_cane_raw_file(
            file_path, channel="Fp1", chunk_samples=self.N
        )
        assert single.shape[1:] == (1, self.N)
        assert torch.isfinite(single).all()

        all_channels = CANEDataset.load_and_preprocess_cane_raw_file(
            file_path, channel="all", chunk_samples=self.N
        )
        assert all_channels.shape[1:] == (8, self.N)
        assert torch.isfinite(all_channels).all()
