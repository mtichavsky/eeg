"""
Tests for the explainability harness, band definitions and statistics.

Everything here runs on synthetic data, so the suite is usable on a machine with neither the
EEG datasets nor a GPU. The SHAP-specific tests live in ``tests/test_shap.py``.
"""

from pathlib import Path

import pytest

from thesis.explain import harness
from thesis.explain.groups import EEG_BANDS, band_bins
from thesis.explain.stats import (
    benjamini_hochberg,
    nadeau_bengio_corrected_t,
    paired_wilcoxon,
    spearman,
)
from thesis.stft import FREQ_BIN_WIDTH_HZ, NUM_FREQ_BINS


class TestBands:
    """Frequency-band to bin-index mapping."""

    def test_bands_are_disjoint_and_ordered(self) -> None:
        seen: set[int] = set()
        previous_max = -1
        for band in EEG_BANDS:
            bins = band_bins(band)
            assert bins, f"{band} is empty"
            assert not seen & set(bins), f"{band} overlaps a lower band"
            assert min(bins) > previous_max
            seen |= set(bins)
            previous_max = max(bins)

    def test_bins_lie_inside_their_band(self) -> None:
        for band, (low, high) in EEG_BANDS.items():
            for index in band_bins(band):
                assert low <= index * FREQ_BIN_WIDTH_HZ < high

    def test_alpha_covers_the_expected_frequencies(self) -> None:
        # Alpha is the band the depression literature cares about; pin it down explicitly.
        assert band_bins("alpha") == [9, 10, 11, 12, 13]

    def test_no_bin_exceeds_the_spectrogram(self) -> None:
        for band in EEG_BANDS:
            assert max(band_bins(band)) < NUM_FREQ_BINS


class TestStats:
    """Paired statistics."""

    HIGH = [0.80, 0.78, 0.82, 0.79, 0.81, 0.77, 0.83, 0.80, 0.79, 0.81]
    LOW = [0.72, 0.70, 0.75, 0.71, 0.73, 0.69, 0.76, 0.72, 0.71, 0.74]

    def test_wilcoxon_detects_a_consistent_difference(self) -> None:
        result = paired_wilcoxon(self.HIGH, self.LOW)
        assert result["mean_difference"] > 0
        assert result["p_value"] < 0.05
        assert result["n"] == 10

    def test_identical_sequences_give_p_one(self) -> None:
        result = paired_wilcoxon(self.HIGH, self.HIGH)
        assert result["p_value"] == 1.0
        assert result["mean_difference"] == 0.0

    def test_difference_sign_convention(self) -> None:
        # mean_difference is a - b.
        assert paired_wilcoxon(self.LOW, self.HIGH)["mean_difference"] < 0
        nb = nadeau_bengio_corrected_t(self.LOW, self.HIGH, 0.1)
        assert nb["mean_difference"] < 0

    def test_nadeau_bengio_is_more_conservative_than_naive(self) -> None:
        from scipy import stats as scipy_stats

        corrected = nadeau_bengio_corrected_t(self.LOW, self.HIGH, 0.1)
        naive = scipy_stats.ttest_rel(self.HIGH, self.LOW)
        assert corrected["p_value"] > float(naive.pvalue)
        assert abs(corrected["t_statistic"]) < abs(float(naive.statistic))

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1])
    def test_nadeau_bengio_rejects_bad_fraction(self, fraction: float) -> None:
        with pytest.raises(ValueError):
            nadeau_bengio_corrected_t(self.LOW, self.HIGH, fraction)

    def test_benjamini_hochberg_is_monotone_and_conservative(self) -> None:
        raw = [0.001, 0.01, 0.04, 0.2, 0.6]
        result = benjamini_hochberg(raw)

        assert all(q >= p for q, p in zip(result.q_values, raw))
        ordered = [q for _, q in sorted(zip(raw, result.q_values))]
        assert ordered == sorted(ordered)
        assert result.n_rejected == sum(result.rejected)

    def test_benjamini_hochberg_preserves_input_order(self) -> None:
        result = benjamini_hochberg([0.6, 0.001, 0.04])
        assert result.q_values[1] < result.q_values[0]

    def test_spearman_ranks(self) -> None:
        assert spearman([1, 2, 3, 4], [1, 2, 3, 4])["rho"] == pytest.approx(1.0)
        assert spearman([1, 2, 3, 4], [4, 3, 2, 1])["rho"] == pytest.approx(-1.0)

    @pytest.mark.parametrize("func", [paired_wilcoxon, lambda a, b: spearman(a, b)])
    def test_length_mismatch_rejected(self, func) -> None:
        with pytest.raises(ValueError):
            func([1.0, 2.0, 3.0], [1.0, 2.0])


class TestRunConfig:
    """Recovering a run's configuration from results.txt."""

    CONFIG = """10-Fold Cross-Validation Results
========================================

Configuration:
  command: train
  condition: ec+eo
  channel: all
  class_mode: 2
  dataset: all
  model: AllTransformerV4
  n_folds: 10
  batch_size: 64
  dropout: 0.1
  skip_artifact_removal: False
  chunk_duration: 10.0

Fold Results:
  fold 1: 0.77
"""

    def test_parses_the_configuration_block(self, tmp_path: Path) -> None:
        (tmp_path / "results.txt").write_text(self.CONFIG)
        config = harness.read_run_config(tmp_path)

        assert config.model == "AllTransformerV4"
        assert config.channel == "all"
        assert config.num_classes == 2
        assert config.conditions == ["EC", "EO"]
        assert config.n_folds == 10
        assert config.batch_size == 64
        assert config.dropout == pytest.approx(0.1)
        assert config.skip_artifact_removal is False

    def test_stops_at_the_end_of_the_block(self, tmp_path: Path) -> None:
        (tmp_path / "results.txt").write_text(self.CONFIG)
        assert "fold" not in harness.read_run_config(tmp_path).raw

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            harness.read_run_config(tmp_path)

    def test_incomplete_block_raises(self, tmp_path: Path) -> None:
        (tmp_path / "results.txt").write_text("Configuration:\n  model: X\n")
        with pytest.raises(ValueError, match="missing configuration keys"):
            harness.read_run_config(tmp_path)

    def test_freq_cutoff_defaults_to_70_hz(self, tmp_path: Path) -> None:
        (tmp_path / "results.txt").write_text(self.CONFIG)
        config = harness.read_run_config(tmp_path)
        assert config.freq_cutoff_hz == 70.0
        assert config.spec_shape == (72, 41)
        assert config.in_channels == 8

    def test_freq_cutoff_sets_the_spectrogram_shape(self, tmp_path: Path) -> None:
        text = self.CONFIG.replace("  channel: all\n", "  channel: in-ear\n  freq_cutoff: 30.0\n")
        (tmp_path / "results.txt").write_text(text)
        config = harness.read_run_config(tmp_path)
        assert config.freq_cutoff_hz == 30.0
        assert config.spec_shape == (31, 41)
        assert config.in_channels == 1
