"""Tests for ``helpers-print/single_dataset_summary.py`` using synthetic checkpoints."""

import logging
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import wilcoxon

from tests.helper_scripts import load_helper_script, write_fold_checkpoint, write_run

summary = load_helper_script("single_dataset_summary")
bipolar = load_helper_script("bipolar_ablation_summary")


def _fold(dataset: str, tp: int, tn: int, fp: int, fn: int) -> dict[str, tuple[int, int, int, int]]:
    return {dataset: (tp, tn, fp, fn)}


def _cane_folds() -> list[dict[str, tuple[int, int, int, int]]]:
    """Ten folds of a CANE-like run: ~70 % pathological, model right about at the base rate."""
    rng = np.random.default_rng(0)
    folds = []
    for _ in range(10):
        pos, neg = 70 + int(rng.integers(-5, 6)), 30 + int(rng.integers(-5, 6))
        tp = int(pos * rng.uniform(0.85, 0.95))
        tn = int(neg * rng.uniform(0.2, 0.4))
        folds.append(_fold("cane", tp, tn, neg - tn, pos - tp))
    return folds


class TestMajorityRate:
    def test_pathological_majority(self) -> None:
        # TN=20, FP=10, FN=5, TP=65
        assert summary.Counts(tp=65, tn=20, fp=10, fn=5).baseline == pytest.approx(0.70)

    def test_healthy_majority(self) -> None:
        # TN=60, FP=10, FN=5, TP=25
        assert summary.Counts(tp=25, tn=60, fp=10, fn=5).baseline == pytest.approx(0.70)


class TestLoadSingleRun:
    def test_values_match_checkpoint(self, tmp_path: Path) -> None:
        folds = [_fold("sad", 30, 20, 10, 40), _fold("sad", 25, 35, 5, 35)]
        write_run(tmp_path / "sad_041_t8-t7_ec", folds, subject_accs=[0.6, 0.8])
        run = summary.load_single_run(tmp_path / "sad_041_t8-t7_ec", "sad", "ec")
        assert run.n_folds == 2
        assert run.chunk_acc == pytest.approx([50 / 100, 60 / 100])
        assert run.chunk_sens == pytest.approx([30 / 70, 25 / 60])
        assert run.chunk_spec == pytest.approx([20 / 30, 35 / 40])
        assert run.subject_acc == pytest.approx([0.6, 0.8])
        assert run.majority == pytest.approx([70 / 100, 60 / 100])
        pooled = run.pooled
        assert (pooled.tp, pooled.tn, pooled.fp, pooled.fn) == (55, 55, 15, 75)

    def test_multiple_datasets_in_per_dataset_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        fold = {"cane": (10, 5, 3, 2), "sad": (4, 4, 1, 1)}
        write_fold_checkpoint(tmp_path / "run" / "fold_1_best.pth", fold)
        with caplog.at_level(logging.WARNING):
            summary.load_single_run(tmp_path / "run", "cane", "ec")
        assert "expected only 'cane'" in caplog.text

    def test_no_checkpoints_raises(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError):
            summary.load_single_run(tmp_path / "empty", "mdd", "ec")

    def test_run_name_follows_convention(self) -> None:
        assert summary.run_name("cane", "eo") == "cane_041_t8-t7_eo"
        assert summary.run_name("all", "ec", "040") == "all_040_t8-t7_ec"


class TestPairedStatistics:
    def test_shares_helpers_with_bipolar_summary(self) -> None:
        assert summary.nadeau_bengio_ttest is bipolar.nadeau_bengio_ttest
        assert summary.benjamini_hochberg is bipolar.benjamini_hochberg

    def test_matches_direct_computation(self, tmp_path: Path) -> None:
        write_run(tmp_path / "cane_041_t8-t7_ec", _cane_folds())
        run = summary.load_single_run(tmp_path / "cane_041_t8-t7_ec", "cane", "ec")
        result = summary.paired_vs_majority(run)

        diffs = np.asarray(run.chunk_acc) - np.asarray(run.majority)
        expected_t, expected_p = bipolar.nadeau_bengio_ttest(diffs, n_train=9, n_test=1)
        assert result.mean_diff_pp == pytest.approx(diffs.mean() * 100)
        assert result.wilcoxon_p == pytest.approx(wilcoxon(diffs)[1])
        assert result.nb_t == pytest.approx(expected_t)
        assert result.nb_p == pytest.approx(expected_p)
        assert result.n_above == int((diffs > 0).sum())

    def test_clearly_above_majority_is_significant(self, tmp_path: Path) -> None:
        # ~90 % accuracy on a 60 % pathological dataset, every fold: a positive control.
        folds = []
        for shift in range(10):
            folds.append(_fold("mdd", 52 + shift % 3, 36, 4, 8 - shift % 3))
        write_run(tmp_path / "mdd_041_t8-t7_ec", folds)
        run = summary.load_single_run(tmp_path / "mdd_041_t8-t7_ec", "mdd", "ec")
        result = summary.paired_vs_majority(run)
        assert result.mean_diff_pp > 20
        assert result.n_above == 10
        assert result.nb_p < 0.01

    def test_majority_predictor_does_not_crash(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Always predicts pathological on a 70 % pathological dataset: diff is exactly 0 per fold.
        write_run(tmp_path / "run", [_fold("cane", 70, 0, 30, 0)] * 10)
        run = summary.load_single_run(tmp_path / "run", "cane", "ec")
        with caplog.at_level(logging.WARNING), np.errstate(all="ignore"):
            result = summary.paired_vs_majority(run)
        # Depending on the scipy version an all-zero difference gives p = 1 or raises (-> NaN).
        assert np.isnan(result.wilcoxon_p) or result.wilcoxon_p == pytest.approx(1.0)
        assert result.mean_diff_pp == pytest.approx(0.0)
        assert result.nb_p == pytest.approx(1.0)


class TestCli:
    def test_end_to_end_with_combined_comparison(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        write_run(tmp_path / "cane_041_t8-t7_ec", _cane_folds())
        write_run(
            tmp_path / "sad_041_t8-t7_ec",
            [_fold("sad", 20 + i % 3, 20, 10, 10 - i % 3) for i in range(10)],
        )
        # Combined all_040 run with the pre-fix (swapped) per-dataset orientation.
        combined = [
            {"mdd": (50, 40, 5, 5), "cane": (60, 5, 25, 10), "sad": (10, 20, 8, 12)}
            for _ in range(10)
        ]
        write_run(tmp_path / "all_040_t8-t7_ec", combined, swapped=True)

        with caplog.at_level(logging.INFO):
            code = summary.main(["--root", str(tmp_path), "--per-fold"], configure_logging=False)
        text = "\n".join(caplog.messages)
        assert code == 0
        assert "[skip]" in text  # the eo and mdd runs do not exist
        assert "Majority-class baseline" in text
        assert "Wilcoxon" in text and "BH p" in text
        assert "DESCRIPTIVE ONLY" in text
        assert "different folds" in text
        # combined CANE: 65 correct of 100 per fold, majority 70 % -> gain -5.0 pp
        cane_line = next(
            line for line in text.splitlines() if line.startswith("CANE") and "|" in line
        )
        assert "65.0" in cane_line and "70.0" in cane_line and "-5.0" in cane_line

    def test_no_runs_returns_error(self, tmp_path: Path) -> None:
        assert summary.main(["--root", str(tmp_path)], configure_logging=False) == 1


class TestSingleFoldRun:
    """A run with a single fold checkpoint must not crash ``nadeau_bengio_ttest``.

    ``nadeau_bengio_ttest(diffs, n_train=len(diffs) - 1, n_test=1)`` divides by ``n_train``, which
    is 0 for a single fold -- a bare ``ZeroDivisionError`` before the fix.
    """

    def test_paired_vs_majority_skips_stats_without_crashing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        write_run(tmp_path / "sad_041_t8-t7_ec", [_fold("sad", 30, 20, 10, 40)])
        run = summary.load_single_run(tmp_path / "sad_041_t8-t7_ec", "sad", "ec")
        assert run.n_folds == 1

        with caplog.at_level(logging.WARNING):
            result = summary.paired_vs_majority(run)

        assert "only 1 fold" in caplog.text
        assert np.isnan(result.wilcoxon_p)
        assert np.isnan(result.nb_t)
        assert np.isnan(result.nb_p)
        # chunk_acc = (30+20)/100 = 0.5; majority = max(70, 30)/100 = 0.7.
        assert result.mean_diff_pp == pytest.approx((0.5 - 0.7) * 100)
        assert result.n_above == 0

    def test_end_to_end_with_one_fold_and_ten_fold_runs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # cane: 10 folds, clearly above the majority rate (a real, non-NaN BH p-value).
        cane_folds = []
        for shift in range(10):
            cane_folds.append(_fold("cane", 52 + shift % 3, 36, 4, 8 - shift % 3))
        write_run(tmp_path / "cane_041_t8-t7_ec", cane_folds)
        # sad: only one fold checkpoint kept -- the case that used to crash.
        write_run(tmp_path / "sad_041_t8-t7_ec", [_fold("sad", 30, 20, 10, 40)])

        with caplog.at_level(logging.INFO):
            code = summary.main(["--root", str(tmp_path)], configure_logging=False)
        text = "\n".join(caplog.messages)

        assert code == 0
        # Main/baseline/combined tables still print for both runs.
        assert "CANE" in text and "SAD" in text
        assert "Majority-class baseline" in text
        assert "DESCRIPTIVE ONLY" in text
        assert "only 1 fold" in text

        # Within the paired-test section: the 1-fold SAD row shows n/a, the 10-fold CANE row
        # still gets a real (non-NaN-poisoned) BH-adjusted p-value.
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if "Per-fold paired test" in line)
        end = next(i for i, line in enumerate(lines[start:], start) if "combined training" in line)
        section = lines[start:end]
        sad_line = next(line for line in section if line.startswith("SAD"))
        cane_line = next(line for line in section if line.startswith("CANE"))
        assert "n/a" in sad_line
        assert "n/a" not in cane_line


class TestBalancedAccuracy:
    """Balanced-accuracy-vs-chance section (item 4 of the module docstring)."""

    def test_imbalanced_run_below_majority_but_above_chance(self, tmp_path: Path) -> None:
        # CANE-like: pos=72, neg=28 (72% pathological) every fold; tp/fn vary slightly across
        # folds (0, +1, +2 repeating) so the paired test has nonzero variance, while tn/fp (and
        # so specificity, 20/28) stay constant. Pooled accuracy still sits BELOW the 72% majority
        # rate, while balanced accuracy is well above chance -- the exact pattern real CANE shows
        # that this section exists to surface.
        shifts = [i % 3 for i in range(10)]  # 0,1,2,0,1,2,0,1,2,0 -> four 0s, three 1s, three 2s
        folds = [_fold("cane", 50 + s, 20, 8, 22 - s) for s in shifts]
        write_run(tmp_path / "cane_041_t8-t7_ec", folds)
        run = summary.load_single_run(tmp_path / "cane_041_t8-t7_ec", "cane", "ec")

        # Sanity check: reproduces "accuracy below majority" before looking at balanced accuracy.
        assert run.pooled.baseline == pytest.approx(0.72)
        assert run.pooled.accuracy < run.pooled.baseline

        result = summary.balanced_acc_vs_chance(run)
        bal_per_fold = np.array([((50 + s) / 72 + 20 / 28) / 2 for s in shifts])
        # Pooled tp = 4*50 + 3*51 + 3*52 = 509 (out of 720 positives); fp/tn fixed at 80/200.
        expected_pooled_bal = (509 / 720 + 200 / 280) / 2
        assert result.pooled_bal_acc == pytest.approx(expected_pooled_bal)
        assert result.per_fold_mean == pytest.approx(bal_per_fold.mean())
        assert result.per_fold_std == pytest.approx(bal_per_fold.std(ddof=1))
        assert result.n_valid_folds == 10
        assert result.n_dropped_folds == 0
        assert result.pooled_bal_acc > 0.5

        diffs = bal_per_fold - 0.5
        expected_t, expected_p = bipolar.nadeau_bengio_ttest(diffs, n_train=9, n_test=1)
        assert result.paired.mean_diff_pp == pytest.approx(diffs.mean() * 100)
        assert result.paired.nb_t == pytest.approx(expected_t)
        assert result.paired.nb_p == pytest.approx(expected_p)
        assert result.paired.nb_p < 0.01  # clearly above chance despite the small variation

    def test_fold_with_no_negatives_dropped_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Fold 3 has tn=0, fp=0 (no negatives in validation) -> specificity is NaN -> that fold's
        # balanced accuracy is undefined and must be dropped, not crash the test.
        folds = [_fold("cane", 50, 20, 8, 22)] * 10
        folds[2] = _fold("cane", 60, 0, 0, 12)
        write_run(tmp_path / "cane_041_t8-t7_ec", folds)
        run = summary.load_single_run(tmp_path / "cane_041_t8-t7_ec", "cane", "ec")
        assert np.isnan(run.counts[2].specificity)

        with caplog.at_level(logging.WARNING):
            result = summary.balanced_acc_vs_chance(run)

        assert "1 fold(s) have undefined balanced accuracy" in caplog.text
        assert "[3]" in caplog.text
        assert result.n_valid_folds == 9
        assert result.n_dropped_folds == 1
        # The remaining 9 valid folds are still tested (not crashed, not all dropped).
        assert not np.isnan(result.paired.nb_p)

    def test_single_fold_run_is_na_and_excluded_from_bh(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        write_run(tmp_path / "sad_041_t8-t7_ec", [_fold("sad", 30, 20, 10, 40)])
        write_run(tmp_path / "cane_041_t8-t7_ec", [_fold("cane", 50, 20, 8, 22)] * 10)

        with caplog.at_level(logging.INFO):
            code = summary.main(["--root", str(tmp_path)], configure_logging=False)
        assert code == 0
        text = "\n".join(caplog.messages)
        lines = text.splitlines()
        start = next(i for i, line in enumerate(lines) if "Balanced accuracy vs chance" in line)
        end = next(i for i, line in enumerate(lines[start:], start) if "combined training" in line)
        section = lines[start:end]
        sad_line = next(line for line in section if line.startswith("SAD"))
        cane_line = next(line for line in section if line.startswith("CANE"))
        assert "n/a" in sad_line
        assert "n/a" not in cane_line

    def test_combined_table_includes_balanced_accuracy(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        write_run(tmp_path / "cane_041_t8-t7_ec", [_fold("cane", 50, 20, 8, 22)] * 10)
        combined = [{"cane": (60, 5, 25, 10)} for _ in range(10)]
        write_run(tmp_path / "all_040_t8-t7_ec", combined, swapped=True)

        with caplog.at_level(logging.INFO):
            code = summary.main(["--root", str(tmp_path)], configure_logging=False)
        assert code == 0
        text = "\n".join(caplog.messages)
        cane_line = next(
            line for line in text.splitlines() if line.startswith("CANE") and "|" in line
        )
        # single balanced accuracy: (50/72 + 20/28) / 2 * 100 ~= 70.4
        assert "70.4" in cane_line
        # combined balanced accuracy: (60/70 + 5/30) / 2 * 100 ~= 51.2
        assert "51.2" in cane_line
