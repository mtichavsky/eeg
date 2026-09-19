"""Tests for ``helpers-print/dataset_prior_baseline.py`` using synthetic checkpoints.

No dataset, model or GPU is involved: checkpoints are ``torch.save`` dicts with the same
``val_metrics`` layout ``main.py`` writes. Both fp/fn orientations are covered (the swapped one is
what every run trained before the ``compute_per_dataset_metrics`` argument-order fix stored).
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.helper_scripts import load_helper_script, write_fold_checkpoint, write_run
from thesis.metrics import write_per_dataset_table

prior = load_helper_script("dataset_prior_baseline")
Counts: Any = prior.Counts

# True per-dataset (tp, tn, fp, fn) for two folds. Fold 2 deliberately has FP != FN overall.
FOLD_1 = {
    "mdd": (40, 30, 10, 20),  # 100 chunks, 60 pathological
    "cane": (60, 4, 26, 10),  # 100 chunks, 70 pathological
    "sad": (5, 25, 5, 15),  # 50 chunks, 20 pathological
}
FOLD_2 = {
    "mdd": (45, 35, 5, 15),  # 100 chunks, 60 pathological
    "cane": (65, 3, 27, 5),  # 100 chunks, 70 pathological
    "sad": (8, 20, 10, 12),  # 50 chunks, 20 pathological
}


def _true_pooled(folds: list[dict[str, tuple[int, int, int, int]]]) -> dict[str, Any]:
    pooled: dict[str, Any] = {}
    for fold in folds:
        for name, (tp, tn, fp, fn) in fold.items():
            pooled[name] = pooled.get(name, Counts()) + Counts(tp, tn, fp, fn)
    return pooled


class TestDetectOrientation:
    cm = np.array([[100, 30], [20, 50]])  # TN=100, FP=30, FN=20, TP=50

    def test_unswapped(self) -> None:
        assert prior.detect_orientation(Counts(tp=50, tn=100, fp=30, fn=20), self.cm) == "unswapped"

    def test_swapped(self) -> None:
        assert prior.detect_orientation(Counts(tp=50, tn=100, fp=20, fn=30), self.cm) == "swapped"

    def test_ambiguous_when_fp_equals_fn(self) -> None:
        cm = np.array([[100, 25], [25, 50]])
        assert prior.detect_orientation(Counts(tp=50, tn=100, fp=25, fn=25), cm) == "ambiguous"

    def test_mismatch_raises_with_clear_message(self) -> None:
        with pytest.raises(prior.CountMismatchError, match="match neither"):
            prior.detect_orientation(Counts(tp=50, tn=99, fp=30, fn=20), self.cm)


class TestLoadRun:
    @pytest.mark.parametrize("swapped", [True, False])
    def test_both_orientations_recover_true_counts(self, tmp_path: Path, swapped: bool) -> None:
        write_run(tmp_path / "run", [FOLD_1, FOLD_2], swapped=swapped)
        run = prior.load_run(tmp_path / "run", expect_folds=2)
        assert run.counts == _true_pooled([FOLD_1, FOLD_2])
        assert {f.orientation for f in run.folds} == {"swapped" if swapped else "unswapped"}
        assert ("swapped" if swapped else "unswapped") in run.note

    def test_stored_swapped_counts_really_differ_from_truth(self, tmp_path: Path) -> None:
        """Guards the fixture: without correction the pooled fp/fn would be exchanged."""
        write_run(tmp_path / "run", [FOLD_1, FOLD_2], swapped=True)
        run = prior.load_run(tmp_path / "run", assume="unswapped", expect_folds=2)
        truth = _true_pooled([FOLD_1, FOLD_2])
        assert run.counts["cane"] == truth["cane"].swapped()
        assert run.counts["cane"] != truth["cane"]

    def test_mismatching_checkpoint_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "run" / "fold_1_best.pth"
        write_fold_checkpoint(path, FOLD_1, swapped=True)
        import torch

        ckpt = torch.load(path, weights_only=False)
        ckpt["val_metrics"]["per_dataset"]["mdd"]["tn"] += 1  # corrupt one count
        torch.save(ckpt, path)
        with pytest.raises(prior.CountMismatchError, match="fold 1"):
            prior.load_run(tmp_path / "run", expect_folds=1)

    def test_ambiguous_fold_follows_other_folds(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Fold A has FP == FN overall (ambiguous), fold B is unambiguous and swapped.
        fold_a = {"mdd": (10, 10, 3, 1), "cane": (10, 10, 1, 3)}  # FP=4, FN=4
        write_run(tmp_path / "run", [fold_a, FOLD_1], swapped=True)
        run = prior.load_run(tmp_path / "run", expect_folds=2)
        assert run.counts == _true_pooled([fold_a, FOLD_1])
        assert "FP == FN" in run.note

    def test_all_ambiguous_raises_unless_assumed(self, tmp_path: Path) -> None:
        fold = {"mdd": (10, 10, 3, 1), "cane": (10, 10, 1, 3)}
        write_run(tmp_path / "run", [fold], swapped=True)
        with pytest.raises(prior.CountMismatchError, match="--assume"):
            prior.load_run(tmp_path / "run", expect_folds=1)
        forced = prior.load_run(tmp_path / "run", assume="swapped", expect_folds=1)
        assert forced.counts == _true_pooled([fold])

    def test_missing_folds_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        write_run(tmp_path / "run", [FOLD_1], swapped=True)
        with caplog.at_level(logging.WARNING):
            prior.load_run(tmp_path / "run", expect_folds=10)
        assert "1 fold checkpoints found, expected 10" in caplog.text

    def test_run_without_checkpoints_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(prior.SkipRunError, match="no fold_N_best"):
            prior.load_run(tmp_path / "empty")

    def test_run_without_per_dataset_is_skipped(self, tmp_path: Path) -> None:
        write_fold_checkpoint(tmp_path / "old" / "fold_1_best.pth", FOLD_1, with_per_dataset=False)
        with pytest.raises(prior.SkipRunError, match="per_dataset"):
            prior.load_run(tmp_path / "old")

    def test_multiclass_checkpoint_reports_accuracy_only(self, tmp_path: Path) -> None:
        import torch

        path = tmp_path / "four" / "fold_1_best.pth"
        path.parent.mkdir()
        cm = np.arange(16).reshape(4, 4)
        per_dataset = {
            "mdd": {"correct": 30, "total": 40, "tp": 1, "tn": 2, "fp": 3, "fn": 4},
            "sad": {"correct": 5, "total": 10, "tp": 0, "tn": 0, "fp": 0, "fn": 0},
        }
        torch.save(
            {"val_metrics": {"chunk": {"confusion_matrix": cm}, "per_dataset": per_dataset}}, path
        )
        run = prior.load_run(tmp_path / "four", expect_folds=1)
        assert not run.binary
        assert run.correct == {"mdd": 30, "sad": 5}
        assert run.total == {"mdd": 40, "sad": 10}


class TestPriorBaseline:
    def test_hand_computed_baseline_and_contributions(self) -> None:
        """Pooled over FOLD_1 + FOLD_2, worked out by hand.

        MDD:  200 chunks, 120 pathological -> baseline 120/200; correct = 40+30+45+35 = 150.
        CANE: 200 chunks, 140 pathological -> baseline 140/200; correct = 60+4+65+3 = 132.
        SAD:  100 chunks,  40 pathological -> baseline  60/100; correct = 5+25+8+20 = 58.
        Overall: 500 chunks, correct 340, majority 320 -> accuracy 68 %, baseline 64 %.
        """
        rows, overall = prior.prior_baseline(_true_pooled([FOLD_1, FOLD_2]))
        by_name = {r.dataset: r for r in rows}
        assert [r.dataset for r in rows] == ["mdd", "cane", "sad"]
        assert by_name["mdd"].baseline == pytest.approx(0.60)
        assert by_name["cane"].baseline == pytest.approx(0.70)
        assert by_name["sad"].baseline == pytest.approx(0.60)
        assert by_name["sad"].majority == "H"
        assert by_name["mdd"].majority == "P"
        assert overall.accuracy == pytest.approx(0.68)
        assert overall.baseline == pytest.approx(0.64)
        assert overall.gain_pp == pytest.approx(4.0)
        assert by_name["mdd"].contribution_pp == pytest.approx(30 / 500 * 100)
        assert by_name["cane"].contribution_pp == pytest.approx(-8 / 500 * 100)
        assert by_name["sad"].contribution_pp == pytest.approx(-2 / 500 * 100)
        assert sum(r.contribution_pp for r in rows) == pytest.approx(overall.gain_pp)

    def test_sensitivity_specificity_use_true_orientation(self, tmp_path: Path) -> None:
        write_run(tmp_path / "run", [FOLD_1], swapped=True)
        run = prior.load_run(tmp_path / "run", expect_folds=1)
        rows, _ = prior.prior_baseline(run.counts)
        mdd = {r.dataset: r for r in rows}["mdd"]
        assert mdd.sensitivity == pytest.approx(40 / 60)  # TP / (TP + FN), not TP / (TP + FP)
        assert mdd.specificity == pytest.approx(30 / 40)


class TestCli:
    def test_main_reports_baseline_and_skipped_runs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        write_run(tmp_path / "all_a", [FOLD_1, FOLD_2], swapped=True)
        (tmp_path / "all_b").mkdir()
        with caplog.at_level(logging.INFO):
            code = prior.main(
                ["--root", str(tmp_path), "--expect-folds", "2", "all_*"], configure_logging=False
            )
        assert code == 0
        assert "=== all_a" in caplog.text
        assert "+4.0" in caplog.text  # overall gain
        assert "Skipped runs" in caplog.text
        assert "all_b" in caplog.text


class TestFromResultsTxt:
    @staticmethod
    def _results_txt(
        folds: list[dict[str, tuple[int, int, int, int]]], swapped: bool, path: Path
    ) -> None:
        """Write a results.txt table using the real ``write_per_dataset_table``."""
        fold_metrics: list[dict[str, dict[str, float | int]]] = []
        for fold in folds:
            metrics: dict[str, dict[str, float | int]] = {}
            for name, (tp, tn, fp, fn) in fold.items():
                stored_fp, stored_fn = (fn, fp) if swapped else (fp, fn)
                metrics[name] = {
                    "tp": tp,
                    "tn": tn,
                    "fp": stored_fp,
                    "fn": stored_fn,
                    "correct": tp + tn,
                    "total": tp + tn + fp + fn,
                }
            fold_metrics.append(metrics)
        lines: list[str] = []
        write_per_dataset_table(lines.append, fold_metrics)
        path.write_text("".join(lines))

    @pytest.mark.parametrize("swapped", [True, False])
    def test_reconstruction_matches_truth_within_rounding(
        self, tmp_path: Path, swapped: bool
    ) -> None:
        # Realistic sizes: with N ~ 1000-4000 chunks per dataset the rounded percentages pin the
        # counts down to a few chunks.
        big = [
            {
                "mdd": (900, 700, 150, 120),
                "cane": (1300, 350, 400, 300),
                "sad": (250, 300, 100, 140),
            }
        ]
        txt = tmp_path / "results.txt"
        self._results_txt(big, swapped=swapped, path=txt)
        run = prior.load_from_results_txt(txt, "swapped" if swapped else "fixed")
        truth = _true_pooled(big)
        for name, true_counts in truth.items():
            got = run.counts[name]
            assert got.total == true_counts.total
            for field_name in ("tp", "tn", "fp", "fn"):
                assert abs(getattr(got, field_name) - getattr(true_counts, field_name)) <= 4, (
                    name,
                    field_name,
                )
        rows_true, overall_true = prior.prior_baseline(truth)
        rows_got, overall_got = prior.prior_baseline(run.counts)
        assert overall_got.gain_pp == pytest.approx(overall_true.gain_pp, abs=0.2)
        for r_true, r_got in zip(rows_true, rows_got):
            assert r_got.contribution_pp == pytest.approx(r_true.contribution_pp, abs=0.2)

    def test_degenerate_equal_ppv_npv_raises(self) -> None:
        with pytest.raises(ValueError, match="not identifiable"):
            prior.reconstruct_counts(0.75, 0.8, 0.8, 1000, "swapped")

    def test_missing_table_is_skipped(self, tmp_path: Path) -> None:
        txt = tmp_path / "results.txt"
        txt.write_text("no table here\n")
        with pytest.raises(prior.SkipRunError, match="Per-Dataset"):
            prior.load_from_results_txt(txt)

    @staticmethod
    def _cm_lines(counts: Any) -> list[str]:
        """Aggregated confusion matrix lines (see ``thesis.metrics._format_confusion_matrix``)."""
        return [
            f"      Healthy      {counts.tn:>10}      {counts.fp:>10}\n",
            f" Pathological      {counts.fn:>10}      {counts.tp:>10}\n",
        ]


class TestReconstructDegenerateCounts:
    """Round-trips through the printed-percentage formula of ``write_per_dataset_table``."""

    @staticmethod
    def _round2(value: float) -> float:
        return round(value * 100, 2) / 100

    @pytest.mark.parametrize(
        "tp, tn, fp, fn",
        [
            (0, 700, 300, 0),  # PPV == 0 (tp == 0), NPV == 1 (well-defined, non-degenerate)
            (400, 0, 0, 600),  # NPV == 0 (tn == 0), PPV == 1
            (0, 500, 0, 500),  # PPV == 0 via the tp+fp == 0 default in write_per_dataset_table
        ],
    )
    def test_swapped_round_trip_with_zero_ppv_or_npv(
        self, tp: int, tn: int, fp: int, fn: int
    ) -> None:
        n = tp + tn + fp + fn
        accuracy = self._round2((tp + tn) / n)
        ppv = self._round2(tp / (tp + fp)) if (tp + fp) else 0.0
        npv = self._round2(tn / (tn + fn)) if (tn + fn) else 0.0
        got = prior.reconstruct_counts(accuracy, ppv, npv, n, "swapped")
        assert got.total == n
        for field_name, truth in (("tp", tp), ("tn", tn), ("fp", fp), ("fn", fn)):
            assert abs(getattr(got, field_name) - truth) <= 1, field_name

    def test_ppv_and_npv_both_zero_is_truly_unidentifiable(self) -> None:
        # tp == 0 and tn == 0: correct == 0, so wrong == n can't be split into fp/fn at all.
        with pytest.raises(prior.UnidentifiableCountsError, match="both 0"):
            prior.reconstruct_counts(0.0, 0.0, 0.0, 1000, "swapped")

    def test_equal_nonzero_ppv_npv_still_raises_unidentifiable(self) -> None:
        with pytest.raises(prior.UnidentifiableCountsError, match="not identifiable"):
            prior.reconstruct_counts(0.75, 0.8, 0.8, 1000, "swapped")

    def test_out_of_range_sens_raises_plain_valueerror(self) -> None:
        # Not a "counts are unidentifiable" situation -- a caller bug -- so a plain ValueError,
        # not the dedicated UnidentifiableCountsError that main() treats as a per-run skip.
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            prior.reconstruct_counts(0.5, 1.5, 0.5, 100, "swapped")
        with pytest.raises(Exception) as exc_info:
            prior.reconstruct_counts(0.5, 1.5, 0.5, 100, "swapped")
        assert not isinstance(exc_info.value, prior.UnidentifiableCountsError)

    def test_fixed_orientation_handles_zero_sens_without_special_casing(self) -> None:
        # The "fixed" linear system doesn't divide by sens/spec directly, so a 0 value on either
        # side degrades gracefully -- only sens == spec (the "swapped" test above) is singular.
        tp, tn, fp, fn = 0, 700, 300, 0
        n = tp + tn + fp + fn
        accuracy = self._round2((tp + tn) / n)
        sens = self._round2(tp / (tp + fn)) if (tp + fn) else 0.0
        spec = self._round2(tn / (tn + fp)) if (tn + fp) else 0.0
        got = prior.reconstruct_counts(accuracy, sens, spec, n, "fixed")
        assert got.total == n
        for field_name, truth in (("tp", tp), ("tn", tn), ("fp", fp), ("fn", fn)):
            assert abs(getattr(got, field_name) - truth) <= 1, field_name


class TestMulticlassResultsTxt:
    """A >2-class results.txt must not be reconstructed as a binary dataset-prior baseline."""

    def test_detected_from_class_mode_config_line(self, tmp_path: Path) -> None:
        fold_metrics: list[dict[str, dict[str, float | int]]] = [
            {
                "mdd": {"tp": 30, "tn": 40, "fp": 10, "fn": 20, "correct": 70, "total": 100},
                "cane": {"tp": 50, "tn": 20, "fp": 10, "fn": 20, "correct": 70, "total": 100},
            }
        ]
        lines: list[str] = ["Configuration:\n", "  command: train\n", "  class_mode: 4\n"]
        write_per_dataset_table(lines.append, fold_metrics)
        txt = tmp_path / "results.txt"
        txt.write_text("".join(lines))

        run = prior.load_from_results_txt(txt)
        assert not run.binary
        assert run.counts == {}
        assert run.correct == {"mdd": 70, "cane": 70}
        assert run.total == {"mdd": 100, "cane": 100}
        assert "multi-class" in run.note

    def test_detected_from_confusion_matrix_without_class_mode_line(self, tmp_path: Path) -> None:
        # No "class_mode" config line at all (e.g. an older results.txt) -- fall back to the
        # confusion-matrix row labels, which only a >2-class run has.
        lines = [
            "ACTUAL v/PRED ->        Healthy         Anxiety      Depression        Comorbid \n",
            "        Healthy             100              10               5               2 \n",
            "        Anxiety              10             100               1               1 \n",
            "     Depression               2               1             100               1 \n",
            "       Comorbid               1               1               1             100 \n",
        ]
        fold_metrics: list[dict[str, dict[str, float | int]]] = [
            {"sad": {"tp": 1, "tn": 1, "fp": 1, "fn": 1, "correct": 45, "total": 50}}
        ]
        write_per_dataset_table(lines.append, fold_metrics)
        txt = tmp_path / "results.txt"
        txt.write_text("".join(lines))

        run = prior.load_from_results_txt(txt)
        assert not run.binary
        assert run.correct == {"sad": 45}
        assert run.total == {"sad": 50}

    def test_binary_results_txt_is_not_flagged_multiclass(self, tmp_path: Path) -> None:
        txt = tmp_path / "results.txt"
        TestFromResultsTxt._results_txt([FOLD_1], swapped=True, path=txt)
        run = prior.load_from_results_txt(txt, "swapped")
        assert run.binary


class TestAutoResultsOrientation:
    """``--results-orientation auto`` (default): pick the reading that fits the file's own CM."""

    def test_auto_detects_swapped(self, tmp_path: Path) -> None:
        big = {"mdd": (900, 700, 150, 120), "cane": (1300, 350, 400, 300)}
        txt = tmp_path / "results.txt"
        TestFromResultsTxt._results_txt([big], swapped=True, path=txt)
        with txt.open("a") as f:
            f.writelines(
                TestFromResultsTxt._cm_lines(sum((Counts(*v) for v in big.values()), Counts()))
            )

        run = prior.load_from_results_txt(txt)
        assert "auto-detected swapped" in run.note
        truth = _true_pooled([big])
        for name, true_counts in truth.items():
            for field_name in ("tp", "tn", "fp", "fn"):
                assert (
                    abs(getattr(run.counts[name], field_name) - getattr(true_counts, field_name))
                    <= 4
                )

    def test_auto_detects_fixed(self, tmp_path: Path) -> None:
        big = {"mdd": (900, 700, 150, 120), "cane": (1300, 350, 400, 300)}
        txt = tmp_path / "results.txt"
        TestFromResultsTxt._results_txt([big], swapped=False, path=txt)
        with txt.open("a") as f:
            f.writelines(
                TestFromResultsTxt._cm_lines(sum((Counts(*v) for v in big.values()), Counts()))
            )

        run = prior.load_from_results_txt(txt)
        assert "auto-detected fixed" in run.note
        truth = _true_pooled([big])
        for name, true_counts in truth.items():
            for field_name in ("tp", "tn", "fp", "fn"):
                assert (
                    abs(getattr(run.counts[name], field_name) - getattr(true_counts, field_name))
                    <= 4
                )

    def test_missing_confusion_matrix_is_skipped_with_hint(self, tmp_path: Path) -> None:
        txt = tmp_path / "results.txt"
        TestFromResultsTxt._results_txt([FOLD_1], swapped=True, path=txt)
        with pytest.raises(prior.UnidentifiableCountsError, match="--results-orientation"):
            prior.load_from_results_txt(txt)

    def test_tie_is_skipped_with_hint(self, tmp_path: Path) -> None:
        # Both orientations reconstruct these counts with zero deviation from the aggregated
        # matrix (found by search) -- neither is preferred, so auto-detection can't choose.
        fold = {"mdd": (4, 46, 16, 17), "cane": (50, 5, 29, 28)}
        txt = tmp_path / "results.txt"
        TestFromResultsTxt._results_txt([fold], swapped=True, path=txt)
        total = sum((Counts(*v) for v in fold.values()), Counts())
        with txt.open("a") as f:
            f.writelines(TestFromResultsTxt._cm_lines(total))

        with pytest.raises(prior.UnidentifiableCountsError, match="equally well"):
            prior.load_from_results_txt(txt)


class TestSkippedRunsDoNotAbort:
    """A bad run (checkpoint or results.txt) must not prevent the rest of the command running."""

    def test_checkpoint_count_mismatch_is_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import torch

        write_run(tmp_path / "good", [FOLD_1, FOLD_2], swapped=True)
        bad_path = tmp_path / "bad" / "fold_1_best.pth"
        write_fold_checkpoint(bad_path, FOLD_1, swapped=True)
        ckpt = torch.load(bad_path, weights_only=False)
        ckpt["val_metrics"]["per_dataset"]["mdd"]["tn"] += 1  # corrupt one count
        torch.save(ckpt, bad_path)

        with caplog.at_level(logging.INFO):
            code = prior.main(
                ["--root", str(tmp_path), "--expect-folds", "2", "good", "bad"],
                configure_logging=False,
            )
        assert code == 0
        assert "=== good" in caplog.text
        assert "Skipped runs" in caplog.text
        assert "bad" in caplog.text

    def test_results_txt_unidentifiable_run_is_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        good_dir = tmp_path / "good"
        good_dir.mkdir()
        TestFromResultsTxt._results_txt([FOLD_1], swapped=True, path=good_dir / "results.txt")

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        # sens == spec (both 0.5): unidentifiable under the forced "swapped" orientation.
        fold_metrics: list[dict[str, dict[str, float | int]]] = [
            {"mdd": {"tp": 50, "tn": 50, "fp": 50, "fn": 50, "correct": 100, "total": 200}}
        ]
        lines: list[str] = []
        write_per_dataset_table(lines.append, fold_metrics)
        (bad_dir / "results.txt").write_text("".join(lines))

        with caplog.at_level(logging.INFO):
            code = prior.main(
                [
                    "--root",
                    str(tmp_path),
                    "--from-results-txt",
                    "--results-orientation",
                    "swapped",
                    "good",
                    "bad",
                ],
                configure_logging=False,
            )
        assert code == 0
        assert "=== good" in caplog.text
        assert "Skipped runs" in caplog.text
        assert "bad" in caplog.text
