"""Tests for per-dataset confusion matrix support in ``helpers-print/dataset_prior_baseline.py``.

Checkpoints and results.txt files are synthetic but built with the real
``compute_per_dataset_metrics`` / ``write_per_dataset_table``, so writer and parser are tested
against each other. Runs without the matrices are covered by ``test_dataset_prior_baseline.py``.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from tests.helper_scripts import (
    load_helper_script,
    metrics_from_matrices,
    write_fold_checkpoint,
    write_matrix_fold_checkpoint,
)
from thesis.metrics import _format_confusion_matrix, write_per_dataset_table

prior = load_helper_script("dataset_prior_baseline")

# 4-class confusion matrices (rows = true Healthy/Anxiety/Depression/Comorbid, columns = predicted).
# MDD has no Anxiety/Comorbid chunks, SAD no Depression/Comorbid ones, like the real datasets.
FOLD_A = {
    "mdd": np.array([[30, 0, 5, 5], [0, 0, 0, 0], [4, 0, 60, 1], [0, 0, 0, 0]]),
    "cane": np.array([[10, 5, 0, 15], [3, 20, 0, 7], [0, 0, 0, 0], [1, 4, 0, 35]]),
    "sad": np.array([[20, 4, 1, 0], [6, 10, 0, 4], [0, 0, 0, 0], [0, 0, 0, 0]]),
}
FOLD_B = {
    "mdd": np.array([[35, 0, 3, 2], [0, 0, 0, 0], [2, 0, 70, 3], [0, 0, 0, 0]]),
    "cane": np.array([[12, 4, 1, 13], [2, 25, 0, 3], [0, 0, 0, 0], [0, 5, 0, 30]]),
    "sad": np.array([[18, 6, 0, 1], [5, 12, 0, 3], [0, 0, 0, 0], [0, 0, 0, 0]]),
}


def _pooled(folds: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {d: np.sum([fold[d] for fold in folds], axis=0) for d in folds[0]}


def _write_run(run_dir: Path, folds: list[dict[str, np.ndarray]]) -> None:
    for i, fold in enumerate(folds, start=1):
        write_matrix_fold_checkpoint(run_dir / f"fold_{i}_best.pth", fold)


def _results_txt(path: Path, folds: list[dict[str, np.ndarray]], class_mode: int) -> None:
    """Write a results.txt tail with the pooled matrix and the per-dataset sections."""
    num_classes = folds[0]["mdd"].shape[0]
    computed = [metrics_from_matrices(fold) for fold in folds]
    lines: list[str] = ["Configuration:\n", f"  class_mode: {class_mode}\n", "\n"]
    _format_confusion_matrix(
        [chunk["confusion_matrix"] for chunk, _ in computed], num_classes, lines.append
    )
    write_per_dataset_table(lines.append, [per_dataset for _, per_dataset in computed])
    path.write_text("".join(lines))


class TestPriorBaselineFromMatrices:
    def test_single_fold_by_hand(self) -> None:
        rows, overall = prior.prior_baseline_from_matrices(FOLD_A)
        by_ds = {r.dataset: r for r in rows}

        # mdd: 105 chunks, correct 30+60 = 90, majority true class Depression (65)
        assert by_ds["mdd"].chunks == 105
        assert by_ds["mdd"].accuracy == pytest.approx(90 / 105)
        assert by_ds["mdd"].baseline == pytest.approx(65 / 105)
        assert by_ds["mdd"].majority == "D"
        assert by_ds["mdd"].gain_pp == pytest.approx(25 / 105 * 100)
        assert by_ds["mdd"].contribution_pp == pytest.approx(25 / 250 * 100)
        # cane: 100 chunks, correct 10+20+35 = 65, majority Comorbid (40)
        assert by_ds["cane"].accuracy == pytest.approx(0.65)
        assert by_ds["cane"].baseline == pytest.approx(0.40)
        assert by_ds["cane"].majority == "C"
        assert by_ds["cane"].class_shares == pytest.approx((0.30, 0.30, 0.0, 0.40))
        # sad: 45 chunks, correct 20+10 = 30, majority Healthy (25)
        assert by_ds["sad"].majority == "H"
        assert by_ds["sad"].pathological == pytest.approx(20 / 45)
        assert by_ds["sad"].contribution_pp == pytest.approx(5 / 250 * 100)
        # overall: correct 185 of 250, majority-total 65 + 40 + 25 = 130
        assert overall.chunks == 250
        assert overall.accuracy == pytest.approx(0.74)
        assert overall.baseline == pytest.approx(0.52)
        assert overall.gain_pp == pytest.approx(22.0)
        assert sum(r.contribution_pp for r in rows) == pytest.approx(overall.gain_pp)
        assert np.isnan(overall.sensitivity) and np.isnan(overall.specificity)

    def test_dataset_with_a_single_true_class_has_full_prior_baseline(self) -> None:
        # A dataset whose chunks are all one true class: the prior is 100 % accurate, so the model
        # can only lose against it (negative gain).
        only_healthy = {"sad": np.array([[8, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])}
        (row,), overall = prior.prior_baseline_from_matrices(only_healthy)

        assert row.baseline == 1.0
        assert row.gain_pp == pytest.approx(-20.0)
        assert overall.gain_pp == pytest.approx(-20.0)


class TestLoadRunWithMatrices:
    def test_four_class_run_pools_the_stored_matrices(self, tmp_path: Path) -> None:
        _write_run(tmp_path / "run", [FOLD_A, FOLD_B])

        run = prior.load_run(tmp_path / "run", expect_folds=2)

        assert not run.binary
        assert set(run.matrices) == {"mdd", "cane", "sad"}
        for dataset, matrix in _pooled([FOLD_A, FOLD_B]).items():
            np.testing.assert_array_equal(run.matrices[dataset], matrix)
        assert run.correct["mdd"] == 90 + 105
        assert run.total["cane"] == 100 + 95
        assert len(run.folds) == 2

    def test_binary_run_with_matrices_needs_no_orientation_detection(self, tmp_path: Path) -> None:
        # FP == FN in every fold: the pre-fix path raises "cannot be detected", the matrices don't.
        fold = {"mdd": np.array([[30, 5], [5, 60]]), "cane": np.array([[10, 8], [8, 74]])}
        _write_run(tmp_path / "run", [fold])

        run = prior.load_run(tmp_path / "run", expect_folds=1)

        assert run.binary
        assert run.counts["cane"] == prior.Counts(tp=74, tn=10, fp=8, fn=8)
        assert run.counts["mdd"] == prior.Counts(tp=60, tn=30, fp=5, fn=5)

    def test_binary_matrix_orientation_is_true_class_rows(self, tmp_path: Path) -> None:
        fold = {"mdd": np.array([[30, 7], [3, 60]])}  # TN=30 FP=7 FN=3 TP=60
        _write_run(tmp_path / "run", [fold])

        counts = prior.load_run(tmp_path / "run", expect_folds=1).counts["mdd"]

        assert (counts.tp, counts.tn, counts.fp, counts.fn) == (60, 30, 7, 3)
        assert counts.sensitivity == pytest.approx(60 / 63)
        assert counts.specificity == pytest.approx(30 / 37)

    def test_matrices_disagreeing_with_the_chunk_matrix_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "run" / "fold_1_best.pth"
        write_matrix_fold_checkpoint(path, FOLD_A)
        ckpt = torch.load(path, weights_only=False)
        ckpt["val_metrics"]["per_dataset"]["sad"]["confusion_matrix"][0, 0] += 1
        torch.save(ckpt, path)

        with pytest.raises(prior.CountMismatchError, match="chunk confusion matrix"):
            prior.load_run(tmp_path / "run", expect_folds=1)

    def test_a_fold_without_matrices_falls_back_to_the_old_path(self, tmp_path: Path) -> None:
        _write_run(tmp_path / "run", [FOLD_A, FOLD_B])
        path = tmp_path / "run" / "fold_2_best.pth"
        ckpt = torch.load(path, weights_only=False)
        for metrics in ckpt["val_metrics"]["per_dataset"].values():
            del metrics["confusion_matrix"]
        torch.save(ckpt, path)

        run = prior.load_run(tmp_path / "run", expect_folds=2)

        assert run.matrices == {}  # accuracy-only, exactly like a pre-matrix 4-class run
        assert not run.binary


class TestReporting:
    def test_four_class_run_gets_the_decomposition_table(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_run(tmp_path / "all_4c", [FOLD_A])

        with caplog.at_level(logging.INFO):
            code = prior.main(
                ["--root", str(tmp_path), "--expect-folds", "1", "all_4c"], configure_logging=False
            )

        assert code == 0
        text = caplog.text
        assert "=== all_4c" in text
        assert "Contrib pp" in text and "H%" in text and "C%" in text
        assert "baseline not computed" not in text
        assert "+22.0" in text  # overall gain
        assert "confusion matrix (rows" not in text  # matrices only on request
        assert "Summary" in text  # the run enters the cross-run summary too

    def test_confusion_matrices_flag_prints_each_dataset_matrix(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write_run(tmp_path / "all_4c", [FOLD_A])

        with caplog.at_level(logging.INFO):
            prior.main(
                ["--root", str(tmp_path), "--expect-folds", "1", "--confusion-matrices", "all_4c"],
                configure_logging=False,
            )

        assert "MDD confusion matrix (rows = actual, columns = predicted)" in caplog.text
        assert "CANE confusion matrix" in caplog.text and "SAD confusion matrix" in caplog.text
        # the first mdd row is [30, 0, 5, 5]
        assert ["H", "30", "0", "5", "5"] in [m.split() for m in caplog.messages]

    def test_confusion_matrices_flag_works_for_runs_without_stored_matrices(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:

        write_fold_checkpoint(
            tmp_path / "all_bin" / "fold_1_best.pth",
            {"mdd": (60, 30, 7, 3), "sad": (5, 25, 5, 15)},  # (tp, tn, fp, fn), true counts
            swapped=True,
        )

        with caplog.at_level(logging.INFO):
            prior.main(
                ["--root", str(tmp_path), "--expect-folds", "1", "--confusion-matrices", "all_bin"],
                configure_logging=False,
            )

        rows = [m.split() for m in caplog.messages]
        # mdd [[TN, FP], [FN, TP]] = [[30, 7], [3, 60]]; oriented despite the pre-fix storage
        assert ["H", "30", "7"] in rows
        assert ["P", "3", "60"] in rows

    def test_without_the_flag_binary_output_has_no_matrices(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:

        write_fold_checkpoint(
            tmp_path / "all_bin" / "fold_1_best.pth", {"mdd": (60, 30, 7, 3)}, swapped=True
        )

        with caplog.at_level(logging.INFO):
            prior.main(
                ["--root", str(tmp_path), "--expect-folds", "1", "all_bin"], configure_logging=False
            )

        assert "confusion matrix (rows" not in caplog.text


class TestResultsTxtMatrices:
    def test_four_class_results_txt_round_trips(self, tmp_path: Path) -> None:
        txt = tmp_path / "results.txt"
        _results_txt(txt, [FOLD_A, FOLD_B], class_mode=4)

        run = prior.load_from_results_txt(txt)

        assert not run.binary
        for dataset, matrix in _pooled([FOLD_A, FOLD_B]).items():
            np.testing.assert_array_equal(run.matrices[dataset], matrix)
        assert run.source == "results.txt"
        rows, overall = prior.prior_baseline_from_matrices(run.matrices)
        direct_rows, direct_overall = prior.prior_baseline_from_matrices(_pooled([FOLD_A, FOLD_B]))
        assert overall.gain_pp == pytest.approx(direct_overall.gain_pp)
        assert [r.majority for r in rows] == [r.majority for r in direct_rows]

    def test_binary_results_txt_with_matrices_is_exact(self, tmp_path: Path) -> None:
        # Both orientation readings of the printed sens/spec would be ambiguous here (FP == FN
        # for every dataset); the matrices are exact regardless.
        fold = {"mdd": np.array([[30, 5], [5, 60]]), "cane": np.array([[10, 8], [8, 74]])}
        txt = tmp_path / "results.txt"
        _results_txt(txt, [fold], class_mode=2)

        run = prior.load_from_results_txt(txt)

        assert run.binary
        assert run.counts["cane"] == prior.Counts(tp=74, tn=10, fp=8, fn=8)

    def test_existing_parser_still_reads_the_table_and_pooled_matrix(self, tmp_path: Path) -> None:
        """The added section must not disturb the old per-dataset table / pooled matrix parse."""
        fold = {"mdd": np.array([[30, 7], [3, 60]]), "cane": np.array([[10, 8], [9, 73]])}
        txt = tmp_path / "results.txt"
        _results_txt(txt, [fold], class_mode=2)

        rows, matrix = prior._parse_results_lines(txt.read_text().splitlines())

        assert set(rows) == {"mdd", "cane"}
        assert rows["mdd"][3] == 100
        pooled = np.sum([fold["mdd"], fold["cane"]], axis=0)
        np.testing.assert_array_equal(matrix, pooled)

    def test_results_txt_without_the_section_still_uses_the_old_path(self, tmp_path: Path) -> None:
        lines: list[str] = ["Configuration:\n", "  class_mode: 4\n"]
        legacy: list[dict[str, dict[str, Any]]] = [
            {"mdd": {"tp": 1, "tn": 2, "fp": 3, "fn": 4, "correct": 30, "total": 40}}
        ]
        write_per_dataset_table(lines.append, legacy)
        txt = tmp_path / "results.txt"
        txt.write_text("".join(lines))

        run = prior.load_from_results_txt(txt)

        assert run.matrices == {}
        assert run.correct == {"mdd": 30}
        assert "accuracy is available" in run.note

    def test_malformed_section_is_skipped(self, tmp_path: Path) -> None:
        txt = tmp_path / "results.txt"
        _results_txt(txt, [FOLD_A], class_mode=4)
        # Drop the last row of the last matrix: no longer square, entries no longer sum up.
        lines = txt.read_text().splitlines(keepends=True)
        txt.write_text("".join(lines[:-1]))

        with pytest.raises(prior.SkipRunError, match="malformed"):
            prior.load_from_results_txt(txt)

    def test_main_reports_a_four_class_results_txt(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        run_dir = tmp_path / "all_4c"
        run_dir.mkdir()
        _results_txt(run_dir / "results.txt", [FOLD_A], class_mode=4)

        with caplog.at_level(logging.INFO):
            prior.main(
                ["--root", str(tmp_path), "--from-results-txt", "all_4c"], configure_logging=False
            )

        assert "=== all_4c (results.txt" in caplog.text
        assert "+22.0" in caplog.text
