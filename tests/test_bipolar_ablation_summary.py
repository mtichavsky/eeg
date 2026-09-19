"""Tests for ``helpers-print/bipolar_ablation_summary.py`` using synthetic checkpoints.

``print_paired_stats`` calls ``nadeau_bengio_ttest(diffs, n_train=len(diffs) - 1, n_test=1)``.
For a run with a single fold checkpoint, ``len(diffs) == 1``, so ``n_train == 0`` and the
correction term ``n_test / n_train`` raises a bare ``ZeroDivisionError`` -- crashing the whole
summary if any montage/condition being compared has only one fold left (e.g. a sweep summarised
mid-run). These tests pin the fix mirrored from ``single_dataset_summary.py``'s
``paired_vs_majority``/``print_paired_table``: skip the significance test (NaN) with a printed
warning for fewer than 2 folds, print "n/a" for those cells, and exclude the NaN p-value from
``benjamini_hochberg`` so it cannot poison the other pairs' adjusted p-values.

Note: ``HEADSET_MONTAGES`` has exactly 3 entries, so a single condition has exactly 3 pairwise
comparisons. A "real" (>=2 matched folds) pair and a "degenerate" (matched, <2 folds) pair can
never coexist within the *same* condition -- a degenerate pair needs both its montages at <2
folds, and any third montage then either also has <2 folds (another degenerate pair, not a real
one) or a different fold count (a fold-count-mismatch, skipped entirely, not degenerate). So the
"does a NaN poison unrelated BH-adjusted values" case is tested across two conditions (ec: three
real pairs; eo: a degenerate pair plus mismatches) within the *same* ``print_paired_stats`` call,
which is exactly how a real sweep-in-progress summary would look (one condition finished with all
folds, the other still running with one fold saved so far).
"""

from pathlib import Path

import numpy as np
import pytest

from tests.helper_scripts import load_helper_script, write_run

bipolar = load_helper_script("bipolar_ablation_summary")


def _folds(
    rng: np.random.Generator, base_acc: float, n: int = 10, total: int = 100, dataset: str = "mdd"
) -> list[dict[str, tuple[int, int, int, int]]]:
    """``n`` folds of a synthetic run with chunk accuracy near ``base_acc`` (small per-fold noise
    so paired differences are neither all-identical -- which degenerates Wilcoxon -- nor huge).
    """
    folds = []
    for _ in range(n):
        correct = int(round(total * (base_acc + rng.uniform(-0.03, 0.03))))
        correct = max(0, min(total, correct))
        tp = correct // 2
        tn = correct - tp
        wrong = total - correct
        fp = wrong // 2
        fn = wrong - fp
        folds.append({dataset: (tp, tn, fp, fn)})
    return folds


def _single_fold(tp: int, tn: int, fp: int, fn: int) -> list[dict[str, tuple[int, int, int, int]]]:
    return [{"mdd": (tp, tn, fp, fn)}]


class TestPrintPairedStatsSingleFold:
    """The regression: a matched-length, single-fold pair must not raise ``ZeroDivisionError``."""

    def test_single_fold_pair_does_not_crash_and_prints_na(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_run(tmp_path / "fp2fp1", _single_fold(60, 30, 10, 0))
        write_run(tmp_path / "c4c3", _single_fold(55, 25, 15, 5))
        write_run(tmp_path / "t8t7", _single_fold(52, 22, 18, 8))
        run_a = bipolar.load_run(tmp_path / "fp2fp1")
        run_b = bipolar.load_run(tmp_path / "c4c3")
        run_c = bipolar.load_run(tmp_path / "t8t7")
        assert run_a.n_folds == run_b.n_folds == run_c.n_folds == 1

        runs = {"Fp2-Fp1": {"ec": run_a}, "C4-C3": {"ec": run_b}, "T8-T7": {"ec": run_c}}

        # Must complete without raising ZeroDivisionError.
        bipolar.print_paired_stats(runs, n_folds=1)

        out = capsys.readouterr().out
        assert "only 1 fold(s)" in out
        assert "n/a" in out

    def test_nadeau_bengio_ttest_itself_still_raises_for_n_train_zero(self) -> None:
        """The guard lives in ``print_paired_stats``, not in ``nadeau_bengio_ttest`` -- its
        signature and behaviour are left unchanged (``single_dataset_summary.py`` imports it).
        """
        with pytest.raises(ZeroDivisionError):
            bipolar.nadeau_bengio_ttest(np.array([0.05]), n_train=0, n_test=1)


class TestBenjaminiHochbergUnaffectedByNaN:
    """A NaN nb_p from a degenerate pair in one condition must not poison the BH-adjusted
    p-values of fully-powered pairs in another condition processed by the same call.

    (Within one condition this can't happen with only 3 montages -- see module docstring -- so
    this mirrors a real sweep summary: one condition (ec) fully done, the other (eo) with only
    one fold saved so far for two of the three montages.)
    """

    def test_ec_bh_values_unaffected_by_degenerate_eo_pair(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rng = np.random.default_rng(0)
        # ec: three distinct, real 10-fold runs -> three distinct real nb_p values.
        write_run(tmp_path / "fp2fp1_ec", _folds(rng, base_acc=0.82))
        write_run(tmp_path / "c4c3_ec", _folds(rng, base_acc=0.74))
        write_run(tmp_path / "t8t7_ec", _folds(rng, base_acc=0.66))
        ec_a = bipolar.load_run(tmp_path / "fp2fp1_ec")
        ec_b = bipolar.load_run(tmp_path / "c4c3_ec")
        ec_c = bipolar.load_run(tmp_path / "t8t7_ec")

        # eo: Fp2-Fp1 and C4-C3 each have a single matching fold (degenerate, NaN); T8-T7 has 10
        # folds, so it mismatches both of them (already-handled SKIPPED path).
        write_run(tmp_path / "fp2fp1_eo", _single_fold(50, 20, 20, 10))
        write_run(tmp_path / "c4c3_eo", _single_fold(48, 22, 18, 12))
        write_run(tmp_path / "t8t7_eo", _folds(rng, base_acc=0.70))
        eo_a = bipolar.load_run(tmp_path / "fp2fp1_eo")
        eo_b = bipolar.load_run(tmp_path / "c4c3_eo")
        eo_c = bipolar.load_run(tmp_path / "t8t7_eo")

        runs = {
            "Fp2-Fp1": {"ec": ec_a, "eo": eo_a},
            "C4-C3": {"ec": ec_b, "eo": eo_b},
            "T8-T7": {"ec": ec_c, "eo": eo_c},
        }

        # Independently compute the expected BH-adjusted p-values for ec's three real pairs.
        pairs = [("Fp2-Fp1", "C4-C3"), ("Fp2-Fp1", "T8-T7"), ("C4-C3", "T8-T7")]
        ec_runs = {"Fp2-Fp1": ec_a, "C4-C3": ec_b, "T8-T7": ec_c}
        expected_nb_p = []
        for a, b in pairs:
            diffs = np.asarray(ec_runs[a].chunk_acc) - np.asarray(ec_runs[b].chunk_acc)
            _, nb_p = bipolar.nadeau_bengio_ttest(diffs, n_train=9, n_test=1)
            expected_nb_p.append(nb_p)
        expected_bh = bipolar.benjamini_hochberg(expected_nb_p)

        bipolar.print_paired_stats(runs, n_folds=10)
        out = capsys.readouterr().out

        assert "n/a" in out  # the degenerate eo pairs
        assert "only 1 fold(s)" in out
        for bh in expected_bh:
            assert f"{bh:.4f}" in out


class TestMismatchedFoldCountsStillGracefulAlongsideFix:
    """Pairs with different fold counts between the two runs already print SKIPPED and continue
    (pre-existing behaviour); confirm this still holds unchanged next to the new < 2 fold guard.
    """

    def test_mismatched_fold_counts_do_not_crash(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rng = np.random.default_rng(1)
        write_run(tmp_path / "ten_folds_a", _folds(rng, base_acc=0.75, n=10))
        write_run(tmp_path / "ten_folds_b", _folds(rng, base_acc=0.65, n=10))
        write_run(tmp_path / "one_fold", _single_fold(50, 20, 20, 10))
        run_ten_a = bipolar.load_run(tmp_path / "ten_folds_a")
        run_ten_b = bipolar.load_run(tmp_path / "ten_folds_b")
        run_one = bipolar.load_run(tmp_path / "one_fold")

        runs = {
            "Fp2-Fp1": {"ec": run_ten_a},
            "C4-C3": {"ec": run_one},
            "T8-T7": {"ec": run_ten_b},
        }

        bipolar.print_paired_stats(runs, n_folds=10)
        out = capsys.readouterr().out
        assert "SKIPPED (fold count mismatch" in out
