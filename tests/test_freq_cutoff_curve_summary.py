"""Tests for helpers-print/freq_cutoff_curve_summary.py using in-memory runs."""

from pathlib import Path

from tests.helper_scripts import load_helper_script

fccs = load_helper_script("freq_cutoff_curve_summary")
fcs = load_helper_script("freq_cutoff_summary")


def _run(chunk_acc: list[float]) -> "fcs.CutoffRun":
    return fcs.CutoffRun(run_dir=Path("x"), n_folds=len(chunk_acc), chunk_acc=chunk_acc)


def test_paired_vs_reference_signs_and_skips_mismatched_folds() -> None:
    ref = [0.80, 0.78, 0.82, 0.79, 0.81, 0.77, 0.83, 0.80, 0.78, 0.82]
    runs = {
        70: _run(ref),
        50: _run([a - 0.02 for a in ref]),
        40: _run([a - 0.05 for a in ref]),
        30: _run([a - 0.09 for a in ref]),
    }
    out = fccs.paired_vs_reference(runs, {50: True, 40: True, 30: False})
    assert sorted(out) == [40, 50]  # 30 Hz skipped: fold identity failed
    assert abs(out[50][0] - 0.02) < 1e-9
    assert abs(out[40][0] - 0.05) < 1e-9


def test_paired_vs_reference_without_reference_run_is_empty() -> None:
    assert fccs.paired_vs_reference({50: _run([0.7] * 10)}, {50: True}) == {}
