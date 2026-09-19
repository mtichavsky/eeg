"""Summarize the frequency-cutoff curve for CNN-AttnS (in-ear): 70, 50, 40 and 30 Hz.

Reads the 70/30 Hz runs of ``run-freq-cutoff-meta.sh`` (``all_041_inear_ec+eo_f{70,30}``) and the
50/40 Hz runs of ``run-freq-cutoff-curve-meta.sh`` (``all_042_inear_ec+eo_f{50,40}``) and prints:

1. Fold-identity check of every cutoff against the 70 Hz run (paired statistics are skipped for a
   cutoff whose folds differ).
2. Chunk/subject accuracy (mean +/- std over folds) and parameter count per cutoff.
3. Paired per-fold difference to 70 Hz (70 minus cutoff): mean, bootstrap 95% CI, Wilcoxon
   signed-rank p and the Nadeau-Bengio corrected p, Benjamini-Hochberg-adjusted across cutoffs.
4. Per-dataset chunk ACCURACY only (per-dataset sensitivity/specificity are unreliable in runs
   trained before the ``compute_per_dataset_metrics`` argument-order fix).

Usage:
    poetry run python helpers-print/freq_cutoff_curve_summary.py [--root experiments]
"""

import argparse
import sys
from pathlib import Path

# ``helpers-print`` is not a package (hyphen), so make the sibling modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bipolar_ablation_summary import _fmt, benjamini_hochberg  # noqa: E402
from freq_cutoff_summary import (  # noqa: E402
    DATASETS,
    CutoffRun,
    check_fold_identity,
    load_run,
    paired_comparison,
)

CURVE_RUNS: dict[int, str] = {
    70: "all_041_inear_ec+eo_f70",
    50: "all_042_inear_ec+eo_f50",
    40: "all_042_inear_ec+eo_f40",
    30: "all_041_inear_ec+eo_f30",
}
REFERENCE_CUTOFF = 70


def load_curve(root: Path) -> dict[int, CutoffRun]:
    """Load every run of :data:`CURVE_RUNS` that exists under ``root``.

    :param Path root: Directory containing the run directories.
    :return: Loaded runs by cutoff (Hz); missing or empty runs are skipped with a message.
    :rtype: dict[int, CutoffRun]
    """
    runs: dict[int, CutoffRun] = {}
    for cutoff, name in CURVE_RUNS.items():
        run_dir = root / name
        if not run_dir.exists():
            print(f"[skip] {run_dir} does not exist")
            continue
        try:
            runs[cutoff] = load_run(run_dir)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
    return runs


def paired_vs_reference(
    runs: dict[int, CutoffRun], fold_ok: dict[int, bool]
) -> dict[int, tuple[float, float, float, float, float, float]]:
    """Paired chunk-accuracy comparison of the 70 Hz run against every other cutoff.

    :param dict runs: Loaded runs by cutoff.
    :param dict fold_ok: Whether each cutoff has the same validation folds as the 70 Hz run.
    :return: Per cutoff: (mean diff, CI low, CI high, Wilcoxon p, Nadeau-Bengio p, BH-adjusted
        Wilcoxon p), differences in fractions (70 minus cutoff).
    :rtype: dict
    """
    ref = runs.get(REFERENCE_CUTOFF)
    if ref is None:
        return {}
    raw = {}
    for cutoff, run in runs.items():
        if cutoff == REFERENCE_CUTOFF or not fold_ok.get(cutoff, False):
            continue
        raw[cutoff] = paired_comparison(ref.chunk_acc, run.chunk_acc)
    cutoffs = sorted(raw)
    adjusted = benjamini_hochberg([raw[c].wilcoxon_p for c in cutoffs]) if cutoffs else []
    return {
        c: (
            raw[c].mean_diff,
            raw[c].ci_low,
            raw[c].ci_high,
            raw[c].wilcoxon_p,
            raw[c].nb_p,
            float(adj),
        )
        for c, adj in zip(cutoffs, adjusted)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("experiments"), help="Directory containing run dirs."
    )
    args = parser.parse_args()

    runs = load_curve(args.root)
    if REFERENCE_CUTOFF not in runs:
        print(f"[error] the {REFERENCE_CUTOFF} Hz reference run is missing; nothing to compare")
        return

    print("\n=== Fold-identity check (validation subjects vs the 70 Hz run) ===\n")
    fold_ok = {
        c: check_fold_identity(runs[REFERENCE_CUTOFF], run)
        for c, run in runs.items()
        if c != REFERENCE_CUTOFF
    }

    print("\n=== Curve: CNN-AttnS (in-ear), ec+eo, mean +/- std over folds (%) ===\n")
    print(f"{'Cut':>4} {'Folds':>5} {'Chunk Acc':>14} {'Subj Acc':>14} {'Params':>9}")
    for cutoff in sorted(runs, reverse=True):
        run = runs[cutoff]
        params = f"{run.param_count:,}" if run.param_count is not None else "n/a"
        print(
            f"{cutoff:>4} {run.n_folds:>5} {_fmt(run.chunk_acc):>14} "
            f"{_fmt(run.subject_acc):>14} {params:>9}"
        )

    print("\n=== Paired chunk accuracy: 70 Hz minus cutoff (positive = 70 Hz better) ===\n")
    for cutoff, (diff, lo, hi, w_p, nb_p, bh_p) in sorted(
        paired_vs_reference(runs, fold_ok).items(), reverse=True
    ):
        print(
            f"  {cutoff:>3} Hz  diff = {diff * 100:+6.2f}pp [{lo * 100:+6.2f}, {hi * 100:+6.2f}]  "
            f"Wilcoxon p = {w_p:.4f} (BH {bh_p:.4f})  NB p = {nb_p:.4f}"
        )

    print("\n=== Per-dataset chunk accuracy (%; CANE column is IDUN in-ear) ===\n")
    print(f"{'Cut':>4} " + " ".join(f"{d.upper():>14}" for d in DATASETS))
    for cutoff in sorted(runs, reverse=True):
        cells = [
            _fmt(runs[cutoff].per_dataset[d]) if d in runs[cutoff].per_dataset else "n/a"
            for d in DATASETS
        ]
        print(f"{cutoff:>4} " + " ".join(f"{c:>14}" for c in cells))


if __name__ == "__main__":
    main()
