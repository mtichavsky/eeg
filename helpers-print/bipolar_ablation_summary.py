"""Summarize the bipolar surrogate ablation (Fp2-Fp1 vs C4-C3 vs T8-T7 vs in-ear).

Reads the run directories produced by ``run-bipolar.sh`` (see
docs/plans/2026-09-14-bipolar-surrogate-ablation.md) and prints:

1. Main table: montage x {EC, EO}, chunk/subject accuracy, sensitivity, specificity
   as mean +/- std over folds.
2. Per-dataset (MDD/CANE/SAD) chunk accuracy per run.
3. Paired statistics across the three headset montages (Fp2-Fp1, C4-C3, T8-T7), which
   share identical folds within a condition: Wilcoxon signed-rank + the Nadeau-Bengio
   corrected paired t-test, with Benjamini-Hochberg correction over the 3 pairwise
   comparisons within each condition.
4. The in-ear (IDUN) row reported descriptively (unpaired: different folds from the
   headset montages -- see the plan's Facts table).

Before trusting the paired statistics, run --check-folds: it confirms the "Val subjects
(N): [...]" log lines match fold-by-fold across the three headset montages, per condition.

Usage:
    poetry run python helpers-print/bipolar_ablation_summary.py [--root experiments] [--check-folds]
"""

import argparse
import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import wilcoxon

# Montage -> {condition -> run directory name}, matching the plan's run table.
RUNS: dict[str, dict[str, str]] = {
    "Fp2-Fp1": {"ec": "all_040_fp2-fp1_ec", "eo": "all_040_fp2-fp1_eo"},
    "C4-C3": {"ec": "all_040_c4-c3_ec", "eo": "all_040_c4-c3_eo"},
    "T8-T7": {"ec": "all_040_t8-t7_ec", "eo": "all_040_t8-t7_eo"},
    "in-ear": {"ec": "all_040_inear_ec", "eo": "all_040_inear_eo"},
}
HEADSET_MONTAGES = ["Fp2-Fp1", "C4-C3", "T8-T7"]
CONDITIONS = ["ec", "eo"]
DATASETS = ["mdd", "cane", "sad"]

FOLD_FILE_RE = re.compile(r"fold_(\d+)_best\.pth$")
VAL_SUBJECTS_RE = re.compile(r"Val subjects \((\d+)\): (\[.*\])")


@dataclass
class RunMetrics:
    run_dir: Path
    n_folds: int
    chunk_acc: list[float] = field(default_factory=list)
    chunk_sens: list[float] = field(default_factory=list)
    chunk_spec: list[float] = field(default_factory=list)
    subject_acc: list[float] = field(default_factory=list)
    subject_sens: list[float] = field(default_factory=list)
    subject_spec: list[float] = field(default_factory=list)
    # per_dataset[dataset][fold_idx] = chunk accuracy
    per_dataset: dict[str, list[float]] = field(default_factory=dict)


def _fold_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    folds = []
    for path in run_dir.glob("fold_*_best.pth"):
        match = FOLD_FILE_RE.search(path.name)
        if match:
            folds.append((int(match.group(1)), path))
    return sorted(folds)


def load_run(run_dir: Path) -> RunMetrics:
    """Load per-fold chunk/subject/per-dataset metrics from a run's checkpoints."""
    folds = _fold_checkpoints(run_dir)
    if not folds:
        raise FileNotFoundError(f"No fold_N_best.pth files found in {run_dir}")

    metrics = RunMetrics(run_dir=run_dir, n_folds=len(folds))
    for _, path in folds:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        val_metrics = ckpt["val_metrics"]
        chunk, subject = val_metrics["chunk"], val_metrics["subject"]

        metrics.chunk_acc.append(float(chunk["accuracy"]))
        metrics.chunk_sens.append(float(chunk["recall"]))
        metrics.chunk_spec.append(float(chunk["specificity"]))
        metrics.subject_acc.append(float(subject["accuracy"]))
        metrics.subject_sens.append(float(subject["recall"]))
        metrics.subject_spec.append(float(subject["specificity"]))

        for dataset_name, dataset_metrics in val_metrics.get("per_dataset", {}).items():
            metrics.per_dataset.setdefault(dataset_name, []).append(
                float(dataset_metrics["accuracy"])
            )

    return metrics


def _fmt(values: list[float]) -> str:
    arr = np.asarray(values) * 100
    return f"{arr.mean():5.2f} +/- {arr.std(ddof=1):4.2f}"


def print_main_table(runs: dict[str, dict[str, RunMetrics]]) -> None:
    print("\n=== Main table: chunk/subject accuracy, sensitivity, specificity (%) ===\n")
    header = (
        f"{'Montage':<10} {'Cond':<4} {'Chunk Acc':>14} {'Chunk Sens':>14} "
        f"{'Chunk Spec':>14} {'Subj Acc':>14} {'Subj Sens':>14} {'Subj Spec':>14}"
    )
    print(header)
    print("-" * len(header))
    for montage in [*HEADSET_MONTAGES, "in-ear"]:
        for condition in CONDITIONS:
            run = runs.get(montage, {}).get(condition)
            if run is None:
                continue
            label = montage if condition == "ec" else ""
            print(
                f"{label:<10} {condition:<4} "
                f"{_fmt(run.chunk_acc):>14} {_fmt(run.chunk_sens):>14} "
                f"{_fmt(run.chunk_spec):>14} {_fmt(run.subject_acc):>14} "
                f"{_fmt(run.subject_sens):>14} {_fmt(run.subject_spec):>14}"
            )
        if montage == "in-ear":
            print("  (in-ear is unpaired with the headset montages -- different folds)")


def print_per_dataset_table(runs: dict[str, dict[str, RunMetrics]]) -> None:
    print("\n=== Per-dataset chunk accuracy (%) ===\n")
    header = f"{'Montage':<10} {'Cond':<4} {'MDD':>14} {'CANE':>14} {'SAD':>14}"
    print(header)
    print("-" * len(header))
    for montage in [*HEADSET_MONTAGES, "in-ear"]:
        for condition in CONDITIONS:
            run = runs.get(montage, {}).get(condition)
            if run is None:
                continue
            cells = [
                _fmt(run.per_dataset[d]) if d in run.per_dataset else "n/a".rjust(14)
                for d in DATASETS
            ]
            print(f"{montage:<10} {condition:<4} " + " ".join(f"{c:>14}" for c in cells))
    print(
        "\nOn MDD and SAD, T8-T7 and in-ear are identical inputs -- any difference there "
        "comes only from training on different CANE data.\n"
        "On CANE, T8-T7 vs in-ear is the direct headset-surrogate vs real-in-ear comparison."
    )


def nadeau_bengio_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau-Bengio corrected paired t-test for k-fold cross-validation.

    Standard paired t-tests assume independent samples; CV folds share training data
    and are not independent, which inflates false-positive rates. This correction
    inflates the variance estimate by (1/k + n_test/n_train) to compensate.

    :param np.ndarray diffs: Per-fold differences (metric_a - metric_b), one per fold.
    :param int n_train: Number of training subjects in one fold (assumed equal size).
    :param int n_test: Number of test/validation subjects in one fold.
    :return: (t_statistic, two_sided_p_value).
    :rtype: tuple[float, float]
    """
    from scipy.stats import t as t_dist

    k = len(diffs)
    mean_diff = diffs.mean()
    var_diff = diffs.var(ddof=1)
    correction = 1.0 / k + n_test / n_train
    denom = np.sqrt(correction * var_diff)
    if denom == 0:
        return 0.0, 1.0
    t_stat = mean_diff / denom
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=k - 1))
    return float(t_stat), float(p_value)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in the input order."""
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = np.asarray(p_values)[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity (adjusted p-values must not decrease going backwards)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty(n)
    result[order] = adjusted
    return result.tolist()


def print_paired_stats(runs: dict[str, dict[str, RunMetrics]], n_folds: int = 10) -> None:
    print("\n=== Paired statistics across headset montages (chunk accuracy) ===\n")
    print(
        "Nadeau-Bengio correction assumes equal fold sizes: n_test/n_train ~= "
        f"1/{n_folds - 1} for {n_folds}-fold CV. Refine if fold sizes are very unequal.\n"
    )
    pairs = [
        (HEADSET_MONTAGES[i], HEADSET_MONTAGES[j])
        for i in range(len(HEADSET_MONTAGES))
        for j in range(i + 1, len(HEADSET_MONTAGES))
    ]
    for condition in CONDITIONS:
        print(f"-- condition: {condition} --")
        raw_p_values = []
        rows = []
        for a, b in pairs:
            run_a = runs.get(a, {}).get(condition)
            run_b = runs.get(b, {}).get(condition)
            if run_a is None or run_b is None:
                continue
            acc_a = np.asarray(run_a.chunk_acc)
            acc_b = np.asarray(run_b.chunk_acc)
            if len(acc_a) != len(acc_b):
                print(f"  {a} vs {b}: SKIPPED (fold count mismatch: {len(acc_a)} vs {len(acc_b)})")
                continue
            diffs = acc_a - acc_b
            try:
                _, wilcoxon_p = wilcoxon(diffs)
            except ValueError as exc:
                wilcoxon_p = float("nan")
                print(f"  {a} vs {b}: Wilcoxon failed ({exc})")
            n_folds_here = len(diffs)
            t_stat, nb_p = nadeau_bengio_ttest(diffs, n_train=n_folds_here - 1, n_test=1)
            rows.append((a, b, diffs.mean() * 100, wilcoxon_p, t_stat, nb_p))
            raw_p_values.append(nb_p)

        if not rows:
            continue
        adjusted = benjamini_hochberg(raw_p_values)
        for (a, b, mean_diff_pct, wilcoxon_p, t_stat, nb_p), nb_p_adj in zip(rows, adjusted):
            print(
                f"  {a:>8} - {b:<8}: mean diff = {mean_diff_pct:+.2f}pp, "
                f"Wilcoxon p = {wilcoxon_p:.4f}, "
                f"NB t = {t_stat:+.3f} (p = {nb_p:.4f}, BH-adjusted p = {nb_p_adj:.4f})"
            )


def parse_val_subjects(log_path: Path) -> list[list[str]]:
    """Parse "Val subjects (N): [...]" lines from a training log, in fold order."""
    if not log_path.exists():
        return []
    fold_subjects = []
    with open(log_path) as f:
        for line in f:
            match = VAL_SUBJECTS_RE.search(line)
            if match:
                subjects = ast.literal_eval(match.group(2))
                fold_subjects.append(sorted(subjects))
    return fold_subjects


def _find_training_log(run_dir: Path) -> Optional[Path]:
    """Return the most recently modified ``training_*.log`` in ``run_dir``, or None.

    A restarted run can leave multiple ``training_*.log`` files behind (e.g. a partial log
    from a crashed attempt plus the full one from the successful rerun). ``glob()`` order is
    not chronological, so picking ``logs[0]`` risks silently reading a stale/partial log
    instead of the current run's. Sorting by mtime and warning when more than one is found
    makes the ambiguity visible instead of picking one arbitrarily.
    """
    logs = list(run_dir.glob("training_*.log"))
    if not logs:
        return None
    if len(logs) > 1:
        chosen = max(logs, key=lambda p: p.stat().st_mtime)
        print(
            f"  WARNING: {len(logs)} training_*.log files found in {run_dir}, "
            f"using most recent: {chosen.name}"
        )
        return chosen
    return logs[0]


def check_fold_identity(runs: dict[str, dict[str, RunMetrics]]) -> bool:
    """Confirm the three headset montages saw the same val-subject sets, fold by fold."""
    print("\n=== Fold-identity check (headset montages) ===\n")
    all_match = True
    for condition in CONDITIONS:
        subject_lists = {}
        for montage in HEADSET_MONTAGES:
            run = runs.get(montage, {}).get(condition)
            if run is None:
                continue
            log_path = _find_training_log(run.run_dir)
            if log_path is None:
                print(f"  {montage}/{condition}: no training_*.log found, cannot verify")
                all_match = False
                continue
            subject_lists[montage] = parse_val_subjects(log_path)

        if len(subject_lists) < 2:
            continue
        reference_montage, reference = next(iter(subject_lists.items()))
        for montage, subjects in subject_lists.items():
            if montage == reference_montage:
                continue
            if subjects != reference:
                print(
                    f"  MISMATCH [{condition}]: {reference_montage} vs {montage} val-subject "
                    "lists differ by fold -- pairing is broken, stop and investigate "
                    "(e.g. a CANE file raising NaNValuesError for one channel pair but not "
                    "another)."
                )
                all_match = False
            else:
                print(f"  OK [{condition}]: {reference_montage} == {montage} (all folds)")
    return all_match


def eye_movement_check(runs: dict[str, dict[str, RunMetrics]]) -> None:
    print("\n=== Eye-movement check for Fp2-Fp1 ===\n")
    ec_run = runs.get("Fp2-Fp1", {}).get("ec")
    eo_run = runs.get("Fp2-Fp1", {}).get("eo")
    if ec_run is None or eo_run is None:
        print("  Fp2-Fp1 EC and/or EO run missing, skipping.")
        return
    ec_mean = np.mean(ec_run.chunk_acc) * 100
    eo_mean = np.mean(eo_run.chunk_acc) * 100
    print(f"  Fp2-Fp1 EC chunk acc: {ec_mean:.2f}%, EO chunk acc: {eo_mean:.2f}%")
    if eo_mean - ec_mean > 2.0:
        print(
            "  EO notably higher than EC: consistent with horizontal eye movement "
            "dominating Fp2-Fp1 in eyes-open recordings, not pathology signal."
        )
    else:
        print("  No large EO-specific gain: less evidence of an eye-movement shortcut.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("experiments"), help="Directory containing run dirs."
    )
    parser.add_argument(
        "--check-folds",
        action="store_true",
        help="Only run the fold-identity check (do this before trusting paired stats).",
    )
    args = parser.parse_args()

    runs: dict[str, dict[str, RunMetrics]] = {}
    for montage, conditions in RUNS.items():
        runs[montage] = {}
        for condition, dir_name in conditions.items():
            run_dir = args.root / dir_name
            if not run_dir.exists():
                print(f"[skip] {run_dir} does not exist")
                continue
            try:
                runs[montage][condition] = load_run(run_dir)
            except FileNotFoundError as exc:
                print(f"[skip] {exc}")

    if args.check_folds:
        check_fold_identity(runs)
        return

    fold_identity_ok = check_fold_identity(runs)
    print_main_table(runs)
    print_per_dataset_table(runs)
    if fold_identity_ok:
        print_paired_stats(runs)
    else:
        print("\nSkipping paired statistics: fold-identity check failed above.")
    eye_movement_check(runs)


if __name__ == "__main__":
    main()
