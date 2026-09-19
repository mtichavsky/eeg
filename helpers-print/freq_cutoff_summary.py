"""Summarize the frequency-cutoff ablation (Experiment B: 70 Hz vs 30 Hz spectrogram crop).

Reads the four run directories produced by ``run-freq-cutoff-meta.sh`` (see
docs/plans/2026-09-14-frequency-cutoff-ablation.md) and prints, per model
(CNNAttnS in-ear, AllTransformerV4 8-channel):

1. Run provenance: git commit, ``freq_cutoff`` from ``results.txt`` and from the checkpoints,
   completed folds.
2. Main table: 70 Hz vs 30 Hz chunk/subject accuracy, sensitivity, specificity as mean +/- std
   over folds, plus the parameter count read from the checkpoints.
3. Fold-identity check: the "Val subjects (N): [...]" log lines of the f70 and f30 run of a model
   must match fold by fold. If they differ, the paired statistics for that model are skipped.
4. Paired per-fold differences (f70 minus f30): mean, bootstrap 95% CI, Wilcoxon signed-rank and
   the Nadeau-Bengio corrected t-test, Benjamini-Hochberg-adjusted across the models.
5. Per-dataset (MDD / CANE / SAD) chunk ACCURACY only. Per-dataset sensitivity/specificity are
   deliberately not used: they are mislabelled by a known bug (stored fp/fn are swapped).
   In the in-ear runs the "CANE" column is IDUN real in-ear data.
6. Reproducibility: the 70 Hz reruns next to the Table II numbers (descriptive).

Usage:
    poetry run python helpers-print/freq_cutoff_summary.py [--root experiments] [--check-folds]
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.stats import wilcoxon

# ``helpers-print`` is not a package (hyphen), so make the sibling module importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bipolar_ablation_summary import (  # noqa: E402
    _find_training_log,
    _fmt,
    _fold_checkpoints,
    benjamini_hochberg,
    nadeau_bengio_ttest,
    parse_val_subjects,
)

CUTOFFS = (70, 30)

# Display name -> {cutoff_hz -> run directory name}, matching the plan's run table.
RUNS: dict[str, dict[int, str]] = {
    "CNNAttnS (in-ear)": {
        70: "all_041_inear_ec+eo_f70",
        30: "all_041_inear_ec+eo_f30",
    },
    "AllTransformerV4 (8-channel)": {
        70: "all_041_all_ec+eo_f70",
        30: "all_041_all_ec+eo_f30",
    },
}

# Table II reference numbers (percent): (chunk accuracy, subject accuracy), 70 Hz, ec+eo.
TABLE_II: dict[str, tuple[float, float]] = {
    "CNNAttnS (in-ear)": (77.0, 79.2),
    "AllTransformerV4 (8-channel)": (76.5, 78.6),
}

DATASETS = ["mdd", "cane", "sad"]
EXPECTED_FOLDS = 10
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0

# (label, attribute on CutoffRun) for the metrics compared pairwise.
PAIRED_METRICS: list[tuple[str, str]] = [
    ("Chunk Acc", "chunk_acc"),
    ("Subj Acc", "subject_acc"),
    ("Chunk Sens", "chunk_sens"),
    ("Chunk Spec", "chunk_spec"),
]

_CONFIG_LINE_RE = re.compile(r"^\s{2}([A-Za-z0-9_]+): (.*)$")


@dataclass
class CutoffRun:
    """Per-fold metrics and provenance of one training run."""

    run_dir: Path
    n_folds: int
    chunk_acc: list[float] = field(default_factory=list)
    chunk_sens: list[float] = field(default_factory=list)
    chunk_spec: list[float] = field(default_factory=list)
    subject_acc: list[float] = field(default_factory=list)
    subject_sens: list[float] = field(default_factory=list)
    subject_spec: list[float] = field(default_factory=list)
    # per_dataset[dataset] = chunk accuracy per fold (fold order).
    per_dataset: dict[str, list[float]] = field(default_factory=dict)
    param_count: Optional[int] = None
    # freq_cutoff_hz stored in each checkpoint (None when a checkpoint has no such key).
    ckpt_cutoffs: list[Optional[float]] = field(default_factory=list)
    # Key -> value strings from the "Configuration:" block of results.txt.
    config: dict[str, str] = field(default_factory=dict)


@dataclass
class PairedResult:
    """Paired f70-minus-f30 comparison of one metric for one model."""

    mean_diff: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float
    t_stat: float
    nb_p: float


def count_parameters(state_dict: dict[str, torch.Tensor]) -> int:
    """Count the elements of all tensors in a ``state_dict``.

    The spectrogram models of this project have no buffers (no BatchNorm running statistics),
    so this equals ``sum(p.numel() for p in model.parameters())``; verified for CNNAttnS and
    AllTransformerV4 at both cutoffs.

    :param dict state_dict: The ``model_state_dict`` of a checkpoint.
    :return: Total number of elements.
    :rtype: int
    """
    return sum(int(tensor.numel()) for tensor in state_dict.values())


def parse_results_config(results_path: Path) -> dict[str, str]:
    """Parse the ``Configuration:`` block of a ``results.txt`` into a key -> value dict.

    :param Path results_path: Path to ``results.txt``.
    :return: Configuration values as strings; empty if the file is missing.
    :rtype: dict[str, str]
    """
    if not results_path.exists():
        return {}
    config: dict[str, str] = {}
    in_config = False
    for line in results_path.read_text().splitlines():
        if line.startswith("Configuration:"):
            in_config = True
            continue
        if not in_config:
            continue
        match = _CONFIG_LINE_RE.match(line)
        if match:
            config[match.group(1)] = match.group(2).strip()
        elif line.strip() == "":
            break
    return config


def load_run(run_dir: Path) -> CutoffRun:
    """Load per-fold metrics, parameter count and provenance from a run directory.

    :param Path run_dir: Directory holding ``fold_N_best.pth`` and ``results.txt``.
    :return: The loaded run.
    :rtype: CutoffRun
    :raises FileNotFoundError: If the directory holds no ``fold_N_best.pth``.
    """
    folds = _fold_checkpoints(run_dir)
    if not folds:
        raise FileNotFoundError(f"No fold_N_best.pth files found in {run_dir}")

    run = CutoffRun(run_dir=run_dir, n_folds=len(folds))
    run.config = parse_results_config(run_dir / "results.txt")
    for _, path in folds:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        val_metrics = ckpt["val_metrics"]
        chunk, subject = val_metrics["chunk"], val_metrics["subject"]

        run.chunk_acc.append(float(chunk["accuracy"]))
        run.chunk_sens.append(float(chunk["recall"]))
        run.chunk_spec.append(float(chunk["specificity"]))
        run.subject_acc.append(float(subject["accuracy"]))
        run.subject_sens.append(float(subject["recall"]))
        run.subject_spec.append(float(subject["specificity"]))

        for dataset_name, dataset_metrics in (val_metrics.get("per_dataset") or {}).items():
            run.per_dataset.setdefault(dataset_name, []).append(float(dataset_metrics["accuracy"]))

        cutoff = ckpt.get("freq_cutoff_hz")
        run.ckpt_cutoffs.append(None if cutoff is None else float(cutoff))
        if run.param_count is None:
            run.param_count = count_parameters(ckpt["model_state_dict"])
    return run


def bootstrap_ci(diffs: np.ndarray, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean of per-fold differences.

    Folds share training data, so this interval is optimistic; it is an effect-size summary,
    not a substitute for the corrected test.

    :param np.ndarray diffs: Per-fold differences.
    :param int seed: RNG seed, fixed for reproducible output.
    :return: (lower, upper) bounds of the 95% interval.
    :rtype: tuple[float, float]
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(BOOTSTRAP_RESAMPLES, len(diffs)))
    means = diffs[idx].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_comparison(values_70: list[float], values_30: list[float]) -> PairedResult:
    """Paired f70-minus-f30 statistics over folds.

    :param list values_70: Per-fold metric at the 70 Hz cutoff (fractions).
    :param list values_30: Per-fold metric at the 30 Hz cutoff, same fold order (fractions).
    :return: Mean difference and CI (fractions), Wilcoxon p, Nadeau-Bengio t and p.
    :rtype: PairedResult
    :raises ValueError: If the fold counts differ.
    """
    if len(values_70) != len(values_30):
        raise ValueError(f"fold count mismatch: {len(values_70)} vs {len(values_30)}")
    diffs = np.asarray(values_70) - np.asarray(values_30)
    if np.all(diffs == 0):
        # scipy warns and returns p = 1.0 here; short-circuit to keep the output clean.
        wilcoxon_p = 1.0
    else:
        wilcoxon_p = float(wilcoxon(diffs).pvalue)
    # Same convention as the bipolar ablation: n_test / n_train = 1 / (k - 1) for k-fold CV.
    t_stat, nb_p = nadeau_bengio_ttest(diffs, n_train=len(diffs) - 1, n_test=1)
    ci_low, ci_high = bootstrap_ci(diffs)
    return PairedResult(float(diffs.mean()), ci_low, ci_high, wilcoxon_p, t_stat, nb_p)


def check_fold_identity(run_70: CutoffRun, run_30: CutoffRun) -> bool:
    """Confirm two runs saw the same validation subjects, fold by fold.

    :param CutoffRun run_70: The 70 Hz run.
    :param CutoffRun run_30: The 30 Hz run.
    :return: True if both logs are found, non-empty and identical fold by fold.
    :rtype: bool
    """
    subject_lists: list[list[list[str]]] = []
    for run in (run_70, run_30):
        log_path = _find_training_log(run.run_dir)
        if log_path is None:
            print(f"  {run.run_dir.name}: no training_*.log found, cannot verify")
            return False
        subject_lists.append(parse_val_subjects(log_path))
    subjects_70, subjects_30 = subject_lists
    if not subjects_70 or not subjects_30:
        print("  no 'Val subjects' lines found in one of the logs, cannot verify")
        return False
    if subjects_70 != subjects_30:
        print(
            f"  MISMATCH: {run_70.run_dir.name} ({len(subjects_70)} folds) vs "
            f"{run_30.run_dir.name} ({len(subjects_30)} folds) val-subject lists differ -- "
            "pairing is broken"
        )
        return False
    print(f"  OK: {run_70.run_dir.name} == {run_30.run_dir.name} ({len(subjects_70)} folds)")
    return True


def _cfg(run: CutoffRun, key: str) -> str:
    return run.config.get(key, "n/a")


def print_provenance(runs: dict[str, dict[int, CutoffRun]]) -> None:
    print("\n=== Run provenance ===\n")
    header = (
        f"{'Model':<30} {'Cut':>4} {'Folds':>5} {'results.txt freq_cutoff':>24} "
        f"{'ckpt freq_cutoff_hz':>20} {'git_commit':>16}"
    )
    print(header)
    print("-" * len(header))
    for model, by_cutoff in runs.items():
        for cutoff in CUTOFFS:
            run = by_cutoff.get(cutoff)
            if run is None:
                continue
            ckpt_values = sorted({c for c in run.ckpt_cutoffs}, key=lambda c: (c is None, c))
            ckpt_str = ",".join("missing" if c is None else f"{c:g}" for c in ckpt_values)
            print(
                f"{model:<30} {cutoff:>4} {run.n_folds:>5} {_cfg(run, 'freq_cutoff'):>24} "
                f"{ckpt_str:>20} {_cfg(run, 'git_commit'):>16}"
            )


def print_main_table(runs: dict[str, dict[int, CutoffRun]]) -> None:
    print("\n=== Main table: chunk/subject accuracy, sensitivity, specificity (%) ===\n")
    header = (
        f"{'Model':<30} {'Cut':>4} {'Chunk Acc':>14} {'Chunk Sens':>14} {'Chunk Spec':>14} "
        f"{'Subj Acc':>14} {'Subj Sens':>14} {'Subj Spec':>14} {'Params':>9}"
    )
    print(header)
    print("-" * len(header))
    for model, by_cutoff in runs.items():
        for cutoff in CUTOFFS:
            run = by_cutoff.get(cutoff)
            if run is None:
                continue
            params = "n/a" if run.param_count is None else f"{run.param_count:,}"
            print(
                f"{model:<30} {cutoff:>4} {_fmt(run.chunk_acc):>14} {_fmt(run.chunk_sens):>14} "
                f"{_fmt(run.chunk_spec):>14} {_fmt(run.subject_acc):>14} "
                f"{_fmt(run.subject_sens):>14} {_fmt(run.subject_spec):>14} {params:>9}"
            )
    print("\nStd is over folds (ddof=1). Parameter counts are read from the checkpoints.")


def print_paired_stats(runs: dict[str, dict[int, CutoffRun]], fold_ok: dict[str, bool]) -> None:
    print("\n=== Paired per-fold differences (70 Hz minus 30 Hz) ===\n")
    print(
        "Positive = 70 Hz better. CI is a percentile bootstrap over folds (optimistic: CV folds "
        "share training data). Wilcoxon signed-rank is primary; NB = Nadeau-Bengio corrected "
        f"t-test with n_test/n_train = 1/{EXPECTED_FOLDS - 1}. BH adjusts across the models "
        "within each metric.\n"
    )
    for label, attr in PAIRED_METRICS:
        print(f"-- {label} --")
        results: list[tuple[str, PairedResult]] = []
        for model, by_cutoff in runs.items():
            run_70, run_30 = by_cutoff.get(70), by_cutoff.get(30)
            if run_70 is None or run_30 is None:
                continue
            if not fold_ok.get(model, False):
                print(f"  {model}: SKIPPED (fold-identity check failed)")
                continue
            try:
                results.append(
                    (model, paired_comparison(getattr(run_70, attr), getattr(run_30, attr)))
                )
            except ValueError as exc:
                print(f"  {model}: SKIPPED ({exc})")
        if not results:
            continue
        adj_w = benjamini_hochberg(
            [0.0 if np.isnan(r.wilcoxon_p) else r.wilcoxon_p for _, r in results]
        )
        adj_nb = benjamini_hochberg([r.nb_p for _, r in results])
        for (model, res), w_adj, nb_adj in zip(results, adj_w, adj_nb):
            print(
                f"  {model:<30} diff = {res.mean_diff * 100:+6.2f}pp "
                f"[{res.ci_low * 100:+6.2f}, {res.ci_high * 100:+6.2f}]  "
                f"Wilcoxon p = {res.wilcoxon_p:.4f} (BH {w_adj:.4f})  "
                f"NB t = {res.t_stat:+.3f} p = {res.nb_p:.4f} (BH {nb_adj:.4f})"
            )
        print()


def print_per_dataset_table(runs: dict[str, dict[int, CutoffRun]]) -> None:
    print("\n=== Per-dataset chunk accuracy (%) ===\n")
    header = f"{'Model':<30} {'Cut':>4} " + " ".join(f"{d.upper():>14}" for d in DATASETS)
    print(header)
    print("-" * len(header))
    for model, by_cutoff in runs.items():
        for cutoff in CUTOFFS:
            run = by_cutoff.get(cutoff)
            if run is None:
                continue
            cells = [_fmt(run.per_dataset[d]) if d in run.per_dataset else "n/a" for d in DATASETS]
            print(f"{model:<30} {cutoff:>4} " + " ".join(f"{c:>14}" for c in cells))
        run_70, run_30 = by_cutoff.get(70), by_cutoff.get(30)
        if run_70 is not None and run_30 is not None:
            diffs = []
            for d in DATASETS:
                a, b = run_70.per_dataset.get(d), run_30.per_dataset.get(d)
                if a is not None and b is not None and len(a) == len(b):
                    diffs.append(f"{(np.mean(a) - np.mean(b)) * 100:+.2f}pp")
                else:
                    diffs.append("n/a")
            print(f"{'  70 minus 30 (mean)':<35} " + " ".join(f"{c:>14}" for c in diffs))
    print(
        "\nPer-dataset sensitivity/specificity are intentionally not reported (stored fp/fn are "
        "swapped by a known bug).\nIn the in-ear runs the CANE column is IDUN real in-ear data."
    )


def print_reproducibility(runs: dict[str, dict[int, CutoffRun]]) -> None:
    print("\n=== Reproducibility: 70 Hz rerun vs Table II (ec+eo, %) ===\n")
    header = (
        f"{'Model':<30} {'Table II chunk':>15} {'rerun chunk':>15} {'Table II subj':>15} "
        f"{'rerun subj':>15}"
    )
    print(header)
    print("-" * len(header))
    for model, by_cutoff in runs.items():
        run_70 = by_cutoff.get(70)
        if run_70 is None or model not in TABLE_II:
            continue
        ref_chunk, ref_subj = TABLE_II[model]
        rerun_chunk = float(np.mean(run_70.chunk_acc)) * 100
        rerun_subj = float(np.mean(run_70.subject_acc)) * 100
        print(
            f"{model:<30} {ref_chunk:>15.1f} {rerun_chunk:>9.2f} ({rerun_chunk - ref_chunk:+.2f}) "
            f"{ref_subj:>9.1f} {rerun_subj:>9.2f} ({rerun_subj - ref_subj:+.2f})"
        )
    print(
        "\nTable II was trained at commit 454c49e-dirty; torch is not seeded, so a gap is "
        "run-to-run noise plus code drift. Descriptive only."
    )


def load_all(root: Path) -> dict[str, dict[int, CutoffRun]]:
    """Load every run of :data:`RUNS` that exists under ``root``.

    :param Path root: Directory containing the run directories.
    :return: Loaded runs per model and cutoff; missing runs are skipped with a message.
    :rtype: dict[str, dict[int, CutoffRun]]
    """
    runs: dict[str, dict[int, CutoffRun]] = {}
    for model, by_cutoff in RUNS.items():
        runs[model] = {}
        for cutoff, dir_name in by_cutoff.items():
            run_dir = root / dir_name
            if not run_dir.exists():
                print(f"[skip] {run_dir} does not exist")
                continue
            try:
                run = load_run(run_dir)
            except FileNotFoundError as exc:
                print(f"[skip] {exc}")
                continue
            if run.n_folds != EXPECTED_FOLDS:
                print(
                    f"[warn] {run_dir.name}: {run.n_folds} folds found, expected {EXPECTED_FOLDS}"
                )
            runs[model][cutoff] = run
    return runs


def check_all_fold_identity(runs: dict[str, dict[int, CutoffRun]]) -> dict[str, bool]:
    """Run the fold-identity check for each model that has both cutoffs.

    :param dict runs: Loaded runs.
    :return: Per-model pass/fail; models lacking a run are absent.
    :rtype: dict[str, bool]
    """
    print("\n=== Fold-identity check (f70 vs f30 val subjects) ===\n")
    fold_ok: dict[str, bool] = {}
    for model, by_cutoff in runs.items():
        run_70, run_30 = by_cutoff.get(70), by_cutoff.get(30)
        if run_70 is None or run_30 is None:
            continue
        print(f"{model}:")
        fold_ok[model] = check_fold_identity(run_70, run_30)
    return fold_ok


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

    runs = load_all(args.root)
    fold_ok = check_all_fold_identity(runs)
    if args.check_folds:
        return

    print_provenance(runs)
    print_main_table(runs)
    print_paired_stats(runs, fold_ok)
    print_per_dataset_table(runs)
    print_reproducibility(runs)


if __name__ == "__main__":
    main()
