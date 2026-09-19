"""Summarize the single-dataset CV runs (CNN-AttnS, T8-T7) behind objection 1.

Exp A (``all_040_*``) trained one model on MDD + CANE + SAD together and found that essentially
all of its gain over the dataset-prior baseline comes from MDD (see
``docs/plans/2026-09-19-explainability-research-notes.md`` section 3.4). The runs launched by
``run-single-dataset-meta.sh`` train the same model, with the same hyperparameters, on ONE
dataset at a time (``cane_041_t8-t7_{ec,eo}``, ``sad_041_...``, ``mdd_041_...``) to see whether
CANE/SAD signal exists when the model is not shared with MDD. MDD is the positive control: it
should stay well above its base rate.

For every present run this script prints

1. chunk and subject accuracy, chunk sensitivity and specificity (mean +/- std over folds, %).
   Chunk sensitivity/specificity come from the checkpoint's ``chunk`` metrics, which are correct
   (only the per-dataset ones were affected by the swapped-argument bug);
2. the majority-class baseline of the dataset, computed from the chunk confusion matrix, per fold
   (mean +/- std) and pooled, next to the pooled model accuracy and the gain over it;
3. a per-fold paired comparison of the model's chunk accuracy against that fold's own majority
   rate: mean difference, Wilcoxon signed-rank p and the Nadeau-Bengio corrected paired t-test.
   The statistics functions are imported from ``bipolar_ablation_summary.py`` (not copied), so
   both summaries always use the same procedure. Benjamini-Hochberg over the runs shown is
   applied to the Nadeau-Bengio p-values (the family is "all single-dataset comparisons");
4. when available, the per-dataset chunk accuracy of the combined-training runs
   ``all_040_t8-t7_{ec,eo}`` next to it. **Those runs use different folds** (folds of the pooled
   subject set), so this comparison is descriptive only.

Caveats: with 10 folds a fold holds only a handful of subjects (SAD: 4-6), so per-fold accuracy is
coarse and the Nadeau-Bengio correction (which assumes equal fold sizes) is only approximate.
"Above the majority rate" is a low bar for a pooled fold-wise test; read the mean difference and
the per-fold table (``--per-fold``) alongside the p-values.

Usage::

    poetry run python helpers-print/single_dataset_summary.py [--root experiments] [--per-fold]
"""

import argparse
import importlib.util
import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import numpy as np
import torch
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent


def _load_sibling(name: str) -> ModuleType:
    """Import a sibling script of ``helpers-print/`` (the directory is not a package).

    :param str name: Module file name without ``.py``.
    :return: The imported module (registered in ``sys.modules``).
    :rtype: ModuleType
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_bipolar = _load_sibling("bipolar_ablation_summary")
_prior = _load_sibling("dataset_prior_baseline")

# Shared with the bipolar ablation summary so both use the identical statistical procedure.
nadeau_bengio_ttest = _bipolar.nadeau_bengio_ttest
benjamini_hochberg = _bipolar.benjamini_hochberg
_fmt = _bipolar._fmt
# Loaded dynamically (helpers-print is not a package), so instances are typed as Any.
Counts: Any = _prior.Counts

logger = logging.getLogger("single_dataset_summary")

DATASETS = ["mdd", "cane", "sad"]
CONDITIONS = ["ec", "eo"]


@dataclass
class SingleRun:
    """Per-fold metrics of one single-dataset run.

    :param Path run_dir: Run directory.
    :param str dataset: Dataset the run was trained on (``mdd``/``cane``/``sad``).
    :param str condition: ``ec`` or ``eo``.
    :param list chunk_acc: Per-fold chunk accuracy.
    :param list chunk_sens: Per-fold chunk sensitivity.
    :param list chunk_spec: Per-fold chunk specificity.
    :param list subject_acc: Per-fold subject accuracy.
    :param list counts: Per-fold chunk confusion counts (label 1 = pathological).
    """

    run_dir: Path
    dataset: str
    condition: str
    chunk_acc: list[float] = field(default_factory=list)
    chunk_sens: list[float] = field(default_factory=list)
    chunk_spec: list[float] = field(default_factory=list)
    subject_acc: list[float] = field(default_factory=list)
    counts: list[Any] = field(default_factory=list)

    @property
    def majority(self) -> list[float]:
        """Per-fold majority-class rate of the validation chunks."""
        return [c.baseline for c in self.counts]

    @property
    def pooled(self) -> Any:
        """Confusion counts summed over folds."""
        return sum(self.counts, Counts())

    @property
    def n_folds(self) -> int:
        return len(self.chunk_acc)


def run_name(dataset: str, condition: str, version: str = "041", channel: str = "t8-t7") -> str:
    """Experiment directory name following the CLAUDE.md convention.

    :param str dataset: ``mdd``, ``cane``, ``sad`` or ``all``.
    :param str condition: ``ec`` or ``eo``.
    :param str version: Experiment version string.
    :param str channel: Lower-cased channel name.
    :return: ``<dataset>_<version>_<channel>_<condition>``.
    :rtype: str
    """
    return f"{dataset}_{version}_{channel}_{condition}"


def load_single_run(run_dir: Path, dataset: str, condition: str) -> SingleRun:
    """Load per-fold chunk/subject metrics of a single-dataset run from its checkpoints.

    :param Path run_dir: Directory with ``fold_N_best.pth`` files.
    :param str dataset: Dataset name the run was trained on.
    :param str condition: ``ec`` or ``eo``.
    :return: Per-fold metrics.
    :rtype: SingleRun
    :raises FileNotFoundError: If no checkpoints are found.
    """
    folds = _bipolar._fold_checkpoints(run_dir)
    if not folds:
        raise FileNotFoundError(f"No fold_N_best.pth files found in {run_dir}")

    run = SingleRun(run_dir=run_dir, dataset=dataset, condition=condition)
    for _, path in folds:
        val_metrics = torch.load(path, map_location="cpu", weights_only=False)["val_metrics"]
        chunk, subject = val_metrics["chunk"], val_metrics["subject"]
        cm = np.asarray(chunk["confusion_matrix"])
        if cm.shape != (2, 2):
            raise ValueError(f"{path}: expected a binary confusion matrix, got shape {cm.shape}")
        per_dataset = val_metrics.get("per_dataset") or {}
        foreign = sorted(set(per_dataset) - {dataset})
        if foreign:
            logger.warning(f"{path}: per_dataset contains {foreign}, expected only {dataset!r}")
        run.chunk_acc.append(float(chunk["accuracy"]))
        run.chunk_sens.append(float(chunk["recall"]))
        run.chunk_spec.append(float(chunk["specificity"]))
        run.subject_acc.append(float(subject["accuracy"]))
        run.counts.append(
            Counts(tp=int(cm[1, 1]), tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]))
        )
    return run


@dataclass
class PairedResult:
    """Model-vs-majority paired comparison for one run.

    :param float mean_diff_pp: Mean per-fold (accuracy - majority rate) in percentage points.
    :param float wilcoxon_p: Two-sided Wilcoxon signed-rank p-value (NaN if undefined).
    :param float nb_t: Nadeau-Bengio corrected t statistic.
    :param float nb_p: Two-sided Nadeau-Bengio p-value.
    :param int n_above: Number of folds where accuracy exceeds the majority rate.
    """

    mean_diff_pp: float
    wilcoxon_p: float
    nb_t: float
    nb_p: float
    n_above: int


def paired_vs_majority(run: SingleRun) -> PairedResult:
    """Paired per-fold comparison of model accuracy against the per-fold majority rate.

    Uses the same Wilcoxon call and Nadeau-Bengio helper as ``bipolar_ablation_summary.py``.
    Both tests need at least 2 folds (Nadeau-Bengio's variance correction divides by
    ``n_train = folds - 1``, so a single fold gives a ``ZeroDivisionError``); with fewer, the
    statistical test is skipped (NaN) and a warning is logged, but the mean difference and
    ``n_above`` are still meaningful and computed.

    :param SingleRun run: Run to test.
    :return: Test results (``wilcoxon_p``/``nb_t``/``nb_p`` are NaN if ``run.n_folds < 2``).
    :rtype: PairedResult
    """
    diffs = np.asarray(run.chunk_acc) - np.asarray(run.majority)
    if run.n_folds < 2:
        logger.warning(
            f"{run.run_dir.name}: only {run.n_folds} fold(s), skipping the paired significance "
            "test (Wilcoxon/Nadeau-Bengio need at least 2 folds)"
        )
        return PairedResult(
            mean_diff_pp=float(diffs.mean() * 100),
            wilcoxon_p=float("nan"),
            nb_t=float("nan"),
            nb_p=float("nan"),
            n_above=int((diffs > 0).sum()),
        )
    try:
        with warnings.catch_warnings():
            # scipy warns (and returns p = 1) when every difference is exactly zero.
            warnings.simplefilter("ignore", RuntimeWarning)
            wilcoxon_p = float(wilcoxon(diffs)[1])
    except ValueError as exc:
        logger.warning(f"{run.run_dir.name}: Wilcoxon failed ({exc})")
        wilcoxon_p = float("nan")
    t_stat, nb_p = nadeau_bengio_ttest(diffs, n_train=len(diffs) - 1, n_test=1)
    return PairedResult(
        mean_diff_pp=float(diffs.mean() * 100),
        wilcoxon_p=wilcoxon_p,
        nb_t=float(t_stat),
        nb_p=float(nb_p),
        n_above=int((diffs > 0).sum()),
    )


def print_main_table(runs: dict[tuple[str, str], SingleRun]) -> None:
    logger.info(
        "\n=== Single-dataset runs: accuracy, sensitivity, specificity (%, mean +/- std) ===\n"
    )
    header = (
        f"{'Dataset':<8} {'Cond':<4} {'Folds':>5} {'Chunk Acc':>14} {'Subj Acc':>14} "
        f"{'Chunk Sens':>14} {'Chunk Spec':>14}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for (dataset, condition), run in runs.items():
        logger.info(
            f"{dataset.upper():<8} {condition:<4} {run.n_folds:>5} "
            f"{_fmt(run.chunk_acc):>14} {_fmt(run.subject_acc):>14} "
            f"{_fmt(run.chunk_sens):>14} {_fmt(run.chunk_spec):>14}"
        )


def print_baseline_table(runs: dict[tuple[str, str], SingleRun]) -> None:
    logger.info(
        "\n=== Majority-class baseline of each dataset (from the confusion matrix, %) ===\n"
    )
    header = (
        f"{'Dataset':<8} {'Cond':<4} {'Path%':>6} {'Base (pooled)':>13} {'Base / fold':>14} "
        f"{'Acc (pooled)':>12} {'Gain pp':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for (dataset, condition), run in runs.items():
        pooled = run.pooled
        base, acc = pooled.baseline, pooled.accuracy
        path_pct = pooled.positives / pooled.total * 100
        logger.info(
            f"{dataset.upper():<8} {condition:<4} {path_pct:>6.1f} "
            f"{base * 100:>13.1f} {_fmt(run.majority):>14} {acc * 100:>12.1f} "
            f"{(acc - base) * 100:>+8.1f}"
        )


def print_paired_table(runs: dict[tuple[str, str], SingleRun], per_fold: bool = False) -> None:
    logger.info("\n=== Per-fold paired test: chunk accuracy vs that fold's majority rate ===\n")
    logger.info(
        "Wilcoxon signed-rank and the Nadeau-Bengio corrected t-test, as in "
        "bipolar_ablation_summary.py (equal fold sizes assumed: n_test/n_train = 1/(k-1)).\n"
        "BH is applied over the runs listed, to the Nadeau-Bengio p-values.\n"
        "Folds are small (SAD: 4-6 subjects), so per-fold accuracy is coarse.\n"
    )
    keys = list(runs)
    results = {key: paired_vs_majority(runs[key]) for key in keys}
    if not results:
        return
    # benjamini_hochberg sorts its input with np.argsort, which puts NaN last but does not drop
    # it; np.minimum.accumulate then propagates that NaN backwards into every other adjusted
    # p-value. Runs with too few folds for a significance test (nb_p NaN, see paired_vs_majority)
    # are therefore excluded from the correction, not just displayed as "n/a".
    testable = [k for k in keys if not np.isnan(results[k].nb_p)]
    bh_by_key = dict(zip(testable, benjamini_hochberg([results[k].nb_p for k in testable])))
    adjusted = [bh_by_key.get(k, float("nan")) for k in keys]
    header = (
        f"{'Dataset':<8} {'Cond':<4} {'Mean diff':>10} {'Folds >':>8} {'Wilcoxon p':>11} "
        f"{'NB t':>8} {'NB p':>8} {'BH p':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for key, adj in zip(keys, adjusted):
        r = results[key]
        wilcoxon_p = "n/a".rjust(11) if np.isnan(r.wilcoxon_p) else f"{r.wilcoxon_p:>11.4f}"
        nb_t = "n/a".rjust(8) if np.isnan(r.nb_t) else f"{r.nb_t:>+8.3f}"
        nb_p = "n/a".rjust(8) if np.isnan(r.nb_p) else f"{r.nb_p:>8.4f}"
        bh_p = "n/a".rjust(8) if np.isnan(adj) else f"{adj:>8.4f}"
        logger.info(
            f"{key[0].upper():<8} {key[1]:<4} {r.mean_diff_pp:>+8.2f}pp "
            f"{r.n_above:>3}/{runs[key].n_folds:<4} {wilcoxon_p} {nb_t} {nb_p} {bh_p}"
        )
    if per_fold:
        logger.info("\nPer fold (accuracy / majority rate, %):")
        for key in keys:
            run = runs[key]
            cells = " ".join(
                f"{a * 100:5.1f}/{m * 100:4.1f}" for a, m in zip(run.chunk_acc, run.majority)
            )
            logger.info(f"  {key[0].upper():<5} {key[1]}: {cells}")


def print_combined_comparison(runs: dict[tuple[str, str], SingleRun], combined_root: Path) -> None:
    """Print single-dataset vs combined-training per-dataset accuracy (descriptive only).

    :param dict runs: Single-dataset runs by ``(dataset, condition)``.
    :param Path combined_root: Directory containing ``all_040_t8-t7_{ec,eo}``.
    """
    logger.info(
        "\n=== Single-dataset vs combined training (all_040), per-dataset chunk accuracy ===\n"
    )
    logger.info(
        "DESCRIPTIVE ONLY: the combined runs use different folds (folds of the pooled MDD+CANE+SAD "
        "subject set), so no paired test is possible. Combined per-dataset counts have their "
        "fp/fn orientation resolved by dataset_prior_baseline.py.\n"
    )
    header = (
        f"{'Dataset':<8} {'Cond':<4} | {'single acc':>10} {'base':>6} {'gain':>7} | "
        f"{'combined acc':>12} {'base':>6} {'gain':>7}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    any_row = False
    for condition in CONDITIONS:
        combined_dir = combined_root / run_name("all", condition, "040")
        combined: dict[str, Any] = {}
        if combined_dir.exists():
            try:
                combined = _prior.load_run(combined_dir).counts
            except (_prior.SkipRunError, _prior.CountMismatchError) as exc:
                logger.warning(f"{combined_dir.name}: {exc}")
        else:
            logger.info(f"  (no {combined_dir} - combined column omitted for {condition})")
        for dataset in DATASETS:
            single = runs.get((dataset, condition))
            comb = combined.get(dataset)
            if single is None and comb is None:
                continue
            any_row = True
            if single is not None:
                p = single.pooled
                s_acc, s_base = p.accuracy * 100, p.baseline * 100
                s_cells = f"{s_acc:>10.1f} {s_base:>6.1f} {s_acc - s_base:>+7.1f}"
            else:
                s_cells = f"{'n/a':>10} {'':>6} {'':>7}"
            if comb is not None:
                c_acc, c_base = comb.accuracy * 100, comb.baseline * 100
                c_cells = f"{c_acc:>12.1f} {c_base:>6.1f} {c_acc - c_base:>+7.1f}"
            else:
                c_cells = f"{'n/a':>12} {'':>6} {'':>7}"
            logger.info(f"{dataset.upper():<8} {condition:<4} | {s_cells} | {c_cells}")
    if not any_row:
        logger.info("  (nothing to compare)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, default=Path("experiments"), help="Directory containing run dirs."
    )
    parser.add_argument(
        "--combined-root",
        type=Path,
        default=None,
        help="Directory containing the combined all_040_t8-t7_{ec,eo} runs (default: --root).",
    )
    parser.add_argument("--version", default="041", help="Experiment version of the single runs.")
    parser.add_argument("--per-fold", action="store_true", help="Also print fold-by-fold values.")
    return parser


def main(argv: Optional[list[str]] = None, configure_logging: bool = True) -> int:
    """Run the script.

    :param Optional[list[str]] argv: Command-line arguments (``sys.argv[1:]`` if None).
    :param bool configure_logging: Install the plain stdout report handler (tests pass False).
    :return: Process exit code.
    :rtype: int
    """
    args = build_parser().parse_args(argv)
    if configure_logging:
        _prior.setup_report_logging()

    runs: dict[tuple[str, str], SingleRun] = {}
    for dataset in DATASETS:
        for condition in CONDITIONS:
            run_dir = args.root / run_name(dataset, condition, args.version)
            if not run_dir.exists():
                logger.warning(f"[skip] {run_dir} does not exist")
                continue
            try:
                run = load_single_run(run_dir, dataset, condition)
            except FileNotFoundError as exc:
                logger.warning(f"[skip] {exc}")
                continue
            if run.n_folds != 10:
                logger.warning(f"{run_dir.name}: {run.n_folds} folds found, expected 10")
            runs[(dataset, condition)] = run

    if not runs:
        logger.error("no single-dataset runs found")
        return 1
    print_main_table(runs)
    print_baseline_table(runs)
    print_paired_table(runs, per_fold=args.per_fold)
    print_combined_comparison(runs, args.combined_root or args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
