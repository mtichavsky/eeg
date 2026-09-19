"""Dataset-prior baseline: how much of a model's accuracy is just "know the dataset".

Objection 1 in ``docs/plans/2026-09-19-explainability-research-notes.md`` says a model trained on
pooled MDD + CANE + SAD data may learn *which dataset a chunk came from* rather than pathology.
The **dataset-prior baseline** is a classifier that only knows the source dataset of each chunk
and always predicts that dataset's majority class. It uses the validation class balance, so it is
if anything generous to the baseline. For every run this script prints

* per-dataset chunk accuracy (pooled over folds) next to that dataset's majority-class rate,
* the overall prior baseline (each dataset's majority rate, weighted by its chunk count),
* the gain of the model over that baseline, and how many percentage points of the gain come from
  each dataset (the contributions sum to the gain).

Data source
-----------
Each ``fold_N_best.pth`` stores ``val_metrics`` with ``chunk`` (including ``confusion_matrix``,
rows = actual, ``[[TN, FP], [FN, TP]]``) and ``per_dataset`` (``accuracy``, ``tp``, ``tn``,
``fp``, ``fn``, ``correct``, ``total`` per dataset).

Known bug and how this script copes (notes section 3.5)
------------------------------------------------------
Before the fix, ``main.py`` called ``compute_per_dataset_metrics(preds, labels, ...)`` with the
arguments swapped, so the stored per-dataset ``fp`` is really the true FN and the stored ``fn`` is
really the true FP (``tp``, ``tn`` and ``accuracy`` are unaffected). The orientation is detected
per fold: the per-dataset ``(tp, tn, fp, fn)`` summed over datasets is compared with the chunk
confusion matrix of the same checkpoint. If it equals ``(TP, TN, FN, FP)`` the counts are swapped
back, if it equals ``(TP, TN, FP, FN)`` they are used as stored, otherwise the script raises. When
``FP == FN`` for a fold both readings fit, so that fold takes the orientation of the run's other
folds (``--assume`` overrides this). Which case was detected is logged.

4-class runs
------------
For 4-class checkpoints the per-dataset ``tp/tn/fp/fn`` are not meaningful and the class balance of
a dataset is not stored, so only per-dataset accuracy is printed and no baseline is computed.

``--from-results-txt`` (best effort, no checkpoints needed)
-----------------------------------------------------------
Runs whose checkpoints were not kept can still be analysed from the aggregated "Per-Dataset Chunk
Metrics" table of ``results.txt`` (columns accuracy / sensitivity / specificity / chunks). The
table is built from counts pooled over folds, so with ``N`` chunks the four equations

    TP + TN + FP + FN = N,   TP + TN = acc * N,   sens = ..., spec = ...

determine the four counts. What "sens" and "spec" mean depends on the code version:

* ``--results-orientation swapped`` (default; every run that predates the fix): they are really
  ``TP / (TP + FP)`` (PPV) and ``TN / (TN + FN)`` (NPV).
* ``--results-orientation fixed``: they are the true ``TP / (TP + FN)`` and ``TN / (TN + FP)``.

The percentages are printed with two decimals, so the reconstructed counts carry a rounding error
(a few chunks per dataset for typical sizes). The reconstruction is validated against the
aggregated confusion matrix in the same file (sum of the reconstructed counts vs TP/TN/FP/FN) and
against a checkpoint run in ``tests/test_dataset_prior_baseline.py`` and in the PR description.

Usage::

    poetry run python helpers-print/dataset_prior_baseline.py \\
        --root /home/milan/eeg/code/experiments 'all_040_*'
    poetry run python helpers-print/dataset_prior_baseline.py \\
        --root /home/milan/eeg/experiments 'binary/*/*' '4class/*/*'
    poetry run python helpers-print/dataset_prior_baseline.py --from-results-txt \\
        /home/milan/eeg/experiments/results.txt
"""

import argparse
import glob
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

logger = logging.getLogger("dataset_prior_baseline")

FOLD_FILE_RE = re.compile(r"fold_(\d+)_best\.pth$")
DATASET_ORDER = ["mdd", "cane", "sad"]

Orientation = Literal["unswapped", "swapped", "ambiguous"]


class PlainFormatter(logging.Formatter):
    """Print INFO records as bare messages (report output) and prefix everything else."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return message if record.levelno <= logging.INFO else f"{record.levelname}: {message}"


def setup_report_logging() -> None:
    """Route the report through ``logging`` to stdout with the plain formatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(PlainFormatter("%(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)


class CountMismatchError(ValueError):
    """The per-dataset counts of a checkpoint are inconsistent with its chunk confusion matrix."""


class SkipRunError(Exception):
    """A run directory cannot be analysed (no checkpoints, or checkpoints lack per_dataset)."""


@dataclass(frozen=True)
class Counts:
    """Binary confusion counts with label 1 = pathological (positive)."""

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: "Counts") -> "Counts":
        return Counts(
            self.tp + other.tp, self.tn + other.tn, self.fp + other.fp, self.fn + other.fn
        )

    @property
    def total(self) -> int:
        return self.tp + self.tn + self.fp + self.fn

    @property
    def correct(self) -> int:
        return self.tp + self.tn

    @property
    def positives(self) -> int:
        """Number of truly pathological chunks."""
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        """Number of truly healthy chunks."""
        return self.tn + self.fp

    @property
    def majority_correct(self) -> int:
        """Chunks a majority-class predictor gets right."""
        return max(self.positives, self.negatives)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    @property
    def baseline(self) -> float:
        """Accuracy of the majority-class predictor (dataset prior)."""
        return self.majority_correct / self.total

    @property
    def sensitivity(self) -> float:
        return self.tp / self.positives if self.positives else float("nan")

    @property
    def specificity(self) -> float:
        return self.tn / self.negatives if self.negatives else float("nan")

    def swapped(self) -> "Counts":
        """Return the counts with ``fp`` and ``fn`` exchanged."""
        return Counts(self.tp, self.tn, self.fn, self.fp)


@dataclass
class FoldRecord:
    """Oriented per-dataset numbers of one fold checkpoint.

    :param int fold: 1-based fold number.
    :param dict correct: Per-dataset correct chunk counts.
    :param dict total: Per-dataset chunk counts.
    :param dict counts: Per-dataset oriented binary counts (empty for multi-class runs).
    :param str orientation: Detected orientation of the stored per-dataset fp/fn.
    """

    fold: int
    correct: dict[str, int]
    total: dict[str, int]
    counts: dict[str, Counts] = field(default_factory=dict)
    orientation: str = "n/a"


@dataclass
class RunCounts:
    """Per-dataset counts of one run, pooled over folds.

    :param Path path: Run directory (or results.txt when reconstructed from text).
    :param str source: ``"checkpoints"`` or ``"results.txt"``.
    :param dict correct: Pooled per-dataset correct chunk counts.
    :param dict total: Pooled per-dataset chunk counts.
    :param dict counts: Pooled oriented binary counts (empty for multi-class runs).
    :param list folds: Per-fold records (empty when reconstructed from results.txt).
    :param str note: Human-readable description of the orientation handling.
    """

    path: Path
    source: str
    correct: dict[str, int]
    total: dict[str, int]
    counts: dict[str, Counts]
    folds: list[FoldRecord] = field(default_factory=list)
    note: str = ""

    @property
    def binary(self) -> bool:
        return bool(self.counts)

    @property
    def name(self) -> str:
        return self.path.name if self.path.is_dir() else self.path.parent.name


def ordered_datasets(names: list[str]) -> list[str]:
    """Order dataset names as mdd, cane, sad, then anything else alphabetically."""
    known = [d for d in DATASET_ORDER if d in names]
    return known + sorted(n for n in names if n not in DATASET_ORDER)


# --------------------------------------------------------------------------------------------
# Checkpoint mode
# --------------------------------------------------------------------------------------------


def detect_orientation(stored: Counts, confusion_matrix: np.ndarray) -> Orientation:
    """Decide whether stored per-dataset fp/fn are swapped, using the chunk confusion matrix.

    :param Counts stored: Stored per-dataset ``(tp, tn, fp, fn)`` summed over all datasets.
    :param np.ndarray confusion_matrix: Chunk confusion matrix ``[[TN, FP], [FN, TP]]``.
    :return: ``"unswapped"`` if the sums equal ``(TP, TN, FP, FN)``, ``"swapped"`` if they equal
        ``(TP, TN, FN, FP)``, ``"ambiguous"`` if both hold (``FP == FN``).
    :rtype: str
    :raises CountMismatchError: If neither reading matches.
    """
    tn, fp, fn, tp = (int(v) for v in np.asarray(confusion_matrix).ravel())
    as_stored = Counts(tp, tn, fp, fn)
    exchanged = Counts(tp, tn, fn, fp)
    if stored == as_stored and stored == exchanged:
        return "ambiguous"
    if stored == as_stored:
        return "unswapped"
    if stored == exchanged:
        return "swapped"
    raise CountMismatchError(
        f"per-dataset counts summed over datasets (tp={stored.tp}, tn={stored.tn}, "
        f"fp={stored.fp}, fn={stored.fn}) match neither (TP, TN, FP, FN) nor (TP, TN, FN, FP) of "
        f"the chunk confusion matrix (TN={tn}, FP={fp}, FN={fn}, TP={tp}). Likely a dataset "
        "missing from the subject->dataset map ('unknown') or a corrupted checkpoint."
    )


def _fold_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    folds = []
    for path in run_dir.glob("fold_*_best.pth"):
        match = FOLD_FILE_RE.search(path.name)
        if match:
            folds.append((int(match.group(1)), path))
    return sorted(folds)


def load_run(run_dir: Path, assume: str = "auto", expect_folds: int = 10) -> RunCounts:
    """Load and orientation-correct the per-dataset counts of every fold of a run.

    :param Path run_dir: Directory with ``fold_N_best.pth`` files.
    :param str assume: ``"auto"`` to detect the fp/fn orientation per fold, or ``"swapped"`` /
        ``"unswapped"`` to force it.
    :param int expect_folds: Number of folds a complete run has; a warning is logged if the
        run has a different number of fold checkpoints (10-fold CV is mandatory here).
    :return: Pooled, oriented per-dataset counts.
    :rtype: RunCounts
    :raises SkipRunError: If the run has no checkpoints or they lack ``per_dataset``.
    :raises CountMismatchError: If a fold's per-dataset counts contradict its confusion matrix.
    """
    checkpoints = _fold_checkpoints(run_dir)
    if not checkpoints:
        raise SkipRunError("no fold_N_best.pth checkpoints")

    present = [fold for fold, _ in checkpoints]
    missing = [f for f in range(1, expect_folds + 1) if f not in present]
    if missing or len(present) != expect_folds:
        logger.warning(
            f"{run_dir.name}: {len(present)} fold checkpoints found, expected {expect_folds} "
            f"(missing folds {missing}); the pooled numbers below cover a partial run"
        )

    binary: Optional[bool] = None
    folds: list[FoldRecord] = []
    for fold, path in checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        val_metrics = ckpt.get("val_metrics", {})
        per_dataset = val_metrics.get("per_dataset")
        if not per_dataset:
            raise SkipRunError(f"{path.name} has no val_metrics['per_dataset']")
        confusion_matrix = np.asarray(val_metrics["chunk"]["confusion_matrix"])
        if binary is None:
            binary = confusion_matrix.shape == (2, 2)
        record = FoldRecord(
            fold=fold,
            correct={d: int(m["correct"]) for d, m in per_dataset.items()},
            total={d: int(m["total"]) for d, m in per_dataset.items()},
        )
        if binary:
            stored = {
                d: Counts(int(m["tp"]), int(m["tn"]), int(m["fp"]), int(m["fn"]))
                for d, m in per_dataset.items()
            }
            try:
                record.orientation = detect_orientation(
                    sum(stored.values(), Counts()), confusion_matrix
                )
            except CountMismatchError as exc:
                raise CountMismatchError(f"{run_dir.name} fold {fold}: {exc}") from exc
            record.counts = stored  # re-oriented below once the run-level decision is known
        folds.append(record)

    note = "multi-class run: only per-dataset accuracy is available"
    if binary:
        note = _resolve_orientations(run_dir, folds, assume)

    pooled_correct: dict[str, int] = {}
    pooled_total: dict[str, int] = {}
    pooled_counts: dict[str, Counts] = {}
    for record in folds:
        for d in record.total:
            pooled_correct[d] = pooled_correct.get(d, 0) + record.correct[d]
            pooled_total[d] = pooled_total.get(d, 0) + record.total[d]
            if binary:
                pooled_counts[d] = pooled_counts.get(d, Counts()) + record.counts[d]
    return RunCounts(
        path=run_dir,
        source="checkpoints",
        correct=pooled_correct,
        total=pooled_total,
        counts=pooled_counts,
        folds=folds,
        note=note,
    )


def _resolve_orientations(run_dir: Path, folds: list[FoldRecord], assume: str) -> str:
    """Fix each fold's counts in place and return a description of the orientation found.

    :param Path run_dir: Run directory, for log messages.
    :param list folds: Fold records whose ``counts`` still hold the stored orientation.
    :param str assume: ``"auto"``, ``"swapped"`` or ``"unswapped"``.
    :return: Description of the detected orientation.
    :rtype: str
    """
    detected = {f.orientation for f in folds if f.orientation != "ambiguous"}
    if assume != "auto":
        decided: dict[int, str] = {f.fold: assume for f in folds}
        conflicts = [f.fold for f in folds if f.orientation not in ("ambiguous", assume)]
        if conflicts:
            logger.warning(
                f"{run_dir.name}: --assume {assume} contradicts the detected orientation in "
                f"folds {conflicts}"
            )
        description = f"forced by --assume {assume}"
    else:
        if not detected:
            raise CountMismatchError(
                f"{run_dir.name}: every fold has FP == FN, so the fp/fn orientation cannot be "
                "detected. Re-run with --assume swapped (pre-fix runs) or --assume unswapped."
            )
        if len(detected) > 1:
            logger.warning(
                f"{run_dir.name}: mixed orientations across folds "
                f"({sorted(detected)}); handling each fold by its own detection"
            )
        fallback = "swapped" if "swapped" in detected else "unswapped"
        decided = {
            f.fold: (f.orientation if f.orientation != "ambiguous" else fallback) for f in folds
        }
        description = "detected " + "/".join(sorted(detected))
        ambiguous = [f.fold for f in folds if f.orientation == "ambiguous"]
        if ambiguous:
            description += f" (folds {ambiguous} have FP == FN, taken as {fallback})"

    for record in folds:
        if decided[record.fold] == "swapped":
            record.counts = {d: c.swapped() for d, c in record.counts.items()}
    logger.debug(f"[{run_dir.name}] per-dataset fp/fn orientation: {description}")
    return description


# --------------------------------------------------------------------------------------------
# results.txt mode
# --------------------------------------------------------------------------------------------

# Parses the table written by thesis.metrics.write_per_dataset_table; keep the two in sync.
PER_DATASET_HEADER = "Per-Dataset Chunk Metrics"
PER_DATASET_ROW_RE = re.compile(
    r"^\s*(?P<ds>[A-Za-z0-9_]+)\s+(?P<acc>\d+(?:\.\d+)?)%\s+(?P<sens>\d+(?:\.\d+)?)%\s+"
    r"(?P<spec>\d+(?:\.\d+)?)%\s+(?P<n>\d+)\s*$"
)
CM_HEALTHY_RE = re.compile(r"^\s*Healthy\s+(\d+)\s+(\d+)\s*$")
CM_PATHOLOGICAL_RE = re.compile(r"^\s*Pathological\s+(\d+)\s+(\d+)\s*$")


def reconstruct_counts(
    accuracy: float, sens: float, spec: float, n: int, orientation: str = "swapped"
) -> Counts:
    """Rebuild ``(tp, tn, fp, fn)`` of one dataset from its rounded results.txt percentages.

    :param float accuracy: Chunk accuracy as a fraction in [0, 1].
    :param float sens: Printed "sensitivity" as a fraction.
    :param float spec: Printed "specificity" as a fraction.
    :param int n: Number of chunks of the dataset.
    :param str orientation: ``"swapped"``: printed sens/spec are PPV ``TP/(TP+FP)`` and NPV
        ``TN/(TN+FN)`` (pre-fix runs). ``"fixed"``: they are ``TP/(TP+FN)`` and ``TN/(TN+FP)``.
    :return: Reconstructed oriented counts (rounded to integers, summing to ``n``).
    :rtype: Counts
    :raises ValueError: If the equations are degenerate (equal, zero or unit sens/spec).
    """
    correct = round(accuracy * n)
    wrong = n - correct
    if orientation == "fixed":
        if abs(sens - spec) < 1e-9:
            raise ValueError("sensitivity == specificity: positives/negatives are not identifiable")
        positives = (correct - spec * n) / (sens - spec)
        tp = round(sens * positives)
        tn = correct - tp
        fn = round(positives) - tp
        fp = wrong - fn
        return Counts(tp, tn, fp, fn)
    if orientation != "swapped":
        raise ValueError(f"unknown orientation {orientation!r}")
    if not (0 < sens <= 1 and 0 < spec <= 1):
        raise ValueError("PPV/NPV of 0 make the equations degenerate")
    ratio_p = (1 - sens) / sens  # FP / TP
    ratio_n = (1 - spec) / spec  # FN / TN
    if abs(ratio_p - ratio_n) < 1e-9:
        raise ValueError("PPV == NPV: counts are not identifiable")
    tp_float = (wrong - correct * ratio_n) / (ratio_p - ratio_n)
    tp = round(tp_float)
    tn = correct - tp
    fp = round(tp * ratio_p)
    fn = wrong - fp
    return Counts(tp, tn, fp, fn)


def parse_results_txt(
    path: Path,
) -> tuple[dict[str, tuple[float, float, float, int]], Optional[np.ndarray]]:
    """Parse the aggregated per-dataset table (and confusion matrix) of a results.txt.

    :param Path path: results.txt written by ``main.py train``.
    :return: ``({dataset: (acc, sens, spec, chunks)}, confusion_matrix or None)``, the
        percentages converted to fractions and the matrix as ``[[TN, FP], [FN, TP]]``.
    :rtype: tuple
    :raises SkipRunError: If the file has no per-dataset table.
    """
    lines = path.read_text().splitlines()
    rows: dict[str, tuple[float, float, float, int]] = {}
    in_table = False
    healthy: Optional[tuple[int, int]] = None
    pathological: Optional[tuple[int, int]] = None
    for line in lines:
        if PER_DATASET_HEADER in line:
            in_table = True
            continue
        if in_table:
            match = PER_DATASET_ROW_RE.match(line)
            if match and match["ds"].lower() != "total":
                rows[match["ds"].lower()] = (
                    float(match["acc"]) / 100,
                    float(match["sens"]) / 100,
                    float(match["spec"]) / 100,
                    int(match["n"]),
                )
        if (m := CM_HEALTHY_RE.match(line)) and healthy is None:
            healthy = (int(m[1]), int(m[2]))
        if (m := CM_PATHOLOGICAL_RE.match(line)) and pathological is None:
            pathological = (int(m[1]), int(m[2]))
    if not rows:
        raise SkipRunError("no 'Per-Dataset Chunk Metrics' table in results.txt")
    matrix = None
    if healthy is not None and pathological is not None:
        matrix = np.array([healthy, pathological])
    return rows, matrix


def load_from_results_txt(path: Path, orientation: str = "swapped") -> RunCounts:
    """Reconstruct pooled per-dataset counts from a results.txt (best effort).

    :param Path path: results.txt path.
    :param str orientation: ``"swapped"`` or ``"fixed"``, see :func:`reconstruct_counts`.
    :return: Pooled per-dataset counts.
    :rtype: RunCounts
    """
    rows, matrix = parse_results_txt(path)
    counts: dict[str, Counts] = {}
    for dataset, (accuracy, sens, spec, n) in rows.items():
        counts[dataset] = reconstruct_counts(accuracy, sens, spec, n, orientation)
    total = {d: c.total for d, c in counts.items()}
    correct = {d: c.correct for d, c in counts.items()}
    note = f"reconstructed from rounded results.txt percentages (printed sens/spec: {orientation})"
    if matrix is not None:
        summed = sum(counts.values(), Counts())
        tn, fp, fn, tp = (int(v) for v in matrix.ravel())
        deviation = max(
            abs(summed.tp - tp), abs(summed.tn - tn), abs(summed.fp - fp), abs(summed.fn - fn)
        )
        note += f"; max |reconstructed - confusion matrix| = {deviation} chunks"
        logger.info(f"[{path.parent.name}] {note}")
    return RunCounts(
        path=path, source="results.txt", correct=correct, total=total, counts=counts, note=note
    )


# --------------------------------------------------------------------------------------------
# Baseline arithmetic and reporting
# --------------------------------------------------------------------------------------------


@dataclass
class PriorRow:
    """One line of the baseline table.

    :param str dataset: Dataset name (``"overall"`` for the pooled line).
    :param int chunks: Number of validation chunks.
    :param float pathological: Fraction of truly pathological chunks.
    :param float accuracy: Model chunk accuracy.
    :param float baseline: Majority-class accuracy (dataset prior).
    :param float gain_pp: ``accuracy - baseline`` in percentage points.
    :param float contribution_pp: Share of the overall gain, in pp of the overall chunk count.
    :param str majority: ``"P"`` if the majority class is pathological, else ``"H"``.
    :param float sensitivity: True sensitivity (oriented counts).
    :param float specificity: True specificity (oriented counts).
    """

    dataset: str
    chunks: int
    pathological: float
    accuracy: float
    baseline: float
    gain_pp: float
    contribution_pp: float
    majority: str
    sensitivity: float
    specificity: float


def prior_baseline(counts: dict[str, Counts]) -> tuple[list[PriorRow], PriorRow]:
    """Compute per-dataset and overall dataset-prior baselines.

    :param dict counts: Pooled oriented counts per dataset.
    :return: ``(per-dataset rows, overall row)``. The per-dataset ``contribution_pp`` values sum
        to the overall ``gain_pp``.
    :rtype: tuple[list[PriorRow], PriorRow]
    """
    grand_total = sum(c.total for c in counts.values())
    rows = []
    for dataset in ordered_datasets(list(counts)):
        c = counts[dataset]
        rows.append(
            PriorRow(
                dataset=dataset,
                chunks=c.total,
                pathological=c.positives / c.total,
                accuracy=c.accuracy,
                baseline=c.baseline,
                gain_pp=(c.correct - c.majority_correct) / c.total * 100,
                contribution_pp=(c.correct - c.majority_correct) / grand_total * 100,
                majority="P" if c.positives >= c.negatives else "H",
                sensitivity=c.sensitivity,
                specificity=c.specificity,
            )
        )
    overall = sum(counts.values(), Counts())
    majority_total = sum(c.majority_correct for c in counts.values())
    gain_pp = (overall.correct - majority_total) / grand_total * 100
    overall_row = PriorRow(
        dataset="overall",
        chunks=grand_total,
        pathological=overall.positives / grand_total,
        accuracy=overall.accuracy,
        baseline=majority_total / grand_total,
        gain_pp=gain_pp,
        contribution_pp=gain_pp,
        majority="-",
        sensitivity=overall.sensitivity,
        specificity=overall.specificity,
    )
    return rows, overall_row


def _pct(value: float) -> str:
    return "  n/a" if np.isnan(value) else f"{value * 100:5.1f}"


def report_run(run: RunCounts) -> Optional[tuple[list[PriorRow], PriorRow]]:
    """Log the detailed per-run table.

    :param RunCounts run: Pooled counts of one run.
    :return: The computed rows for binary runs, ``None`` for multi-class runs.
    :rtype: Optional[tuple[list[PriorRow], PriorRow]]
    """
    n_folds = len(run.folds)
    folds_str = f"{n_folds} folds" if n_folds else "folds unknown"
    logger.info(f"\n=== {run.name} ({run.source}, {folds_str}) ===")
    logger.info(f"    {run.note}")
    if not run.binary:
        logger.info(f"  {'Dataset':<8} {'Chunks':>7} {'Acc%':>6}")
        for dataset in ordered_datasets(list(run.total)):
            acc = run.correct[dataset] / run.total[dataset]
            logger.info(f"  {dataset.upper():<8} {run.total[dataset]:>7} {_pct(acc):>6}")
        logger.info("  (baseline not computed: class balance per dataset is not stored)")
        return None

    rows, overall = prior_baseline(run.counts)
    header = (
        f"  {'Dataset':<8} {'Chunks':>7} {'Path%':>6} {'Acc%':>6} {'Base%':>6} {'Maj':>3} "
        f"{'Gain pp':>8} {'Contrib pp':>10} {'Sens%':>6} {'Spec%':>6}"
    )
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for row in [*rows, overall]:
        logger.info(
            f"  {row.dataset.upper():<8} {row.chunks:>7} {_pct(row.pathological):>6} "
            f"{_pct(row.accuracy):>6} {_pct(row.baseline):>6} {row.majority:>3} "
            f"{row.gain_pp:>+8.1f} {row.contribution_pp:>+10.1f} "
            f"{_pct(row.sensitivity):>6} {_pct(row.specificity):>6}"
        )
    logger.info(
        "  Path% = pathological fraction P; Base% = max(P, 1-P); Maj = majority class "
        "(P/H); Contrib = (correct - majority correct) / all chunks."
    )
    return rows, overall


def report_summary(results: dict[str, tuple[list[PriorRow], PriorRow]]) -> None:
    """Log the compact cross-run table in the layout of notes section 3.4.

    :param dict results: ``run name -> (per-dataset rows, overall row)``.
    """
    if not results:
        return
    logger.info("\n=== Summary (acc / dataset-prior baseline, %) ===")
    header = (
        f"{'Run':<34} {'MDD':>11} {'CANE':>11} {'SAD':>11} {'Overall':>7} {'Prior':>6} "
        f"{'Gain':>6}  Gain from MDD / CANE / SAD (pp)"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for name, (rows, overall) in results.items():
        by_ds = {r.dataset: r for r in rows}
        cells = []
        contributions = []
        for dataset in DATASET_ORDER:
            row = by_ds.get(dataset)
            cells.append(
                f"{row.accuracy * 100:5.1f}/{row.baseline * 100:4.1f}" if row else "n/a".rjust(10)
            )
            contributions.append(f"{row.contribution_pp:+.1f}" if row else "n/a")
        logger.info(
            f"{name[:34]:<34} {cells[0]:>11} {cells[1]:>11} {cells[2]:>11} "
            f"{overall.accuracy * 100:7.1f} {overall.baseline * 100:6.1f} "
            f"{overall.gain_pp:+6.1f}  " + " / ".join(contributions)
        )


# --------------------------------------------------------------------------------------------
# Discovery / CLI
# --------------------------------------------------------------------------------------------


def expand_patterns(root: Path, patterns: list[str]) -> list[Path]:
    """Expand glob patterns (relative ones against ``root``) into sorted unique paths.

    :param Path root: Base directory for relative patterns.
    :param list patterns: Paths or glob patterns.
    :return: Existing matching paths in sorted order.
    :rtype: list[Path]
    """
    found: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        full = str(candidate if candidate.is_absolute() else root / candidate)
        matches = sorted(glob.glob(full))
        if not matches:
            logger.warning(f"pattern {pattern!r} matched nothing under {root}")
        found.extend(Path(m) for m in matches)
    return list(dict.fromkeys(found))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run directories or glob patterns (relative ones resolve against --root). With "
        "--from-results-txt: results.txt files or run directories containing one.",
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="Base directory for relative run patterns."
    )
    parser.add_argument(
        "--assume",
        choices=["auto", "swapped", "unswapped"],
        default="auto",
        help="Force the fp/fn orientation instead of detecting it from the confusion matrix.",
    )
    parser.add_argument(
        "--expect-folds",
        type=int,
        default=10,
        help="Fold count of a complete run; runs with fewer checkpoints trigger a warning.",
    )
    parser.add_argument(
        "--from-results-txt",
        action="store_true",
        help="Reconstruct counts from the aggregated results.txt table (no checkpoints).",
    )
    parser.add_argument(
        "--results-orientation",
        choices=["swapped", "fixed"],
        default="swapped",
        help="Meaning of the printed sens/spec columns of results.txt (default: swapped, i.e. "
        "PPV/NPV, for every run that predates the main.py argument-order fix).",
    )
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
        setup_report_logging()
    if not args.runs:
        logger.error("no run directories given")
        return 2

    results: dict[str, tuple[list[PriorRow], PriorRow]] = {}
    skipped: list[tuple[str, str]] = []
    for path in expand_patterns(args.root, args.runs):
        try:
            if args.from_results_txt:
                txt = path / "results.txt" if path.is_dir() else path
                if not txt.exists():
                    raise SkipRunError("no results.txt")
                run = load_from_results_txt(txt, args.results_orientation)
            else:
                if not path.is_dir():
                    continue
                run = load_run(path, args.assume, args.expect_folds)
        except SkipRunError as exc:
            skipped.append((str(path), str(exc)))
            continue
        report = report_run(run)
        if report is not None:
            results[run.name] = report

    report_summary(results)
    if skipped:
        logger.info("\n=== Skipped runs ===")
        for name, reason in skipped:
            hint = ""
            if not args.from_results_txt and (Path(name) / "results.txt").exists():
                hint = "  (results.txt exists: try --from-results-txt)"
            logger.info(f"  {name}: {reason}{hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
