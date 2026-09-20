"""Shared test utilities for the ``helpers-print/`` scripts.

``helpers-print`` is not an importable package (hyphen in the name), so tests load the scripts by
path. The synthetic checkpoint writer builds the same dict layout as ``main.py`` saves
(``val_metrics`` with ``chunk`` / ``subject`` / ``per_dataset``), so the scripts can be tested
without any dataset, model or GPU.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import numpy as np
import torch

HELPERS_DIR = Path(__file__).resolve().parent.parent / "helpers-print"


def load_helper_script(name: str) -> ModuleType:
    """Import ``helpers-print/<name>.py`` and register it in ``sys.modules``.

    :param str name: Script name without ``.py``.
    :return: The imported module.
    :rtype: ModuleType
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HELPERS_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_fold_checkpoint(
    path: Path,
    per_dataset_true: dict[str, tuple[int, int, int, int]],
    swapped: bool = True,
    subject_acc: float = 0.5,
    with_per_dataset: bool = True,
) -> np.ndarray:
    """Write a synthetic binary ``fold_N_best.pth`` and return its chunk confusion matrix.

    :param Path path: Checkpoint file to create.
    :param dict per_dataset_true: ``{dataset: (tp, tn, fp, fn)}`` with the TRUE counts.
    :param bool swapped: If True, store per-dataset fp/fn exchanged, as ``main.py`` did before the
        argument-order fix (stored fp = true FN, stored fn = true FP).
    :param float subject_acc: Value stored as subject accuracy.
    :param bool with_per_dataset: If False, omit ``per_dataset`` from ``val_metrics``.
    :return: The chunk confusion matrix ``[[TN, FP], [FN, TP]]``.
    :rtype: np.ndarray
    """
    tp, tn, fp, fn = (sum(col) for col in zip(*per_dataset_true.values()))
    total = tp + tn + fp + fn
    cm = np.array([[tn, fp], [fn, tp]])
    per_dataset: dict[str, dict[str, float | int]] = {}
    for name, (d_tp, d_tn, d_fp, d_fn) in per_dataset_true.items():
        stored_fp, stored_fn = (d_fn, d_fp) if swapped else (d_fp, d_fn)
        d_total = d_tp + d_tn + d_fp + d_fn
        per_dataset[name] = {
            "accuracy": (d_tp + d_tn) / d_total,
            "sensitivity": d_tp / (d_tp + stored_fn) if d_tp + stored_fn else 0.0,
            "specificity": d_tn / (d_tn + stored_fp) if d_tn + stored_fp else 0.0,
            "tp": d_tp,
            "tn": d_tn,
            "fp": stored_fp,
            "fn": stored_fn,
            "correct": d_tp + d_tn,
            "total": d_total,
        }
    chunk = {
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "confusion_matrix": cm,
    }
    _save_fold_checkpoint(path, chunk, per_dataset if with_per_dataset else None, subject_acc)
    return cm


def _save_fold_checkpoint(
    path: Path,
    chunk: dict[str, Any],
    per_dataset: Optional[dict[str, dict[str, Any]]],
    subject_acc: float = 0.5,
) -> None:
    """Write a ``fold_N_best.pth`` with the ``val_metrics`` skeleton consumers expect.

    The single place the checkpoint layout is spelled out, so checkpoints written by the
    different helpers here stay interchangeable.

    :param Path path: Checkpoint file to create.
    :param dict chunk: Chunk-level metrics, including ``confusion_matrix``.
    :param dict per_dataset: Per-dataset metrics, or None to omit the key entirely.
    :param float subject_acc: Value stored as subject accuracy.
    """
    val_metrics: dict[str, object] = {
        "chunk": chunk,
        # recall/specificity are not derived from real subject-level counts (these helpers only
        # track a single synthetic accuracy per fold); they default to the accuracy so that
        # consumers reading all three keys (e.g. bipolar_ablation_summary.load_run) don't KeyError.
        "subject": {"accuracy": subject_acc, "recall": subject_acc, "specificity": subject_acc},
        "condition": {},
        "loss": 0.1,
    }
    if per_dataset is not None:
        val_metrics["per_dataset"] = per_dataset
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": 1, "chunk_acc": chunk["accuracy"], "val_metrics": val_metrics}, path)


def write_run(
    run_dir: Path,
    folds: list[dict[str, tuple[int, int, int, int]]],
    swapped: bool = True,
    subject_accs: Optional[list[float]] = None,
) -> None:
    """Write ``fold_1_best.pth`` ... ``fold_N_best.pth`` for a synthetic run.

    :param Path run_dir: Run directory to create.
    :param list folds: One ``{dataset: (tp, tn, fp, fn)}`` dict (true counts) per fold.
    :param bool swapped: Store per-dataset fp/fn in the pre-fix (exchanged) orientation.
    :param Optional[list[float]] subject_accs: Optional per-fold subject accuracies.
    """
    for i, fold in enumerate(folds, start=1):
        write_fold_checkpoint(
            run_dir / f"fold_{i}_best.pth",
            fold,
            swapped=swapped,
            subject_acc=subject_accs[i - 1] if subject_accs else 0.5,
        )


def metrics_from_matrices(
    matrices: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Build the ``(chunk, per_dataset)`` metrics ``eval_epoch`` would return for ``matrices``.

    The metrics are produced by the real ``classification_metrics`` and
    ``compute_per_dataset_metrics`` from chunk labels expanded out of ``matrices``, so they carry
    exactly what ``main.py`` stores (including the meaningless-for-4-class ``tp``/``tn``/``fp``/
    ``fn``).

    :param dict matrices: ``{dataset: K x K confusion matrix}`` with rows = true class and
        columns = predicted class; K = 2 gives binary metrics, K = 4 4-class ones.
    :return: ``(chunk metrics, per-dataset metrics)``.
    :rtype: tuple[dict, dict]
    """
    from thesis.metrics import classification_metrics, compute_per_dataset_metrics

    num_classes = next(iter(matrices.values())).shape[0]
    y_true: list[int] = []
    y_pred: list[int] = []
    subjects: list[str] = []
    for dataset, matrix in matrices.items():
        for true_class, pred_class in np.ndindex(matrix.shape):
            count = int(matrix[true_class, pred_class])
            y_true += [true_class] * count
            y_pred += [pred_class] * count
            subjects += [f"{dataset}-subject"] * count
    true_arr, pred_arr = np.array(y_true), np.array(y_pred)
    chunk = classification_metrics(true_arr, pred_arr, num_classes=num_classes)
    per_dataset = compute_per_dataset_metrics(
        y_true=true_arr,
        y_pred=pred_arr,
        subjects=subjects,
        subject_dataset_map={f"{d}-subject": d for d in matrices},
        num_classes=num_classes,
    )
    return chunk, per_dataset


def write_matrix_fold_checkpoint(path: Path, matrices: dict[str, np.ndarray]) -> None:
    """Write a synthetic ``fold_N_best.pth`` whose ``per_dataset`` holds confusion matrices.

    :param Path path: Checkpoint file to create.
    :param dict matrices: ``{dataset: K x K confusion matrix}``, see :func:`metrics_from_matrices`.
    """
    chunk, per_dataset = metrics_from_matrices(matrices)
    _save_fold_checkpoint(path, chunk, per_dataset)
