"""Structured JSON logging for training metrics."""

import json
import logging
from datetime import datetime
from pathlib import Path


def _find_log_file() -> Path | None:
    """Find the active file handler's log path from the root logger.

    :return: Path to the log file, or None if no file handler is attached.
    :rtype: Path | None
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            return Path(handler.baseFilename)
    return None


def _metrics_to_json_dict(metrics: dict, num_classes: int, include_confusion: bool = False) -> dict:
    """Convert a classification_metrics() dict to a compact JSON-friendly dict.

    For binary (num_classes=2): extracts acc, prec, rec, spec and optionally tp/tn/fp/fn.
    For multi-class (num_classes=4): extracts acc and per-class prec/rec.

    :param dict metrics: Output of classification_metrics().
    :param int num_classes: Number of classes (2 or 4).
    :param bool include_confusion: If True and binary, include tp/tn/fp/fn from confusion_matrix.
    :return: Compact dict suitable for JSON serialisation.
    :rtype: dict
    """
    if num_classes == 2:
        out: dict = {
            "acc": float(metrics["accuracy"]),
            "prec": float(metrics["precision"]),
            "rec": float(metrics["recall"]),
            "spec": float(metrics["specificity"]),
        }
        if include_confusion:
            cm = metrics.get("confusion_matrix")
            if cm is not None:
                tn, fp, fn, tp = cm.ravel()
                out.update({"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)})
        return out

    # Multi-class: per-class precision and recall
    out = {"acc": float(metrics["accuracy"])}
    class_names = ["normal", "anxiety", "depression", "anxiety+depression"]
    for name in class_names:
        prec_key = f"precision_{name}"
        rec_key = f"recall_{name}"
        if prec_key in metrics:
            out[f"prec_{name}"] = float(metrics[prec_key])
            out[f"rec_{name}"] = float(metrics[rec_key])
    return out


def log_metrics_json(
    phase: str,
    fold: int,
    epoch: int,
    max_epoch: int,
    loss: float,
    chunk_metrics: dict,
    subject_metrics: dict | None = None,
    condition_metrics: dict | None = None,
    num_classes: int = 2,
) -> None:
    """Log training/eval metrics as structured JSON.

    Prints pretty-printed JSON to the console and appends a compact single-line
    JSON record to the active log file.  Non-metric status messages should still
    use the standard logger.

    :param str phase: ``"train"`` or ``"eval"``.
    :param int fold: 1-based fold number.
    :param int epoch: Current epoch number.
    :param int max_epoch: Maximum number of epochs.
    :param float loss: Loss value for this epoch.
    :param dict chunk_metrics: Chunk-level metrics dict from classification_metrics().
    :param dict | None subject_metrics: Subject-level metrics (eval phase only).
    :param dict | None condition_metrics: Per-condition metrics dict, e.g.
        ``{"EC": {...}, "EO": {...}}``.  Each value is a classification_metrics() dict.
    :param int num_classes: Number of classes (2 or 4).
    """
    record: dict = {
        "t": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.") + f"{datetime.now().microsecond // 1000:03d}",
        "phase": phase,
        "fold": fold,
        "epoch": epoch,
        "max_epoch": max_epoch,
        "loss": round(float(loss), 4),
        "chunk": _metrics_to_json_dict(
            chunk_metrics, num_classes, include_confusion=(phase == "eval")
        ),
    }

    if subject_metrics is not None:
        record["subject"] = _metrics_to_json_dict(
            subject_metrics, num_classes, include_confusion=(phase == "eval")
        )

    if condition_metrics is not None:
        cond: dict = {}
        if "EC" in condition_metrics:
            cond["ec_acc"] = round(float(condition_metrics["EC"]["accuracy"]), 4)
        if "EO" in condition_metrics:
            cond["eo_acc"] = round(float(condition_metrics["EO"]["accuracy"]), 4)
        if cond:
            record["condition"] = cond

    # Console: pretty-printed JSON
    print(json.dumps(record, indent=2))

    # File: compact single-line JSON
    log_file = _find_log_file()
    if log_file is not None:
        with open(log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
