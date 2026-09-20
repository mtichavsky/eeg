"""Classification metrics for EEG depression/anxiety detection."""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from thesis.labels import get_display_names

#: Per-dataset metrics: ``{dataset: {metric name: scalar or ``confusion_matrix`` array}}``.
PerDatasetMetrics = dict[str, dict[str, Any]]

#: Section title written above the per-dataset confusion matrices in ``results.txt``.
PER_DATASET_CM_HEADER = "Per-Dataset Chunk Confusion Matrices"


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2
) -> dict[str, float | int | np.floating[Any] | NDArray[Any] | None]:
    """
    Compute classification metrics from true and predicted labels.

    For binary classification (num_classes=2): computes accuracy, precision, recall, specificity.
    For multi-class (num_classes>2): computes accuracy and per-class precision/recall.

    :param np.ndarray y_true: True labels.
    :param np.ndarray y_pred: Predicted labels.
    :param int num_classes: Number of classes (2 for binary, 3 for ternary).
    :return: Dictionary with metrics.
    :rtype: dict
    """
    acc = np.mean(y_true == y_pred)
    metrics: dict[str, float | int | np.floating[Any] | NDArray[Any] | None] = {"accuracy": acc}

    if num_classes == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics.update(
                {
                    "precision": precision,
                    "recall": recall,
                    "specificity": specificity,
                    "confusion_matrix": cm,
                }
            )
        else:
            # Handle edge case where only one class is predicted
            metrics.update(
                {
                    "precision": 0.0,
                    "recall": 0.0,
                    "specificity": 0.0,
                    "confusion_matrix": None,
                }
            )
    else:
        # Multi-class: compute per-class precision and recall
        all_labels = list(range(num_classes))
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        for class_idx in all_labels:
            # Per-class metrics
            tp = cm[class_idx, class_idx]
            fp = cm[:, class_idx].sum() - tp
            fn = cm[class_idx, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            display_names = get_display_names(num_classes)
            class_name = display_names.get(class_idx, f"class_{class_idx}")

            metrics[f"precision_{class_name}"] = precision
            metrics[f"recall_{class_name}"] = recall

        # Store confusion matrix as well
        metrics["confusion_matrix"] = cm

    return metrics


def aggregate_subject_predictions(
    chunk_preds: np.ndarray, chunk_labels: np.ndarray, subjects: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate chunk-level predictions to subject-level using majority voting.

    For each subject, collects all chunk predictions and computes the most frequent
    prediction (mode) as the final subject-level prediction. This is useful for
    EEG analysis where multiple temporal chunks come from the same subject.

    :param np.ndarray chunk_preds: Predictions for each chunk (0/1).
    :param np.ndarray chunk_labels: Ground truth labels for each chunk (0/1).
    :param list[str] subjects: Subject ID for each chunk (e.g., "H S1 EC", "MDD S2 EO").
    :return: Tuple of (subject_predictions, subject_labels, subject_ids) where each array
             contains one value per unique subject.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray]

    Example:
        >>> chunk_preds = np.array([0, 1, 1, 0, 0])
        >>> chunk_labels = np.array([0, 0, 0, 1, 1])
        >>> subjects = ["H S1 EC", "H S1 EC", "H S1 EC", "MDD S2 EO", "MDD S2 EO"]
        >>> subj_preds, subj_labels, subj_ids = aggregate_subject_predictions(
        ...     chunk_preds, chunk_labels, subjects
        ... )
        >>> subj_preds  # H S1 EC gets 1 (majority), MDD S2 EO gets 0 (majority)
        array([1, 0])
        >>> subj_labels  # H S1 EC is 0, MDD S2 EO is 1
        array([0, 1])
    """
    subject_preds_dict: dict[str, list[int]] = {}
    subject_labels_dict: dict[str, int] = {}

    # Group predictions by subject
    for pred, label, subject in zip(chunk_preds, chunk_labels, subjects):
        if subject not in subject_preds_dict:
            subject_preds_dict[subject] = []
            subject_labels_dict[subject] = label  # All chunks from same subject have same label
        subject_preds_dict[subject].append(pred)

    # Compute majority vote for each subject
    subject_preds = []
    subject_labels = []
    subject_ids = []
    for subject in subject_preds_dict:
        chunk_preds_for_subject = subject_preds_dict[subject]
        majority_pred = np.bincount(chunk_preds_for_subject).argmax()
        subject_preds.append(majority_pred)
        subject_labels.append(subject_labels_dict[subject])
        subject_ids.append(subject)

    return np.array(subject_preds), np.array(subject_labels), np.array(subject_ids)


def format_metrics_for_logging(metrics: dict[str, float | int], num_classes: int) -> str:
    """
    Format metrics dictionary for logging.

    :param dict metrics: Metrics dictionary from classification_metrics().
    :param int num_classes: Number of classes (2 or 3).
    :return: Formatted metrics string.
    :rtype: str
    """
    if num_classes == 2:
        # Binary classification: use traditional metrics
        return (
            f"Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Spec: {metrics['specificity']:.4f}"
        )
    else:
        # Multi-class: show per-class precision and recall
        parts = [f"Acc: {metrics['accuracy']:.4f}"]
        for key in metrics:
            if key.startswith("precision_"):
                class_name = key.replace("precision_", "")
                prec = metrics[key]
                rec = metrics.get(f"recall_{class_name}", 0.0)
                parts.append(f"{class_name}: P={prec:.3f} R={rec:.3f}")
        return " | ".join(parts)


def extract_classification_metrics(metrics: dict[str, float], num_classes: int) -> dict[str, float]:
    """
    Extract relevant metrics based on classification type (binary vs multi-class).

    For binary classification: returns accuracy, sensitivity, specificity.
    For multi-class: returns accuracy and per-class recall.

    :param dict[str, float] metrics: Source metrics dictionary from classification_metrics().
    :param int num_classes: Number of classes (2 for binary, >2 for multi-class).
    :return: Dictionary with accuracy and type-specific metrics.
    :rtype: dict[str, float]
    """
    if num_classes == 2:
        return {
            "accuracy": metrics["accuracy"],
            "sensitivity": metrics["recall"],
            "specificity": metrics["specificity"],
        }
    elif num_classes == 4:
        display_names = get_display_names(num_classes)
        result: dict[str, float] = {"accuracy": metrics["accuracy"]}
        for name in display_names.values():
            result[f"recall_{name}"] = metrics.get(f"recall_{name}", 0.0)
        return result
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")


def _format_pct(value: float | np.floating[Any]) -> str:
    """
    Format a decimal value as percentage with 2 decimal places.

    :param float | np.floating value: Decimal value (e.g., 0.85).
    :return: Formatted percentage string (e.g., "85.00%").
    :rtype: str
    """
    return f"{value * 100:.2f}%"


def _format_metric_line(name: str, values: list[float]) -> str:
    """
    Format a metric line with mean, std, min, max as percentages.

    :param str name: Metric name for the output line.
    :param list[float] values: List of metric values across folds.
    :return: Formatted string with statistics.
    :rtype: str
    """
    mean = np.mean(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    return (
        f"{name}: {_format_pct(mean)} ± {_format_pct(std)}, "
        f"Min: {_format_pct(min_val)}, Max: {_format_pct(max_val)}\n"
    )


def _extract_metric_values(fold_metrics: list[dict[str, float]], key: str) -> list[float]:
    """
    Extract a specific metric value from each fold's metrics dict.

    :param list[dict[str, float]] fold_metrics: List of metric dicts from each fold.
    :param str key: Metric key to extract.
    :return: List of extracted values.
    :rtype: list[float]
    """
    return [m.get(key, 0.0) for m in fold_metrics]


def _write_metrics_block(
    writer: Callable[[str], Any],
    metrics: list[dict[str, float]],
    level: str,
) -> None:
    """
    Write a block of metrics for either chunk-level or subject-level evaluation.

    For binary classification: writes accuracy, sensitivity, specificity.
    For multi-class: writes accuracy and per-class recall.

    :param Callable[[str], Any] writer: Function to write output (e.g., logger.info).
    :param list[dict[str, float]] metrics: List of metric dicts from each fold.
    :param str level: Metric level identifier, either "CHUNK" or "SUBJECT".
    :return: None
    :rtype: None
    """
    if not metrics:
        return

    acc_values = _extract_metric_values(metrics, "accuracy")
    label = f"{level} Accuracy Mean"
    writer(_format_metric_line(label, acc_values))

    # Binary classification: sensitivity & specificity
    if "sensitivity" in metrics[0]:
        sens_values = _extract_metric_values(metrics, "sensitivity")
        spec_values = _extract_metric_values(metrics, "specificity")
        writer(_format_metric_line(f"{level} Sensitivity Mean", sens_values))
        writer(_format_metric_line(f"{level} Specificity Mean", spec_values))

    # Multi-class: per-class recall (detect class names dynamically)
    elif "recall_Healthy" in metrics[0]:
        # Extract all recall_* keys from first metrics dict
        class_keys = [k for k in metrics[0].keys() if k.startswith("recall_")]
        for key in class_keys:
            class_name = key.replace("recall_", "")
            values = _extract_metric_values(metrics, key)
            writer(_format_metric_line(f"{level} Recall ({class_name.capitalize()}) Mean", values))


def _class_names(num_classes: int) -> list[str]:
    """
    Return the display name of every class index, falling back to ``Class N``.

    :param int num_classes: Number of classes (2 or 4).
    :return: One display name per class index.
    :rtype: list[str]
    """
    display_names = get_display_names(num_classes)
    return [display_names.get(i, f"Class {i}") for i in range(num_classes)]


def _write_matrix_body(
    writer: Callable[[str], Any],
    matrix: NDArray[Any],
    class_names: list[str],
    indent: str = "",
    separator: bool = False,
) -> None:
    """
    Write the column header and one row per class of a confusion matrix.

    The single source of the matrix column geometry, shared by the pooled matrix
    (:func:`_format_confusion_matrix`) and the per-dataset ones
    (:func:`_write_per_dataset_confusion_matrices`). ``helpers-print/dataset_prior_baseline.py``
    parses both layouts, so a change here must be matched there.

    :param Callable[[str], Any] writer: Function to write output.
    :param NDArray matrix: Square confusion matrix, rows = actual, columns = predicted.
    :param list[str] class_names: Display name of every class index.
    :param str indent: Prefix written before every line.
    :param bool separator: Whether to write a dashed rule between header and rows.
    :return: None
    :rtype: None
    """
    # Use ASCII arrows to avoid 2-wide Unicode rendering in editors/fonts
    header = "ACTUAL v/PRED ->".ljust(16) + "".join(f"{name:>15} " for name in class_names)
    writer(indent + header + "\n")
    if separator:
        # Width matches header + data rows (16 label + 16 per column)
        writer("-" * (16 + 16 * len(class_names)) + "\n")
    for i, row_name in enumerate(class_names):
        row = "".join(f"{int(matrix[i, j]):>15} " for j in range(len(class_names)))
        writer(f"{indent}{row_name:>15} {row}\n")


def _format_confusion_matrix(
    confusion_matrices: list[np.ndarray],
    num_classes: int,
    writer: Callable[[str], Any],
) -> None:
    """
    Aggregate confusion matrices across folds and format for output.

    :param list[np.ndarray] confusion_matrices: List of confusion matrices from each fold.
    :param int num_classes: Number of classes (2 or 4).
    :param Callable[[str], Any] writer: Function to write output (e.g., logger.info or file.write).
    :return: None
    :rtype: None
    """
    # Sum confusion matrices across folds
    aggregated_cm = np.sum(confusion_matrices, axis=0)

    writer("\nAggregated Chunk Confusion Matrix (across folds; TN, FP, FN, TP):\n\n")
    _write_matrix_body(writer, aggregated_cm, _class_names(num_classes), separator=True)

    # Additional statistics
    writer("\n")
    total_predictions = np.sum(aggregated_cm)
    correct_predictions = np.trace(aggregated_cm)
    overall_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    writer(f"Total Predictions: {int(total_predictions)}\n")
    writer(f"Correct Predictions: {int(correct_predictions)}\n")
    writer(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)\n")


def compute_per_dataset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: list[str],
    subject_dataset_map: dict[str, str],
    num_classes: int = 2,
) -> PerDatasetMetrics:
    """
    Compute chunk-level accuracy, sensitivity, specificity and confusion matrix per dataset.

    Sensitivity, specificity and the ``tp``/``tn``/``fp``/``fn`` counts are binary quantities
    (labels 0 and 1, where 0 is healthy and 1 is pathological). In a multi-class run they only
    cover the chunks whose true and predicted labels are both 0 or 1 and are not meaningful; use
    ``accuracy``/``correct``/``total`` and ``confusion_matrix`` instead.

    ``confusion_matrix`` is the full ``num_classes`` x ``num_classes`` chunk confusion matrix of
    the dataset, rows = true class and columns = predicted class (the same orientation as the
    pooled ``confusion_matrix`` of :func:`classification_metrics`). Its row sums are the
    per-true-class chunk counts, so a consumer can derive a dataset's class balance and the
    accuracy of an always-predict-the-majority-class baseline from it.

    :param np.ndarray y_true: True labels for each chunk.
    :param np.ndarray y_pred: Predicted labels for each chunk.
    :param list[str] subjects: Subject ID for each chunk.
    :param dict[str, str] subject_dataset_map: Mapping from subject ID to dataset label.
    :param int num_classes: Number of classes (2 or 4); sets the size of ``confusion_matrix``.
    :return: Per-dataset metrics, e.g. {"mdd": {"accuracy": 0.85, "sensitivity": 0.80,
        "specificity": 0.90, "tp": 400, "tn": 450, "fp": 50, "fn": 100, "correct": 850,
        "total": 1000, "confusion_matrix": array([[450, 50], [100, 400]])}}.
    :rtype: PerDatasetMetrics
    :raises ValueError: If a true or predicted label is outside ``[0, num_classes)``.
    """
    # The range is a property of the whole array, so check it once rather than per chunk.
    if len(y_true) and not (
        0 <= min(y_true.min(), y_pred.min()) and max(y_true.max(), y_pred.max()) < num_classes
    ):
        raise ValueError(
            f"Label out of range for num_classes={num_classes}: "
            f"true in [{y_true.min()}, {y_true.max()}], pred in [{y_pred.min()}, {y_pred.max()}]"
        )

    # The confusion matrix is the only accumulator: every other metric below is derived from it,
    # so the counts cannot drift apart from the matrix.
    dataset_cm: dict[str, NDArray[np.int64]] = {}
    for true, pred, subj in zip(y_true, y_pred, subjects):
        ds = subject_dataset_map.get(subj, "unknown")
        if ds not in dataset_cm:
            dataset_cm[ds] = np.zeros((num_classes, num_classes), dtype=np.int64)
        dataset_cm[ds][int(true), int(pred)] += 1

    result: PerDatasetMetrics = {}
    for ds in sorted(dataset_cm):
        cm = dataset_cm[ds]
        total = int(cm.sum())
        correct = int(np.trace(cm))
        # Binary components (label 1 = positive/pathological); for a multi-class run these cover
        # only the chunks whose true and predicted label are both 0 or 1 (see the docstring).
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        result[ds] = {
            "accuracy": correct / total if total > 0 else 0.0,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "correct": correct,
            "total": total,
            "confusion_matrix": cm,
        }
    return result


def write_per_dataset_table(
    writer: Callable[[str], Any],
    fold_per_dataset_metrics: list[PerDatasetMetrics],
) -> None:
    """
    Write aggregated per-dataset chunk accuracy table across all folds.

    When the metrics carry per-dataset confusion matrices, a "Per-Dataset Chunk Confusion
    Matrices" section follows the table (see :func:`_write_per_dataset_confusion_matrices`).

    :param Callable[[str], Any] writer: Function to write output.
    :param list fold_per_dataset_metrics: List of per-dataset metric dicts, one per fold.
    :return: None
    :rtype: None
    """
    if not fold_per_dataset_metrics:
        return

    # Aggregate counts across folds for each dataset
    agg_correct: dict[str, int] = {}
    agg_total: dict[str, int] = {}
    agg_tp: dict[str, int] = {}
    agg_tn: dict[str, int] = {}
    agg_fp: dict[str, int] = {}
    agg_fn: dict[str, int] = {}

    for fold_metrics in fold_per_dataset_metrics:
        for ds, metrics in fold_metrics.items():
            agg_correct[ds] = agg_correct.get(ds, 0) + int(metrics["correct"])
            agg_total[ds] = agg_total.get(ds, 0) + int(metrics["total"])
            agg_tp[ds] = agg_tp.get(ds, 0) + int(metrics.get("tp", 0))
            agg_tn[ds] = agg_tn.get(ds, 0) + int(metrics.get("tn", 0))
            agg_fp[ds] = agg_fp.get(ds, 0) + int(metrics.get("fp", 0))
            agg_fn[ds] = agg_fn.get(ds, 0) + int(metrics.get("fn", 0))

    datasets = sorted(agg_total.keys())
    if not datasets:
        return

    writer("\nPer-Dataset Chunk Metrics (aggregated across folds):\n")
    writer(
        f"  {'Dataset':<12} {'Accuracy':>10}  {'Sensitivity':>12}"
        f"  {'Specificity':>12}  {'Chunks':>8}\n"
    )
    writer(f"  {'─' * 12}  {'─' * 10}  {'─' * 12}  {'─' * 12}  {'─' * 8}\n")

    grand_correct = 0
    grand_total = 0
    grand_tp = 0
    grand_tn = 0
    grand_fp = 0
    grand_fn = 0
    for ds in datasets:
        total = agg_total[ds]
        correct = agg_correct[ds]
        tp = agg_tp.get(ds, 0)
        tn = agg_tn.get(ds, 0)
        fp = agg_fp.get(ds, 0)
        fn = agg_fn.get(ds, 0)
        acc = correct / total if total > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        writer(
            f"  {ds.upper():<12} {acc * 100:>9.2f}%  {sens * 100:>11.2f}%"
            f"  {spec * 100:>11.2f}%  {total:>8}\n"
        )
        grand_correct += correct
        grand_total += total
        grand_tp += tp
        grand_tn += tn
        grand_fp += fp
        grand_fn += fn

    writer(f"  {'─' * 12}  {'─' * 10}  {'─' * 12}  {'─' * 12}  {'─' * 8}\n")
    grand_acc = grand_correct / grand_total if grand_total > 0 else 0.0
    grand_sens = grand_tp / (grand_tp + grand_fn) if (grand_tp + grand_fn) > 0 else 0.0
    grand_spec = grand_tn / (grand_tn + grand_fp) if (grand_tn + grand_fp) > 0 else 0.0
    total_row = (
        f"  {'Total':<12} {grand_acc * 100:>9.2f}%  {grand_sens * 100:>11.2f}%"
        f"  {grand_spec * 100:>11.2f}%  {grand_total:>8}\n"
    )
    writer(total_row)
    _write_per_dataset_confusion_matrices(writer, fold_per_dataset_metrics)


def _write_per_dataset_confusion_matrices(
    writer: Callable[[str], Any],
    fold_per_dataset_metrics: list[PerDatasetMetrics],
) -> None:
    """
    Write each dataset's chunk confusion matrix, summed across folds.

    Written after (and separate from) the per-dataset accuracy table so parsers of that table
    are unaffected. Nothing is written for results that carry no ``confusion_matrix`` (runs from
    before it was stored). The dataset header lines have the form ``NAME (N chunks)`` and the
    matrix rows start with the class display name, which
    ``helpers-print/dataset_prior_baseline.py`` relies on; keep the two in sync.

    :param Callable[[str], Any] writer: Function to write output.
    :param list fold_per_dataset_metrics: List of per-dataset metric dicts, one per fold.
    :return: None
    :rtype: None
    """
    aggregated: dict[str, NDArray[np.int64]] = {}
    for fold_metrics in fold_per_dataset_metrics:
        for ds, metrics in fold_metrics.items():
            cm = metrics.get("confusion_matrix")
            if cm is None:
                continue
            matrix = np.asarray(cm, dtype=np.int64)
            aggregated[ds] = aggregated[ds] + matrix if ds in aggregated else matrix
    if not aggregated:
        return

    class_names = _class_names(next(iter(aggregated.values())).shape[0])
    writer(
        f"\n{PER_DATASET_CM_HEADER} (aggregated across folds; "
        "rows = actual, columns = predicted):\n"
    )
    for ds in sorted(aggregated):
        cm = aggregated[ds]
        writer(f"\n  {ds.upper()} ({int(cm.sum())} chunks)\n")
        _write_matrix_body(writer, cm, class_names, indent="  ")


def write_results(writer: Callable[[str], Any], cv_results: dict[str, list]) -> None:
    """
    Write cross-validation results summary.

    Formats and outputs chunk-level, subject-level, and combined metrics.
    For binary classification: shows accuracy, sensitivity, specificity.
    For multi-class: shows accuracy and per-class recall.
    All values formatted as percentages.
    Also displays aggregated confusion matrix summed across all folds.

    :param Callable[[str], Any] writer: Function to write output (e.g., logger.info or file.write).
    :param dict[str, list] cv_results: Cross-validation results containing:
        - fold_chunk_metrics: List of chunk-level metric dicts
        - fold_subject_metrics: List of subject-level metric dicts
        - fold_eval_combined_acc: List of combined accuracy values
        - fold_chunk_confusion_matrices: List of confusion matrices from each fold
    :return: None
    :rtype: None
    """
    chunk_metrics = cv_results.get("fold_chunk_metrics", [])
    subject_metrics = cv_results.get("fold_subject_metrics", [])
    fold_train_acc = cv_results.get("fold_train_acc", [])

    if fold_train_acc:
        writer(_format_metric_line("TRAIN Accuracy Mean (at best val epoch)", fold_train_acc))
    _write_metrics_block(writer, chunk_metrics, "CHUNK")
    _write_metrics_block(writer, subject_metrics, "SUBJECT")

    writer(
        _format_metric_line(
            "COMBINED Accuracy Mean (chunk×subject)", cv_results["fold_eval_combined_acc"]
        )
    )

    # Add aggregated confusion matrix
    chunk_cms = cv_results.get("fold_chunk_confusion_matrices", [])
    if chunk_cms and len(chunk_cms) > 0:
        num_classes = chunk_cms[0].shape[0]  # Infer from matrix shape
        _format_confusion_matrix(chunk_cms, num_classes, writer)

    # Add per-dataset breakdown if available
    fold_per_dataset = cv_results.get("fold_per_dataset_metrics", [])
    if fold_per_dataset:
        write_per_dataset_table(writer, fold_per_dataset)
