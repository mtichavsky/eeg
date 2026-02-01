"""Classification metrics for EEG depression/anxiety detection."""

from typing import Any, Callable

import numpy as np
from sklearn.metrics import confusion_matrix


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2
) -> dict[str, float | int | np.ndarray]:
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
    metrics = {"accuracy": acc}

    if num_classes == 2:
        # Binary classification: compute traditional metrics
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

            if num_classes == 4:
                class_names = ["normal", "anxiety", "depression", "anxiety+depression"]
            else:
                raise Exception("Unexpected number of classes")

            class_name = (
                class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
            )

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
        for class_name in ["normal", "mdd", "anxious"]:
            if f"precision_{class_name}" in metrics:
                prec = metrics[f"precision_{class_name}"]
                rec = metrics[f"recall_{class_name}"]
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
        return {
            "accuracy": metrics["accuracy"],
            "recall_normal": metrics.get("recall_normal", 0.0),
            "recall_anxiety": metrics.get("recall_anxiety", 0.0),
            "recall_depression": metrics.get("recall_depression", 0.0),
            "recall_anxiety+depression": metrics.get("recall_anxiety+depression", 0.0),
        }
    else:
        raise ValueError(f"Unsupported num_classes: {num_classes}")


def _format_pct(value: float) -> str:
    """
    Format a decimal value as percentage with 2 decimal places.

    :param float value: Decimal value (e.g., 0.85).
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
    elif "recall_normal" in metrics[0]:
        # Extract all recall_* keys from first metrics dict
        class_keys = [k for k in metrics[0].keys() if k.startswith("recall_")]
        for key in class_keys:
            class_name = key.replace("recall_", "")
            values = _extract_metric_values(metrics, key)
            writer(_format_metric_line(f"{level} Recall ({class_name.capitalize()}) Mean", values))


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

    if num_classes == 2:
        class_names = ["Healthy", "Pathological"]
    elif num_classes == 4:
        class_names = ["Normal", "Anxiety", "Depression", "Anxiety+Depression"]
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    writer("\nAggregated Confusion Matrix (sum across all folds):\n")

    # Header row
    writer(" " * 15 + "".join(f"{name:>15} " for name in class_names) + "\n")
    # Separator
    writer("-" * (15 + 16 * num_classes) + "\n")
    # Data rows
    for i, row_name in enumerate(class_names):
        row_str = f"{row_name:>15} " + "".join(
            f"{int(aggregated_cm[i, j]):>15} " for j in range(num_classes)
        )
        writer(row_str + "\n")

    # Additional statistics
    writer("\n")
    total_predictions = np.sum(aggregated_cm)
    correct_predictions = np.trace(aggregated_cm)
    overall_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    writer(f"Total Predictions: {int(total_predictions)}\n")
    writer(f"Correct Predictions: {int(correct_predictions)}\n")
    writer(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)\n")


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
