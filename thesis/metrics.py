"""Classification metrics for EEG depression/anxiety detection."""

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
                    "tp": int(tp),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                }
            )
        else:
            # Handle edge case where only one class is predicted
            metrics.update(
                {
                    "precision": 0.0,
                    "recall": 0.0,
                    "specificity": 0.0,
                    "tp": 0,
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
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

            class_names = ["normal", "mdd", "anxious"]
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
