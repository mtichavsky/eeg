"""Per-epoch evaluation shared by the training loop and the explainability harness."""

from typing import Any, Sized, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from thesis.metrics import (
    PerDatasetMetrics,
    aggregate_subject_predictions,
    classification_metrics,
    compute_per_dataset_metrics,
)


def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 2,
    subject_dataset_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate model for one epoch, computing both chunk-level and subject-level metrics.

    :param nn.Module model: Model to evaluate.
    :param DataLoader dataloader: Validation data loader (expects Subset of SpectrogramDataset).
    :param nn.Module criterion: Loss function.
    :param torch.device device: Device to evaluate on.
    :param int num_classes: Number of classes (2 or 3).
    :param dict[str, str] | None subject_dataset_map: Optional mapping from subject ID to dataset
        label. When provided, per-dataset chunk accuracy is computed.
    :return: Dictionary containing chunk-level and subject-level metrics.
    :rtype: dict
    """
    model.eval()
    running_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_subjects: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            # Unpack batch: SpectrogramDataset returns (spectrogram, label, subject)
            xb, yb, subjects = batch
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            running_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(yb.detach().cpu().numpy())
            all_subjects.extend(subjects)  # subjects is a list of strings from the batch

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)

    # Chunk-level metrics
    chunk_metrics = classification_metrics(labels_arr, preds_arr, num_classes=num_classes)
    avg_loss = running_loss / len(cast(Sized, dataloader.dataset))

    # Subject-level metrics (majority voting)
    subject_preds, subject_labels, subject_ids = aggregate_subject_predictions(
        preds_arr, labels_arr, all_subjects
    )
    subject_metrics = classification_metrics(subject_labels, subject_preds, num_classes=num_classes)

    # Separate metrics by condition (EC vs EO)
    condition_metrics = {}
    for condition in ["EC", "EO"]:
        # Filter subjects by condition
        condition_mask = [condition in subj_id for subj_id in subject_ids]
        if any(condition_mask):
            cond_preds = subject_preds[condition_mask]
            cond_labels = subject_labels[condition_mask]
            condition_metrics[condition] = classification_metrics(
                cond_labels, cond_preds, num_classes=num_classes
            )

    # Per-dataset metrics (when mapping is available)
    per_dataset: PerDatasetMetrics | None = None
    if subject_dataset_map is not None:
        per_dataset = compute_per_dataset_metrics(
            y_true=labels_arr,
            y_pred=preds_arr,
            subjects=all_subjects,
            subject_dataset_map=subject_dataset_map,
            num_classes=num_classes,
        )

    return {
        "chunk": chunk_metrics,
        "subject": subject_metrics,
        "condition": condition_metrics,
        "loss": avg_loss,
        "per_dataset": per_dataset,
    }
