"""Regression tests for per-dataset metrics computed inside ``eval_epoch``.

``eval_epoch`` once passed ``(preds, labels)`` to
``compute_per_dataset_metrics(y_true, y_pred, ...)``. Accuracy is symmetric, so it stayed correct,
but per-dataset "sensitivity" silently became precision and "specificity" became NPV, and stored
``fp``/``fn`` were exchanged. These tests pin the call site with hand-calculated confusion counts
that would fail if the arguments were swapped.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from main import eval_epoch
from thesis.metrics import compute_per_dataset_metrics


class _FixedPredictionModel(nn.Module):
    """Model whose predicted class is read directly from a scalar input feature."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = x.reshape(x.shape[0], -1)[:, 0]
        return torch.stack([-feature, feature], dim=1)  # argmax == 1 iff feature > 0


class _ChunkDataset(Dataset):
    """Yields ``(input, label, subject)`` like ``SpectrogramDataset``."""

    def __init__(self, preds: list[int], labels: list[int], subjects: list[str]) -> None:
        self.preds = preds
        self.labels = labels
        self.subjects = subjects

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        feature = 1.0 if self.preds[idx] == 1 else -1.0
        return torch.tensor([feature]), self.labels[idx], self.subjects[idx]


# Dataset "a" (subject s1): true=[1,1,1,0] pred=[1,0,0,0] -> TP=1 FN=2 TN=1 FP=0
# Dataset "b" (subject s2): true=[0,0,1]   pred=[1,0,1]   -> TP=1 FN=0 TN=1 FP=1
_LABELS = [1, 1, 1, 0, 0, 0, 1]
_PREDS = [1, 0, 0, 0, 1, 0, 1]
_SUBJECTS = ["s1", "s1", "s1", "s1", "s2", "s2", "s2"]
_DATASET_MAP = {"s1": "a", "s2": "b"}


def _run_eval_epoch() -> dict:
    loader = DataLoader(_ChunkDataset(_PREDS, _LABELS, _SUBJECTS), batch_size=3)
    return eval_epoch(
        _FixedPredictionModel(),
        loader,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        subject_dataset_map=_DATASET_MAP,
    )


class TestEvalEpochPerDataset:
    def test_per_dataset_confusion_counts_use_true_labels(self) -> None:
        per_dataset = _run_eval_epoch()["per_dataset"]

        a, b = per_dataset["a"], per_dataset["b"]
        assert (a["tp"], a["fn"], a["tn"], a["fp"]) == (1, 2, 1, 0)
        assert (b["tp"], b["fn"], b["tn"], b["fp"]) == (1, 0, 1, 1)

    def test_per_dataset_sensitivity_and_specificity(self) -> None:
        per_dataset = _run_eval_epoch()["per_dataset"]

        assert per_dataset["a"]["sensitivity"] == 1 / 3
        assert per_dataset["a"]["specificity"] == 1.0
        assert per_dataset["b"]["sensitivity"] == 1.0
        assert per_dataset["b"]["specificity"] == 0.5

    def test_per_dataset_counts_sum_to_chunk_confusion_matrix(self) -> None:
        result = _run_eval_epoch()
        per_dataset = result["per_dataset"]
        cm = np.array(result["chunk"]["confusion_matrix"])  # rows: actual, cols: predicted

        assert sum(m["tn"] for m in per_dataset.values()) == cm[0, 0]
        assert sum(m["fp"] for m in per_dataset.values()) == cm[0, 1]
        assert sum(m["fn"] for m in per_dataset.values()) == cm[1, 0]
        assert sum(m["tp"] for m in per_dataset.values()) == cm[1, 1]

    def test_per_dataset_accuracy_unchanged(self) -> None:
        per_dataset = _run_eval_epoch()["per_dataset"]

        assert per_dataset["a"]["accuracy"] == 2 / 4
        assert per_dataset["b"]["accuracy"] == 2 / 3


class TestComputePerDatasetMetricsArgumentOrder:
    def test_positional_order_is_true_then_pred(self) -> None:
        y_true = np.array([1, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 0])

        metrics = compute_per_dataset_metrics(y_true, y_pred, ["s1"] * 4, {"s1": "a"})

        assert metrics["a"]["sensitivity"] == 1 / 3
        assert metrics["a"]["specificity"] == 1.0
