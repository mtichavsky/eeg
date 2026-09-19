"""Tests for the per-dataset confusion matrix stored by ``compute_per_dataset_metrics``.

The matrix has rows = true class and columns = predicted class, like the pooled chunk confusion
matrix. All expected values are counted by hand from the fixtures below.
"""

import re

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from main import eval_epoch
from thesis.metrics import (
    PER_DATASET_CM_HEADER,
    classification_metrics,
    compute_per_dataset_metrics,
    write_per_dataset_table,
)

# Binary fixture. Dataset "a" (s1): true=[1,1,1,0] pred=[1,0,0,0]; dataset "b" (s2):
# true=[0,0,1] pred=[1,0,1].
BIN_TRUE = np.array([1, 1, 1, 0, 0, 0, 1])
BIN_PRED = np.array([1, 0, 0, 0, 1, 0, 1])
BIN_SUBJECTS = ["s1", "s1", "s1", "s1", "s2", "s2", "s2"]
BIN_MAP = {"s1": "a", "s2": "b"}

# 4-class fixture with datasets that lack classes, like the real pooled data: "mdd" has only
# Healthy (0) and Depression (2) true chunks, "sad" only Healthy (0) and Anxiety (1), "cane" all
# classes but Depression. The model also predicts classes a dataset never has (e.g. Comorbid for
# mdd).
QUAD_TRUE = np.array([0, 0, 2, 2, 2, 0, 1, 1, 0, 0, 0, 1, 3, 3, 3, 0])
QUAD_PRED = np.array([0, 3, 2, 2, 0, 0, 1, 3, 0, 1, 0, 1, 3, 1, 3, 2])
QUAD_SUBJECTS = ["m1"] * 5 + ["s1"] * 4 + ["c1"] * 7
QUAD_MAP = {"m1": "mdd", "s1": "sad", "c1": "cane"}


class TestBinary:
    def test_matrix_rows_are_true_and_columns_predicted(self) -> None:
        result = compute_per_dataset_metrics(BIN_TRUE, BIN_PRED, BIN_SUBJECTS, BIN_MAP)

        # a: true 1 x3 (pred 1,0,0), true 0 x1 (pred 0) -> [[TN=1, FP=0], [FN=2, TP=1]]
        np.testing.assert_array_equal(result["a"]["confusion_matrix"], [[1, 0], [2, 1]])
        # b: true 0 x2 (pred 1,0), true 1 x1 (pred 1) -> [[TN=1, FP=1], [FN=0, TP=1]]
        np.testing.assert_array_equal(result["b"]["confusion_matrix"], [[1, 1], [0, 1]])

    def test_matrix_agrees_with_the_existing_binary_counts(self) -> None:
        result = compute_per_dataset_metrics(BIN_TRUE, BIN_PRED, BIN_SUBJECTS, BIN_MAP)

        for metrics in result.values():
            tn, fp, fn, tp = metrics["confusion_matrix"].ravel()
            assert (tp, tn, fp, fn) == (metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"])
            assert np.trace(metrics["confusion_matrix"]) == metrics["correct"]
            assert metrics["confusion_matrix"].sum() == metrics["total"]

    def test_existing_keys_are_all_still_present(self) -> None:
        result = compute_per_dataset_metrics(BIN_TRUE, BIN_PRED, BIN_SUBJECTS, BIN_MAP)

        expected = {"accuracy", "sensitivity", "specificity", "tp", "tn", "fp", "fn"}
        assert expected | {"correct", "total", "confusion_matrix"} == set(result["a"])

    def test_matrices_sum_to_the_pooled_chunk_matrix(self) -> None:
        result = compute_per_dataset_metrics(BIN_TRUE, BIN_PRED, BIN_SUBJECTS, BIN_MAP)
        pooled = classification_metrics(BIN_TRUE, BIN_PRED, num_classes=2)["confusion_matrix"]

        np.testing.assert_array_equal(sum(m["confusion_matrix"] for m in result.values()), pooled)

    def test_default_num_classes_is_binary(self) -> None:
        result = compute_per_dataset_metrics(BIN_TRUE, BIN_PRED, BIN_SUBJECTS, BIN_MAP)

        assert result["a"]["confusion_matrix"].shape == (2, 2)

    def test_unmapped_subject_lands_in_unknown(self) -> None:
        result = compute_per_dataset_metrics(BIN_TRUE, BIN_PRED, BIN_SUBJECTS, {"s1": "a"})

        np.testing.assert_array_equal(result["unknown"]["confusion_matrix"], [[1, 1], [0, 1]])


class TestFourClass:
    @staticmethod
    def _compute() -> dict:
        return compute_per_dataset_metrics(
            QUAD_TRUE, QUAD_PRED, QUAD_SUBJECTS, QUAD_MAP, num_classes=4
        )

    def test_full_matrices_by_hand(self) -> None:
        result = self._compute()

        # mdd: true [0,0,2,2,2] pred [0,3,2,2,0]
        np.testing.assert_array_equal(
            result["mdd"]["confusion_matrix"],
            [[1, 0, 0, 1], [0, 0, 0, 0], [1, 0, 2, 0], [0, 0, 0, 0]],
        )
        # sad: true [0,1,1,0] pred [0,1,3,0]
        np.testing.assert_array_equal(
            result["sad"]["confusion_matrix"],
            [[2, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        )
        # cane: true [0,0,1,3,3,3,0] pred [1,0,1,3,1,3,2]
        np.testing.assert_array_equal(
            result["cane"]["confusion_matrix"],
            [[1, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 1, 0, 2]],
        )

    def test_missing_classes_give_all_zero_rows_and_columns(self) -> None:
        result = self._compute()

        # mdd has no Anxiety/Comorbid true chunks, sad has no Depression/Comorbid true chunks.
        assert result["mdd"]["confusion_matrix"].sum(axis=1)[[1, 3]].tolist() == [0, 0]
        assert result["sad"]["confusion_matrix"].sum(axis=1)[[2, 3]].tolist() == [0, 0]
        assert result["mdd"]["confusion_matrix"].shape == (4, 4)

    def test_true_class_support_is_the_row_sum(self) -> None:
        result = self._compute()

        assert result["mdd"]["confusion_matrix"].sum(axis=1).tolist() == [2, 0, 3, 0]
        assert result["sad"]["confusion_matrix"].sum(axis=1).tolist() == [2, 2, 0, 0]
        assert result["cane"]["confusion_matrix"].sum(axis=1).tolist() == [3, 1, 0, 3]

    def test_accuracy_correct_total_are_consistent_with_the_matrix(self) -> None:
        for metrics in self._compute().values():
            matrix = metrics["confusion_matrix"]
            assert metrics["correct"] == np.trace(matrix)
            assert metrics["total"] == matrix.sum()
            assert metrics["accuracy"] == np.trace(matrix) / matrix.sum()

    def test_matrices_sum_to_the_pooled_chunk_matrix(self) -> None:
        result = self._compute()
        pooled = classification_metrics(QUAD_TRUE, QUAD_PRED, num_classes=4)["confusion_matrix"]

        np.testing.assert_array_equal(sum(m["confusion_matrix"] for m in result.values()), pooled)

    @pytest.mark.parametrize("bad", [(4, 0), (0, 4), (-1, 0)])
    def test_out_of_range_label_raises(self, bad: tuple[int, int]) -> None:
        with pytest.raises(ValueError, match="out of range"):
            compute_per_dataset_metrics(
                np.array([bad[0]]), np.array([bad[1]]), ["s1"], {"s1": "a"}, num_classes=4
            )

    def test_four_class_label_rejected_for_a_binary_run(self) -> None:
        with pytest.raises(ValueError, match="num_classes=2"):
            compute_per_dataset_metrics(np.array([2]), np.array([0]), ["s1"], {"s1": "a"})


class _OneHotFromFeature(nn.Module):
    """Model whose predicted class is the (integer) scalar input feature."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.one_hot(x.reshape(x.shape[0], -1)[:, 0].long(), 4).float()


class _QuadDataset(Dataset):
    def __len__(self) -> int:
        return len(QUAD_TRUE)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        return torch.tensor([float(QUAD_PRED[idx])]), int(QUAD_TRUE[idx]), QUAD_SUBJECTS[idx]


class TestEvalEpochFourClass:
    def test_eval_epoch_stores_per_dataset_matrices(self) -> None:
        result = eval_epoch(
            _OneHotFromFeature(),
            DataLoader(_QuadDataset(), batch_size=5),
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            num_classes=4,
            subject_dataset_map=QUAD_MAP,
        )

        expected = compute_per_dataset_metrics(
            QUAD_TRUE, QUAD_PRED, QUAD_SUBJECTS, QUAD_MAP, num_classes=4
        )
        for dataset, metrics in expected.items():
            np.testing.assert_array_equal(
                result["per_dataset"][dataset]["confusion_matrix"], metrics["confusion_matrix"]
            )
        assert (
            sum(m["confusion_matrix"] for m in result["per_dataset"].values())
            == result["chunk"]["confusion_matrix"]
        ).all()


class TestResultsTxtSection:
    @staticmethod
    def _write(fold_metrics: list[dict]) -> str:
        lines: list[str] = []
        write_per_dataset_table(lines.append, fold_metrics)
        return "".join(lines)

    def test_section_sums_matrices_over_folds(self) -> None:
        fold_1 = compute_per_dataset_metrics(
            QUAD_TRUE, QUAD_PRED, QUAD_SUBJECTS, QUAD_MAP, num_classes=4
        )
        text = self._write([fold_1, fold_1])

        assert PER_DATASET_CM_HEADER in text
        assert "MDD (10 chunks)" in text  # 2 folds x 5 chunks
        # mdd's Healthy row [1, 0, 0, 1] and Depression row [1, 0, 2, 0], doubled
        mdd_block = text.split("MDD (10 chunks)")[1].split("SAD (")[0]
        assert re.search(r"Healthy\s+2\s+0\s+0\s+2\s", mdd_block)
        assert re.search(r"Depression\s+2\s+0\s+4\s+0\s", mdd_block)

    def test_no_section_without_stored_matrices(self) -> None:
        legacy = {"mdd": {"tp": 1, "tn": 1, "fp": 1, "fn": 1, "correct": 2, "total": 4}}

        assert PER_DATASET_CM_HEADER not in self._write([legacy])

    def test_existing_table_is_unchanged_by_the_section(self) -> None:
        result = compute_per_dataset_metrics(BIN_TRUE, BIN_PRED, BIN_SUBJECTS, BIN_MAP)
        legacy = {
            k: {m: v for m, v in d.items() if m != "confusion_matrix"} for k, d in result.items()
        }

        with_matrices = self._write([result])
        without = self._write([legacy])
        assert with_matrices.startswith(without)
        assert with_matrices[len(without) :].startswith(f"\n{PER_DATASET_CM_HEADER}")
