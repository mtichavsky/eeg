"""Data integrity tests for classification metrics.

These tests verify that classification_metrics() and aggregate_subject_predictions()
compute correct values using hand-calculated expected results from known confusion matrices.
"""

import numpy as np
import pytest

from thesis.metrics import aggregate_subject_predictions, classification_metrics


class TestClassificationMetricsBinary:
    """Binary classification metric calculations with known confusion matrices."""

    def test_perfect_predictions(self):
        """100% accuracy should yield sensitivity=1.0, specificity=1.0.

        Confusion Matrix:     Predicted
                              0    1
               Actual 0   [  5    0 ]   TN=5, FP=0
                      1   [  0    5 ]   FN=0, TP=5
        """
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        metrics = classification_metrics(y_true, y_pred, num_classes=2)

        assert metrics["accuracy"] == 1.0
        assert metrics["recall"] == 1.0  # sensitivity
        assert metrics["specificity"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["tp"] == 5
        assert metrics["tn"] == 5
        assert metrics["fp"] == 0
        assert metrics["fn"] == 0

    def test_all_wrong_predictions(self):
        """0% accuracy should yield sensitivity=0.0, specificity=0.0.

        Confusion Matrix:     Predicted
                              0    1
               Actual 0   [  0    5 ]   TN=0, FP=5
                      1   [  5    0 ]   FN=5, TP=0
        """
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        metrics = classification_metrics(y_true, y_pred, num_classes=2)

        assert metrics["accuracy"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["specificity"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["tp"] == 0
        assert metrics["tn"] == 0
        assert metrics["fp"] == 5
        assert metrics["fn"] == 5

    def test_partial_errors_hand_calculated(self):
        """Verify metrics match hand-calculated values from known CM.

        y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # 4 class 0, 6 class 1
        y_pred = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]  # Some errors

        Confusion Matrix:     Predicted
                              0    1
               Actual 0   [  3    1 ]   TN=3, FP=1
                      1   [  2    4 ]   FN=2, TP=4

        accuracy = 7/10 = 0.7
        sensitivity = TP/(TP+FN) = 4/6 = 0.6667
        specificity = TN/(TN+FP) = 3/4 = 0.75
        precision = TP/(TP+FP) = 4/5 = 0.8
        """
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

        metrics = classification_metrics(y_true, y_pred, num_classes=2)

        assert metrics["accuracy"] == pytest.approx(0.7)
        assert metrics["recall"] == pytest.approx(4 / 6)  # 0.6667
        assert metrics["specificity"] == pytest.approx(0.75)
        assert metrics["precision"] == pytest.approx(0.8)
        assert metrics["tp"] == 4
        assert metrics["tn"] == 3
        assert metrics["fp"] == 1
        assert metrics["fn"] == 2

    def test_high_sensitivity_low_specificity(self):
        """Catches all positives but has false positives.

        y_true = [0, 0, 0, 0, 1, 1]
        y_pred = [1, 1, 0, 0, 1, 1]  # All class 1 correct, half class 0 wrong

        Confusion Matrix:     Predicted
                              0    1
               Actual 0   [  2    2 ]   TN=2, FP=2
                      1   [  0    2 ]   FN=0, TP=2

        sensitivity = 2/(2+0) = 1.0
        specificity = 2/(2+2) = 0.5
        """
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0, 1, 1])

        metrics = classification_metrics(y_true, y_pred, num_classes=2)

        assert metrics["accuracy"] == pytest.approx(4 / 6)
        assert metrics["recall"] == 1.0  # Perfect sensitivity
        assert metrics["specificity"] == 0.5  # Low specificity
        assert metrics["tp"] == 2
        assert metrics["fp"] == 2

    def test_high_specificity_low_sensitivity(self):
        """Catches all negatives but misses all positives.

        y_true = [0, 0, 0, 0, 1, 1]
        y_pred = [0, 0, 0, 0, 0, 0]  # All predicted as 0

        Confusion Matrix:     Predicted
                              0    1
               Actual 0   [  4    0 ]   TN=4, FP=0
                      1   [  2    0 ]   FN=2, TP=0

        sensitivity = 0/(0+2) = 0.0
        specificity = 4/(4+0) = 1.0
        """
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])

        metrics = classification_metrics(y_true, y_pred, num_classes=2)

        assert metrics["accuracy"] == pytest.approx(4 / 6)
        assert metrics["recall"] == 0.0  # No sensitivity (misses all positives)
        assert metrics["specificity"] == 1.0  # Perfect specificity
        assert metrics["tp"] == 0
        assert metrics["fn"] == 2

    def test_confusion_matrix_values_match_counts(self):
        """TP, TN, FP, FN match expected counts from explicit confusion matrix."""
        # Create a specific scenario: 3 TN, 2 FP, 1 FN, 4 TP
        # Class 0: predict 3 as 0 (TN), predict 2 as 1 (FP)
        # Class 1: predict 1 as 0 (FN), predict 4 as 1 (TP)
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1])

        metrics = classification_metrics(y_true, y_pred, num_classes=2)

        assert metrics["tn"] == 3
        assert metrics["fp"] == 2
        assert metrics["fn"] == 1
        assert metrics["tp"] == 4

        # Verify formulas use these counts correctly
        assert metrics["recall"] == pytest.approx(4 / (4 + 1))  # TP/(TP+FN)
        assert metrics["specificity"] == pytest.approx(3 / (3 + 2))  # TN/(TN+FP)
        assert metrics["precision"] == pytest.approx(4 / (4 + 2))  # TP/(TP+FP)


class TestClassificationMetricsMulticlass:
    """Multi-class (3-class) metric calculations."""

    def test_perfect_3class_predictions(self):
        """All classes predicted correctly gives recall=1.0 for each class."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])

        metrics = classification_metrics(y_true, y_pred, num_classes=3)

        assert metrics["accuracy"] == 1.0
        assert metrics["recall_normal"] == 1.0
        assert metrics["recall_mdd"] == 1.0
        assert metrics["recall_anxious"] == 1.0
        assert metrics["precision_normal"] == 1.0
        assert metrics["precision_mdd"] == 1.0
        assert metrics["precision_anxious"] == 1.0

    def test_partial_3class_errors_hand_calculated(self):
        """Verify per-class recall with known confusion matrix.

        y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        y_pred = [0, 0, 1, 1, 1, 2, 2, 0, 2]

        Confusion Matrix:
                      Predicted
                      0   1   2
        Actual 0  [   2   1   0 ]  -> recall_normal = 2/3
               1  [   0   2   1 ]  -> recall_mdd = 2/3
               2  [   1   0   2 ]  -> recall_anxious = 2/3

        accuracy = 6/9 = 0.6667
        """
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2])

        metrics = classification_metrics(y_true, y_pred, num_classes=3)

        assert metrics["accuracy"] == pytest.approx(6 / 9)
        assert metrics["recall_normal"] == pytest.approx(2 / 3)
        assert metrics["recall_mdd"] == pytest.approx(2 / 3)
        assert metrics["recall_anxious"] == pytest.approx(2 / 3)

    def test_confusion_matrix_shape_and_values(self):
        """Returned confusion matrix is 3x3 with correct values."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2])

        metrics = classification_metrics(y_true, y_pred, num_classes=3)

        cm = metrics["confusion_matrix"]
        assert cm.shape == (3, 3)
        # Row 0 (actual=0): 2 correct, 1 predicted as class 1
        assert cm[0, 0] == 2
        assert cm[0, 1] == 1
        assert cm[0, 2] == 0
        # Row 1 (actual=1): 2 correct, 1 predicted as class 2
        assert cm[1, 0] == 0
        assert cm[1, 1] == 2
        assert cm[1, 2] == 1
        # Row 2 (actual=2): 2 correct, 1 predicted as class 0
        assert cm[2, 0] == 1
        assert cm[2, 1] == 0
        assert cm[2, 2] == 2

    def test_one_class_never_predicted(self):
        """Handle case where one class is never predicted."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 0, 0, 0, 0])  # All predicted as class 0

        metrics = classification_metrics(y_true, y_pred, num_classes=3)

        assert metrics["accuracy"] == pytest.approx(2 / 6)
        assert metrics["recall_normal"] == 1.0  # Both class 0 correct
        assert metrics["recall_mdd"] == 0.0  # None of class 1 predicted
        assert metrics["recall_anxious"] == 0.0  # None of class 2 predicted


class TestAggregateSubjectPredictions:
    """Subject-level majority voting aggregation."""

    def test_clear_majority(self):
        """Majority vote with unambiguous winner for each subject."""
        # S1: 2 votes for 0, S2: 3 votes for 1, S3: 2 votes for 1
        chunk_preds = np.array([0, 0, 1, 1, 1, 1, 1])
        chunk_labels = np.array([0, 0, 1, 1, 1, 1, 1])
        subjects = ["S1", "S1", "S2", "S2", "S2", "S3", "S3"]

        subj_preds, subj_labels, subj_ids = aggregate_subject_predictions(
            chunk_preds, chunk_labels, subjects
        )

        # Order may vary due to dict iteration, so check by subject
        results = dict(zip(subj_ids, subj_preds))
        assert results["S1"] == 0
        assert results["S2"] == 1
        assert results["S3"] == 1

    def test_tie_breaks_to_lower_index(self):
        """Tie goes to lower class index (np.bincount.argmax behavior)."""
        chunk_preds = np.array([0, 1])
        chunk_labels = np.array([0, 0])
        subjects = ["S1", "S1"]

        subj_preds, subj_labels, subj_ids = aggregate_subject_predictions(
            chunk_preds, chunk_labels, subjects
        )

        # bincount([0,1]) = [1, 1], argmax returns 0 (first occurrence)
        assert subj_preds[0] == 0

    def test_labels_from_truth_not_predictions(self):
        """Subject labels come from ground truth, not from majority vote."""
        # All chunks predict 1, but ground truth is 0
        chunk_preds = np.array([1, 1, 1])
        chunk_labels = np.array([0, 0, 0])
        subjects = ["S1", "S1", "S1"]

        subj_preds, subj_labels, subj_ids = aggregate_subject_predictions(
            chunk_preds, chunk_labels, subjects
        )

        assert subj_preds[0] == 1  # Majority of predictions
        assert subj_labels[0] == 0  # Ground truth label

    def test_single_chunk_per_subject(self):
        """Subject with one chunk uses that prediction directly."""
        chunk_preds = np.array([1])
        chunk_labels = np.array([0])
        subjects = ["S1"]

        subj_preds, subj_labels, subj_ids = aggregate_subject_predictions(
            chunk_preds, chunk_labels, subjects
        )

        assert len(subj_preds) == 1
        assert subj_preds[0] == 1
        assert subj_labels[0] == 0
        assert subj_ids[0] == "S1"

    def test_multiple_subjects_different_chunk_counts(self):
        """Multiple subjects with varying numbers of chunks."""
        # S1: 1 chunk, S2: 5 chunks, S3: 2 chunks
        chunk_preds = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        chunk_labels = np.array([0, 1, 1, 1, 1, 1, 0, 0])
        subjects = ["S1", "S2", "S2", "S2", "S2", "S2", "S3", "S3"]

        subj_preds, subj_labels, subj_ids = aggregate_subject_predictions(
            chunk_preds, chunk_labels, subjects
        )

        results = dict(zip(subj_ids, subj_preds))
        labels = dict(zip(subj_ids, subj_labels))

        # S1: single chunk predicts 0
        assert results["S1"] == 0
        assert labels["S1"] == 0

        # S2: 3 votes for 1, 2 votes for 0 -> majority is 1
        assert results["S2"] == 1
        assert labels["S2"] == 1

        # S3: 1 vote for 0, 1 vote for 1 -> tie, argmax returns 0
        assert results["S3"] == 0
        assert labels["S3"] == 0

    def test_preserves_subject_order(self):
        """Subject IDs are returned in order of first appearance."""
        chunk_preds = np.array([0, 1, 0])
        chunk_labels = np.array([0, 1, 0])
        subjects = ["A", "B", "A"]

        _, _, subj_ids = aggregate_subject_predictions(chunk_preds, chunk_labels, subjects)

        # A appears first, then B
        assert list(subj_ids) == ["A", "B"]

    def test_multiclass_majority_voting(self):
        """Majority voting works for 3-class predictions."""
        # S1: votes [0, 0, 1] -> majority 0
        # S2: votes [2, 2, 2] -> majority 2
        chunk_preds = np.array([0, 0, 1, 2, 2, 2])
        chunk_labels = np.array([0, 0, 0, 2, 2, 2])
        subjects = ["S1", "S1", "S1", "S2", "S2", "S2"]

        subj_preds, _, _ = aggregate_subject_predictions(chunk_preds, chunk_labels, subjects)

        results = dict(zip(["S1", "S2"], subj_preds))
        assert results["S1"] == 0
        assert results["S2"] == 2
