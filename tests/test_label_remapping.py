"""Tests for label remapping in multi-class classification.

These tests verify that dataset labels are correctly remapped when combining
datasets with different label conventions (e.g., MDD binary labels vs CANE
4-class labels).

Critical invariants:
- MDD in 4-class mode: labels must be {0, 2} (normal, depression)
- MDD in 2-class mode: labels must be {0, 1} (normal, pathological)
- No MDD chunks should ever have label 1 (anxiety) or 3 (comorbid) in 4-class mode
"""

import numpy as np
import pytest

from thesis.data_preparation import prepare_mdd_dataset


class TestMDDLabelRemapping:
    """Tests for MDD dataset label remapping in multi-class classification."""

    @pytest.fixture
    def rng(self):
        """Provide a fixed random state for reproducibility."""
        return np.random.RandomState(42)

    def test_mdd_binary_mode_labels(self, rng):
        """Verify MDD dataset uses labels {0, 1} in binary (2-class) mode.

        In binary mode, MDD should use its native labels:
        - 0: Healthy control
        - 1: Depression (MDD)
        """
        flat_dataset, _ = prepare_mdd_dataset(
            conditions=["EC"],
            channel="Fp1",
            rng=rng,
            test_mode=True,  # Fast test with minimal data
            num_classes=2,
            label_mapping=None,
        )

        # Sample all labels from the dataset
        labels = [flat_dataset[i][1] for i in range(len(flat_dataset))]
        unique_labels = set(labels)

        assert unique_labels.issubset({0, 1}), (
            f"MDD binary mode should only have labels 0 (normal) and 1 (depression). "
            f"Found: {unique_labels}"
        )

    def test_mdd_4class_mode_labels(self, rng):
        """Verify MDD dataset uses labels {0, 2} in 4-class mode.

        In 4-class mode, MDD labels must be remapped to align with CANE convention:
        - 0: Normal (healthy control)
        - 2: Depression (NOT 1, which is anxiety in 4-class convention)

        This is the critical test that prevents the bug where MDD depression
        chunks were mislabeled as anxiety (class 1) in the confusion matrix.
        """
        flat_dataset, _ = prepare_mdd_dataset(
            conditions=["EC"],
            channel="Fp1",
            rng=rng,
            test_mode=True,
            num_classes=4,
            label_mapping=None,  # Should trigger automatic {0: 0, 1: 2} remapping
        )

        # Sample all labels from the dataset
        labels = [flat_dataset[i][1] for i in range(len(flat_dataset))]
        unique_labels = set(labels)

        # Critical assertion: MDD should NEVER have labels 1 (anxiety) or 3 (comorbid)
        forbidden_labels = unique_labels & {1, 3}
        assert len(forbidden_labels) == 0, (
            f"MDD dataset in 4-class mode has forbidden labels {forbidden_labels}. "
            f"This indicates depression chunks are mislabeled as anxiety/comorbid. "
            f"All labels: {unique_labels}"
        )

        # Verify we only have the expected labels
        assert unique_labels.issubset({0, 2}), (
            f"MDD 4-class mode should only have labels 0 (normal) and 2 (depression). "
            f"Found: {unique_labels}"
        )

    def test_mdd_4class_has_both_classes(self, rng):
        """Verify MDD dataset contains both normal and depression classes in 4-class mode.

        This test ensures the remapping doesn't accidentally collapse both classes
        into one label.
        """
        flat_dataset, subject_classes = prepare_mdd_dataset(
            conditions=["EC"],
            channel="Fp1",
            rng=rng,
            test_mode=True,
            num_classes=4,
            label_mapping=None,
        )

        # Sample all labels
        labels = [flat_dataset[i][1] for i in range(len(flat_dataset))]
        unique_labels = set(labels)

        # In test_mode with MDD, we should have at least one healthy and one depressed subject
        # (assuming test_mode loads at least one file per class)
        if len(flat_dataset) > 10:  # Only check if we have enough data
            assert 0 in unique_labels, "Missing label 0 (normal) in MDD dataset"
            assert 2 in unique_labels, "Missing label 2 (depression) in MDD dataset"

    def test_mdd_label_counts_match_subject_counts(self, rng):
        """Verify chunk labels align with subject classes.

        The number of chunks with each label should match the subjects in
        the corresponding class (approximately, accounting for varying chunk counts).
        """
        flat_dataset, subject_classes = prepare_mdd_dataset(
            conditions=["EC"],
            channel="Fp1",
            rng=rng,
            test_mode=True,
            num_classes=4,
            label_mapping=None,
        )

        # Count chunks per label
        labels = [flat_dataset[i][1] for i in range(len(flat_dataset))]
        label_counts = {label: labels.count(label) for label in set(labels)}

        # Count subjects per class
        num_normal_subjects = len(subject_classes.normal)
        num_depression_subjects = len(subject_classes.depression)

        # If we have normal subjects, we should have label 0 chunks
        if num_normal_subjects > 0:
            assert 0 in label_counts, (
                f"Expected label 0 chunks for {num_normal_subjects} normal subjects, "
                f"but found no label 0 chunks. Label counts: {label_counts}"
            )

        # If we have depression subjects, we should have label 2 chunks
        if num_depression_subjects > 0:
            assert 2 in label_counts, (
                f"Expected label 2 chunks for {num_depression_subjects} depression subjects, "
                f"but found no label 2 chunks. Label counts: {label_counts}"
            )

    def test_mdd_explicit_mapping_overrides_auto_remap(self, rng):
        """Verify explicit label_mapping parameter overrides automatic remapping.

        When an explicit label_mapping is provided, it should be used instead of
        the automatic {0: 0, 1: 2} remapping, even in 4-class mode.
        """
        # Provide explicit binary mapping
        binary_mapping = {0: 0, 1: 1, 2: 1, 3: 1}

        flat_dataset, _ = prepare_mdd_dataset(
            conditions=["EC"],
            channel="Fp1",
            rng=rng,
            test_mode=True,
            num_classes=4,
            label_mapping=binary_mapping,  # Explicit mapping should override
        )

        # Sample labels
        labels = [flat_dataset[i][1] for i in range(len(flat_dataset))]
        unique_labels = set(labels)

        # With explicit binary mapping, we should get 0 and 1 (not 0 and 2)
        assert unique_labels.issubset({0, 1}), (
            f"With explicit binary mapping, expected labels {{0, 1}}, got: {unique_labels}"
        )

    def test_mdd_4class_no_anxiety_chunks_in_confusion_matrix(self, rng):
        """Regression test: Verify MDD chunks don't appear in anxiety class.

        This test specifically guards against the bug where MDD depression chunks
        were appearing as anxiety (class 1) in the confusion matrix because the
        label remapping wasn't applied.

        Before fix: MDD label 1 (depression) → not remapped → class 1 (anxiety in 4-class)
        After fix: MDD label 1 (depression) → remapped to 2 → class 2 (depression in 4-class)
        """
        flat_dataset, _ = prepare_mdd_dataset(
            conditions=["EC"],
            channel="Fp1",
            rng=rng,
            test_mode=True,
            num_classes=4,
            label_mapping=None,
        )

        # Get all chunks and their labels
        all_labels = [flat_dataset[i][1] for i in range(len(flat_dataset))]

        # Count chunks per class
        anxiety_chunks = sum(1 for label in all_labels if label == 1)
        comorbid_chunks = sum(1 for label in all_labels if label == 3)

        # MDD dataset should have ZERO anxiety-only or comorbid chunks
        assert anxiety_chunks == 0, (
            f"Found {anxiety_chunks} chunks with label 1 (anxiety) in MDD dataset. "
            f"This indicates the label remapping fix is not working. "
            f"MDD depression chunks are being mislabeled as anxiety."
        )

        assert comorbid_chunks == 0, (
            f"Found {comorbid_chunks} chunks with label 3 (comorbid) in MDD dataset. "
            f"MDD dataset should not have comorbid class."
        )


class TestLabelRemappingEdgeCases:
    """Edge case tests for label remapping logic."""

    @pytest.fixture
    def rng(self):
        """Provide a fixed random state for reproducibility."""
        return np.random.RandomState(42)

    def test_mdd_with_multiple_conditions(self, rng):
        """Verify label remapping works with multiple conditions (EC + EO)."""
        flat_dataset, _ = prepare_mdd_dataset(
            conditions=["EC", "EO"],  # Multiple conditions
            channel="Fp1",
            rng=rng,
            test_mode=True,
            num_classes=4,
            label_mapping=None,
        )

        labels = [flat_dataset[i][1] for i in range(len(flat_dataset))]
        unique_labels = set(labels)

        # Should still only have {0, 2} even with multiple conditions
        assert unique_labels.issubset({0, 2}), (
            f"MDD with multiple conditions should only have labels {{0, 2}}, got: {unique_labels}"
        )

    def test_mdd_with_multichannel(self, rng):
        """Verify label remapping works with multi-channel data."""
        flat_dataset, _ = prepare_mdd_dataset(
            conditions=["EC"],
            channel="all",  # Multi-channel
            rng=rng,
            test_mode=True,
            num_classes=4,
            label_mapping=None,
        )

        labels = [flat_dataset[i][1] for i in range(len(flat_dataset))]
        unique_labels = set(labels)

        # Should still only have {0, 2} with multi-channel
        assert unique_labels.issubset({0, 2}), (
            f"MDD multi-channel should only have labels {{0, 2}}, got: {unique_labels}"
        )
