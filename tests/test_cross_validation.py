"""Tests for cross-validation split logic."""

from thesis.data_preparation import create_balanced_folds, split_into_folds


class TestSplitIntoFolds:
    """Tests for split_into_folds function that groups subjects across conditions."""

    def test_keeps_subject_conditions_together(self):
        """Verify EC and EO recordings from same subject stay in same fold.

        Uses 10 subjects × 2 conditions = 20 recordings split into 5 folds.
        Random assignment would have (1/5)^10=0.0000001% chance of passing.
        """
        # 10 base subjects, each with EC and EO conditions
        subjects = []
        for i in range(1, 11):
            subjects.append(("mdd", f"H S{i} EC"))
            subjects.append(("mdd", f"H S{i} EO"))

        folds = split_into_folds(subjects, n_folds=5)

        # Build mapping: base_subject -> list of (fold_index, full_subject_id)
        subject_fold_assignments: dict[str, list[tuple[int, str]]] = {}
        for fold_idx, fold in enumerate(folds):
            for _, subject_id in fold:
                base = " ".join(subject_id.split()[:-1])  # "H S1"
                if base not in subject_fold_assignments:
                    subject_fold_assignments[base] = []
                subject_fold_assignments[base].append((fold_idx, subject_id))

        # Verify: each base subject has all recordings in exactly one fold
        for base, assignments in subject_fold_assignments.items():
            fold_indices = set(fold_idx for fold_idx, _ in assignments)
            assert len(fold_indices) == 1, (
                f"Subject {base} has recordings split across folds {fold_indices}. "
                f"Assignments: {assignments}"
            )
            # Also verify both conditions are present
            conditions = [subj.split()[-1] for _, subj in assignments]
            assert "EC" in conditions and "EO" in conditions, (
                f"Subject {base} missing conditions. Found: {conditions}"
            )

    def test_no_subject_leakage_across_folds(self):
        """Verify same subject never appears in multiple folds.

        Uses 15 subjects × 2 conditions = 30 recordings split into 5 folds.
        """
        subjects = []
        for i in range(1, 16):
            subjects.append(("mdd", f"H S{i} EC"))
            subjects.append(("mdd", f"H S{i} EO"))

        folds = split_into_folds(subjects, n_folds=5)

        # Extract base subject IDs (without condition) for each fold
        fold_base_subjects = []
        for fold in folds:
            base_subjects = set(" ".join(subj.split()[:-1]) for _, subj in fold)
            fold_base_subjects.append(base_subjects)

        # Check no base subject appears in multiple folds
        for i, fold1 in enumerate(fold_base_subjects):
            for j, fold2 in enumerate(fold_base_subjects):
                if i != j:
                    overlap = fold1 & fold2
                    assert len(overlap) == 0, f"Subject leakage between fold {i} and {j}: {overlap}"

    def test_all_subjects_included(self):
        """Verify all subjects are included in exactly one fold."""
        subjects = [
            ("mdd", "H S1 EC"),
            ("mdd", "H S1 EO"),
            ("mdd", "H S2 EC"),
            ("mdd", "MDD S1 EC"),
            ("mdd", "MDD S1 EO"),
        ]
        folds = split_into_folds(subjects, n_folds=2)

        # Flatten all folds
        all_fold_subjects = []
        for fold in folds:
            all_fold_subjects.extend(fold)

        # Check each original subject appears exactly once
        for subj in subjects:
            count = all_fold_subjects.count(subj)
            assert count == 1, f"Subject {subj} appears {count} times, expected 1"

    def test_balanced_distribution(self):
        """Verify subjects are distributed relatively evenly across folds."""
        subjects = [("mdd", f"H S{i} EC") for i in range(1, 11)]  # 10 subjects
        subjects += [("mdd", f"H S{i} EO") for i in range(1, 11)]  # Same 10 with EO

        folds = split_into_folds(subjects, n_folds=5)

        # Each fold should have roughly the same number of base subjects
        fold_sizes = [len(set(" ".join(subj.split()[:-1]) for _, subj in fold)) for fold in folds]

        # With 10 base subjects and 5 folds, each fold should have 2 base subjects
        # (which means 4 total recordings if each has EC+EO)
        assert all(size == 2 for size in fold_sizes), f"Uneven distribution: {fold_sizes}"

    def test_single_fold(self):
        """Test edge case with single fold (all data in one fold)."""
        subjects = [
            ("mdd", "H S1 EC"),
            ("mdd", "H S1 EO"),
        ]
        folds = split_into_folds(subjects, n_folds=1)

        assert len(folds) == 1
        assert len(folds[0]) == 2
        assert folds[0] == subjects

    def test_more_folds_than_subjects(self):
        """Test when n_folds > number of unique subjects."""
        subjects = [
            ("mdd", "H S1 EC"),
            ("mdd", "H S1 EO"),
        ]
        folds = split_into_folds(subjects, n_folds=5)

        # Should create 5 folds, but only 1 will have subjects
        non_empty_folds = [f for f in folds if len(f) > 0]
        assert len(non_empty_folds) == 1
        assert len(folds) == 5


class TestCreateBalancedFolds:
    """Tests for create_balanced_folds function used in multi-dataset training."""

    def test_both_datasets_in_each_fold(self):
        """Verify each fold contains subjects from both MDD and CANE datasets."""
        mdd_normal = [("mdd", "H S1 EC"), ("mdd", "H S1 EO"), ("mdd", "H S2 EC")]
        mdd_depressed = [("mdd", "MDD S1 EC"), ("mdd", "MDD S1 EO")]
        cane_normal = [("cane", "H S1 ec"), ("cane", "H S1 eo"), ("cane", "H S2 ec")]
        cane_depressed = []  # CANE doesn't have MDD labeled data
        cane_anxious = [("cane", "AX S1 ec"), ("cane", "AX S1 eo"), ("cane", "AX S2 ec")]

        normal_folds, anxious_folds, mdd_folds = create_balanced_folds(
            mdd_normal, mdd_depressed, cane_normal, cane_depressed, cane_anxious, n_folds=2
        )

        # Check that normal folds contain both MDD and CANE subjects
        for fold in normal_folds:
            datasets = set(dataset for dataset, _ in fold)
            # At least when we have subjects from both, they should appear
            # (empty folds possible if not enough subjects)
            if len(fold) > 0:
                assert "mdd" in datasets and "cane" in datasets

        # Check MDD folds contain MDD subjects
        for fold in mdd_folds:
            if len(fold) > 0:
                datasets = set(dataset for dataset, _ in fold)
                assert "mdd" in datasets

    def test_no_subject_leakage_across_class_folds(self):
        """Verify subjects don't leak between different class folds at same index."""
        mdd_normal = [("mdd", f"H S{i} EC") for i in range(1, 5)]
        mdd_depressed = [("mdd", f"MDD S{i} EC") for i in range(1, 5)]
        cane_normal = [("cane", f"H S{i} ec") for i in range(5, 9)]
        cane_depressed = []
        cane_anxious = [("cane", f"AX S{i} ec") for i in range(1, 5)]

        normal_folds, anxious_folds, mdd_folds = create_balanced_folds(
            mdd_normal, mdd_depressed, cane_normal, cane_depressed, cane_anxious, n_folds=2
        )

        # At each fold index, verify subjects don't overlap
        for i in range(2):
            normal_subjects = set(subj for _, subj in normal_folds[i])
            mdd_subjects = set(subj for _, subj in mdd_folds[i])
            anxious_subjects = set(subj for _, subj in anxious_folds[i])

            # Normal and MDD should not overlap
            assert len(normal_subjects & mdd_subjects) == 0
            # Normal and anxious should not overlap (different classes)
            assert len(normal_subjects & anxious_subjects) == 0
            # MDD and anxious should not overlap
            assert len(mdd_subjects & anxious_subjects) == 0

    def test_stratified_distribution(self):
        """Verify each fold maintains class balance."""
        mdd_normal = [("mdd", f"H S{i} EC") for i in range(1, 11)]  # 10 normal
        mdd_depressed = [("mdd", f"MDD S{i} EC") for i in range(1, 11)]  # 10 depressed
        cane_normal = [("cane", f"H S{i} ec") for i in range(11, 21)]  # 10 normal
        cane_depressed = []
        cane_anxious = [("cane", f"AX S{i} ec") for i in range(1, 11)]  # 10 anxious

        normal_folds, anxious_folds, mdd_folds = create_balanced_folds(
            mdd_normal, mdd_depressed, cane_normal, cane_depressed, cane_anxious, n_folds=5
        )

        # Each fold should have roughly equal number of normal subjects
        normal_sizes = [len(fold) for fold in normal_folds]
        assert all(size == 4 for size in normal_sizes), (
            f"Uneven normal distribution: {normal_sizes}"
        )

        # Each fold should have equal number of anxious subjects
        anxious_sizes = [len(fold) for fold in anxious_folds]
        assert all(size == 2 for size in anxious_sizes), (
            f"Uneven anxious distribution: {anxious_sizes}"
        )

        # Each fold should have equal number of MDD subjects
        mdd_sizes = [len(fold) for fold in mdd_folds]
        assert all(size == 2 for size in mdd_sizes), f"Uneven MDD distribution: {mdd_sizes}"


class TestCrossValidationIntegrity:
    """Integration tests for complete cross-validation workflow."""

    def test_train_val_split_completeness(self):
        """
        Verify that for each fold, train + val = all subjects.

        This simulates the actual CV loop logic.
        """
        subjects = [("mdd", f"H S{i} EC") for i in range(1, 11)]
        subjects += [("mdd", f"MDD S{i} EC") for i in range(11, 21)]

        folds = split_into_folds(subjects, n_folds=5)

        for i in range(5):
            # Validation fold is fold i
            val_subjects = set(folds[i])

            # Training folds are all others
            train_subjects = set()
            for j in range(5):
                if j != i:
                    train_subjects.update(folds[j])

            # Check they don't overlap
            assert len(val_subjects & train_subjects) == 0, f"Overlap in fold {i}"

            # Check they cover all subjects
            all_in_folds = val_subjects | train_subjects
            assert all_in_folds == set(subjects), f"Missing subjects in fold {i}"

    def test_fold_reproducibility(self):
        """Verify same input produces same folds (deterministic)."""
        subjects = [("mdd", f"H S{i} EC") for i in range(1, 6)]

        folds1 = split_into_folds(subjects.copy(), n_folds=3)
        folds2 = split_into_folds(subjects.copy(), n_folds=3)

        # Should be identical
        for i in range(3):
            assert folds1[i] == folds2[i], f"Non-deterministic fold {i}"
