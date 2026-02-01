# Add Aggregated Confusion Matrix to cv_results.txt

## Goal
Add an aggregated (summed) confusion matrix to cv_results.txt that shows per-chunk evaluation results across all folds from the best model of each fold.

## Current State Analysis

### Problem
Confusion matrices are computed during evaluation but not preserved in the final cv_results:
- **Binary classification (2-class)**: Confusion matrix is unpacked to TP/TN/FP/FN scalars in `classification_metrics()` but the matrix object itself is not stored
- **Multi-class (4-class)**: Confusion matrix IS stored in metrics dict but gets lost during `extract_classification_metrics()` which only extracts float values (accuracy, recall)
- `train_one_fold()` captures best model metrics via `extract_classification_metrics()`, which doesn't include confusion matrices
- Therefore, confusion matrices never make it into `cv_results` dict or the final output

### Data Flow
```
classification_metrics() computes CM
  → eval_epoch() gets chunk_metrics with CM data
  → train_one_fold() captures best metrics via extract_classification_metrics() [CM LOST HERE]
  → fold_result dict (no CM field)
  → cv_results dict (no CM field)
  → write_results() outputs to cv_results.txt (no CM)
```

## Implementation Plan

### 1. Capture Confusion Matrix in `train_one_fold()` (main.py ~line 362)

**Location**: After the best model is saved (currently lines 368-369)

**Action**: Add confusion matrix capture alongside `extract_classification_metrics()`

For **binary classification**:
- Reconstruct 2x2 confusion matrix from TP/TN/FP/FN scalars
- Create numpy array: `[[TN, FP], [FN, TP]]`
- Handle edge case: If all TP/TN/FP/FN are 0 (line 46-58 in metrics.py), create zero matrix

For **multi-class**:
- Extract the stored confusion matrix from `chunk_metrics["confusion_matrix"]`

**Code location**: `main.py:362-386` (inside the `if chunk_metrics["accuracy"] > best_chunk_acc:` block)

**New variable**: `best_chunk_confusion_matrix` (numpy array)

### 2. Add Confusion Matrix to fold_result Dict (main.py ~line 412)

**Location**: `train_one_fold()` return statement (currently lines 412-419)

**Action**: Add new field to returned dict:
```python
return {
    "eval_combined_acc": corr_combined_acc,
    "best_epoch": best_epoch,
    "final_epoch": epoch,
    "history": fold_history,
    "chunk_metrics": best_chunk_metrics,
    "subject_metrics": best_subject_metrics,
    "chunk_confusion_matrix": best_chunk_confusion_matrix,  # NEW FIELD
}
```

### 3. Store in cv_results Dict (main.py ~line 789)

**Location**: Where fold results are accumulated (currently lines 789-793)

**Action**: Add new list to cv_results initialization (line 600-606):
```python
cv_results: dict[str, list] = {
    "fold_eval_combined_acc": [],
    "fold_best_epoch": [],
    "fold_final_epoch": [],
    "fold_chunk_metrics": [],
    "fold_subject_metrics": [],
    "fold_chunk_confusion_matrices": [],  # NEW FIELD
}
```

And append to it after each fold (after line 793):
```python
cv_results["fold_chunk_confusion_matrices"].append(fold_result["chunk_confusion_matrix"])
```

### 4. Aggregate and Format in write_results() (thesis/metrics.py)

**Location**: End of `write_results()` function (currently line 312)

**Action**: Add new helper function and call it at the end

**New helper function** (add before `write_results()`):
```python
def _format_confusion_matrix(
    confusion_matrices: list[np.ndarray],
    num_classes: int,
    writer: Callable[[str], Any]
) -> None:
    """
    Aggregate confusion matrices across folds and format for output.

    :param list[np.ndarray] confusion_matrices: List of confusion matrices from each fold
    :param int num_classes: Number of classes (2 or 4)
    :param Callable writer: Function to write output
    """
    # Sum confusion matrices across folds
    aggregated_cm = np.sum(confusion_matrices, axis=0)

    # Class names
    if num_classes == 2:
        class_names = ["Healthy", "Pathological"]
    elif num_classes == 4:
        class_names = ["Normal", "Anxiety", "Depression", "Anxiety+Depression"]
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Format output
    writer("\nAggregated Confusion Matrix (sum across all folds):\n")

    # Header row
    header = "                "  # spacing for row labels
    for name in class_names:
        header += f"{name:>15} "
    writer(header + "\n")

    # Separator
    writer("-" * (15 + 16 * num_classes) + "\n")

    # Data rows
    for i, row_name in enumerate(class_names):
        row_str = f"{row_name:>15} "
        for j in range(num_classes):
            row_str += f"{int(aggregated_cm[i, j]):>15} "
        writer(row_str + "\n")

    # Additional statistics
    writer("\n")
    total_predictions = np.sum(aggregated_cm)
    correct_predictions = np.trace(aggregated_cm)
    overall_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    writer(f"Total Predictions: {int(total_predictions)}\n")
    writer(f"Correct Predictions: {int(correct_predictions)}\n")
    writer(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)\n")
```

**Call in write_results()** (add at end, after line 312):
```python
# Add aggregated confusion matrix
chunk_cms = cv_results.get("fold_chunk_confusion_matrices", [])
if chunk_cms and len(chunk_cms) > 0:
    num_classes = chunk_cms[0].shape[0]  # Infer from matrix shape
    _format_confusion_matrix(chunk_cms, num_classes, writer)
```

## Files to Modify

1. **main.py** (3 locations):
   - Line ~362-386: Capture confusion matrix when best model is saved
   - Line ~412-419: Add field to fold_result return dict
   - Line ~600-606: Add field to cv_results initialization
   - Line ~789-793: Append confusion matrix to cv_results

2. **thesis/metrics.py** (2 locations):
   - Before line 285: Add `_format_confusion_matrix()` helper function
   - Line ~312: Call helper function at end of `write_results()`

## Edge Cases to Handle

1. **Empty confusion matrix list**: Check `if chunk_cms and len(chunk_cms) > 0` before formatting
2. **Binary vs multi-class**: Detect from matrix shape or pass num_classes parameter
3. **Division by zero**: Check `if total_predictions > 0` before computing accuracy
4. **Uninitialized best model**: Ensure confusion matrix is only captured when best model is actually saved

## Expected Output Format

```
Aggregated Confusion Matrix (sum across all folds):
                        Healthy    Pathological
-----------------------------------------------
        Healthy             523              87
   Pathological              45             612

Total Predictions: 1267
Correct Predictions: 1135
Overall Accuracy: 0.8958 (89.58%)
```

## Testing Considerations

1. Test with binary classification (2-class mode)
2. Test with multi-class classification (4-class mode)
3. Verify matrix sums match expected validation set sizes
4. Verify overall accuracy matches the mean chunk accuracy from cross-validation

## Notes

- We're only capturing **chunk-level** confusion matrices (not subject-level) as requested
- Matrices are aggregated from the **best model** of each fold (highest chunk accuracy)
- The aggregation is a **sum** (not average) to show total prediction counts across all validation sets