# TODO: Add Per-Dataset Metrics to Cross-Validation Results

## Objective
Add per-dataset accuracy breakdown to the CV results file to evaluate cross-dataset generalization performance.

## Background
When training on `--dataset all` (combined MDD, CANE, AX_MALIK), we need visibility into how well the model performs on each individual dataset. This helps identify if the model generalizes across datasets or overfits to dataset-specific patterns.

## Tasks

### 1. Rename Output File (Optional)
- Consider renaming `cv_results.txt` → `results.txt` (since CV context is obvious)
- Update reference in `main.py` where results are written

### 2. Add Per-Dataset Accuracy Table
Add a new section to the results file with this format:

```
Per-Dataset Chunk Accuracy:
┌───────────┬──────────┬────────────┐
│ Dataset   │ Accuracy │ Chunk Count│
├───────────┼──────────┼────────────┤
│ MDD       │  XX.XX%  │    NNNN    │
│ CANE      │  XX.XX%  │    NNNN    │
│ AX_MALIK  │  XX.XX%  │    NNNN    │
└───────────┴──────────┴────────────┘
```

### 3. Implementation Details
**Location**: `thesis/metrics.py` - modify `calculate_metrics()` function

**Requirements**:
- Accept dataset labels alongside predictions (need to track which chunks came from which dataset)
- Calculate accuracy separately for each dataset
- Handle missing datasets gracefully (e.g., single-dataset training)
- Include chunk counts for transparency
- Maintain backward compatibility with existing metrics

**Data Flow**:
1. During validation/test, track: `(chunk_pred, chunk_label, dataset_source)`
2. Pass dataset info to `calculate_metrics()`
3. Group by dataset and compute per-dataset accuracy
4. Format table and append to results output

### 4. Acceptance Criteria
- [ ] Per-dataset accuracy table appears in results file
- [ ] Accuracy calculated correctly for each dataset independently
- [ ] Works with `--dataset all` (multi-dataset training)
- [ ] Gracefully handles single-dataset training (shows single dataset row)
- [ ] Chunk counts are accurate and sum to total
- [ ] Does not break existing metrics (chunk/subject accuracy, precision, recall, etc.)

## Why This Matters
Cross-dataset training is only useful if the model actually generalizes. Per-dataset metrics reveal:
- Whether the model learns universal depression/anxiety markers or dataset-specific artifacts
- Which datasets the model struggles with (informing future data collection)
- If class imbalance across datasets affects performance