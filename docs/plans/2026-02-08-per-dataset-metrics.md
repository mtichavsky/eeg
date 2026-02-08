# Plan: Add Per-Dataset Metrics to Cross-Validation Results

## Context

When training on `--dataset all` (combined MDD, CANE, AX_MALIK), there is no visibility into how well the model performs on each individual dataset. This feature adds a per-dataset accuracy breakdown to the CV results file.

**Key challenge:** Dataset source info exists during fold creation as `(dataset_label, subject_id)` tuples but is **lost** by the time batches reach the validation loop (batches only contain subject strings like `"H S1 EC"`). Healthy subjects `"H S*"` can't be distinguished between MDD and CANE by string alone, so we need to propagate the mapping explicitly.

**Approach:** Build a `subject_id → dataset_label` mapping from the fold creation data and pass it through the evaluation pipeline. This is the least invasive approach — no changes to dataset classes, collate functions, or batch data structures.

## Files to Modify

1. `thesis/data_preparation.py` — return subject→dataset mapping from `get_datasets_for_fold()`
2. `thesis/metrics.py` — add per-dataset metrics computation and formatting
3. `main.py` — thread the mapping through eval_epoch → train_one_fold → train_cross_validation, accumulate results, write to file

## Implementation Steps

### Step 1: `thesis/data_preparation.py` — Return subject→dataset mapping

Modify `get_datasets_for_fold()` to return a third value:

```python
def get_datasets_for_fold(...) -> tuple[ConcatDataset | Subset, ConcatDataset | Subset, dict[str, str]]:
```

Build the mapping from the existing `val_subjects_with_dataset`:
```python
val_subject_dataset_map = {subj: ds for ds, subj in val_subjects_with_dataset}
```

Return `(train_dataset, val_dataset, val_subject_dataset_map)`.

### Step 2: `thesis/metrics.py` — Add per-dataset metrics functions

Add new function `compute_per_dataset_metrics()`:
```python
def compute_per_dataset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: list[str],
    subject_dataset_map: dict[str, str],
) -> dict[str, dict[str, float | int]]:
```
- Groups chunks by dataset using the subject→dataset mapping
- For each dataset: computes accuracy and chunk count
- Returns `{"MDD": {"accuracy": 0.85, "chunk_count": 500}, ...}`

Add new function `write_per_dataset_table()`:
```python
def write_per_dataset_table(
    writer: Callable[[str], Any],
    fold_per_dataset_metrics: list[dict[str, dict[str, float | int]]],
) -> None:
```
- Aggregates per-dataset metrics across folds (total correct / total chunks for accuracy, sum for chunk counts)
- Formats a table with dataset name, accuracy, and chunk count
- Gracefully shows single row for single-dataset training

Update `write_results()` to accept and pass through `fold_per_dataset_metrics`.

### Step 3: `main.py` — Thread mapping through the pipeline

**`eval_epoch()`** — Add optional `subject_dataset_map` parameter:
- After computing overall metrics, call `compute_per_dataset_metrics()`
- Include result in return dict as `"per_dataset"` key
- When `subject_dataset_map` is `None`, skip per-dataset computation (backward compatible)

**`train_one_fold()`** — Add `val_subject_dataset_map` parameter:
- Pass to `eval_epoch()` on each validation call
- When best model is found, also capture `best_per_dataset_metrics`
- Include in fold return dict

**`train_cross_validation()`** — Extract mapping and accumulate:
- Unpack 3-tuple from `get_datasets_for_fold()`
- Pass `val_subject_dataset_map` to `train_one_fold()`
- Add `"fold_per_dataset_metrics"` to `cv_results` dict
- Append each fold's per-dataset metrics

**`train()` (results writing)** — Pass through to `write_results()` (which now handles per-dataset table).

### Step 4: Rename output file (optional)

Rename `cv_results.txt` → `results.txt` in `train()` function.

## Expected Output Format

Added to the results file after the confusion matrix:

```
Per-Dataset Chunk Accuracy (aggregated across folds):
  Dataset     Accuracy  Chunk Count
  ─────────   ────────  ───────────
  MDD          85.00%         1200
  CANE         78.50%          800
  AX_MALIK     72.30%          400
  ─────────   ────────  ───────────
  Total        80.42%         2400
```

For single-dataset training, shows one row.

## Verification

1. Run with `--dataset all --test-mode` — verify per-dataset table appears with all 3 datasets
2. Run with `--dataset mdd --test-mode` — verify single-row table (graceful degradation)
3. Verify chunk counts sum to total
4. Verify existing metrics (accuracy, sensitivity, specificity, confusion matrix) are unchanged
5. Run `make format` to ensure code passes linting