# Plan: IDUN In-Ear EEG Dataset Loader

## Context

Currently, `--channel in-ear` creates a **synthetic** in-ear signal by computing a bipolar derivation (T8 - T7) from 8-channel EEG datasets (MDD, CANE, AX_MALIK). The IDUN dataset at `/home/milan/Documents/diplomka/IDUN_IN_EAR/` contains **real** in-ear EEG recordings from an IDUN device across 4 diagnostic classes (normals, anxiety, depression, comorbid) — matching CANE's class structure.

**Goal**: When `--channel in-ear` is specified, automatically replace CANE with IDUN data (real in-ear replaces synthetic surrogate). Works as a drop-in replacement everywhere CANE appears: `--dataset cane` and `--dataset all`.

## IDUN Dataset Characteristics

- **Format**: CSV files with `timestamp,ch1` columns (single-channel)
- **Sampling rate**: 250 Hz (same as MDD)
- **Organization**: `{class_dir}/{subject_id}/eeg_{subject_id}{condition}.csv`
- **Classes**: `normals/`, `anxiety/`, `depression/`, `comorbid/`
- **Conditions**: ec, eo, erp (we use ec/eo only)
- **Subjects**: 55 total (19 normals, 20 anxiety, 1 depression, 15 comorbid)
- **Quality files**: `quality_{subject_id}{condition}.csv` alongside EEG files
- **Edge cases**: Subjects 1001, 1002 have non-standard filenames (no condition suffix) — skip them
- **Case variation**: Filenames mix lowercase/uppercase conditions (ec/EC/Ec) — handle with case-insensitive regex

## Files to Modify

### 1. `thesis/dataset.py` — Add `IDUNDataset` class

Add `IDUN_DIR` constant and the `IDUNDataset` class. Follow CANEDataset's structure for consistency (same interface, same label maps).

```python
IDUN_DIR = Path("/home/milan/Documents/diplomka/IDUN_IN_EAR/")
```

**`IDUNDataset` class** (independent, like CANEDataset):
- `FS = 250` (same as MDD — uses default STFT params, no custom STFT constants needed)
- `CLASS_DIRECTORIES = ["normals", "anxiety", "depression", "comorbid"]`
- `LABEL_MAP = {"normals": "H", "anxiety": "AX", "depression": "DEP", "comorbid": "AXDEP"}`
- `LABEL_INT_MAP = {"normals": 0, "anxiety": 1, "depression": 2, "comorbid": 3}`
- Always single channel: `channel_names = ["in-ear"]`

**`_discover_files()`**: Traverse `{class_dir}/{subject_id}/eeg_{subject_id}{condition}.csv`
- Use case-insensitive regex: `eeg_{subject_id}(ec|eo|erp)\.csv`
- Skip erp condition
- Skip subjects with non-standard naming (1001, 1002 — files like `eeg_1130725001.csv`) — **log `logger.warning()` for each skipped file** with the reason
- Build subject IDs matching CANE format: `"{LABEL} S{num} {CONDITION}"` (e.g., `"H S0012 EC"`)

**`load_and_preprocess_idun_file()`** (static method):
1. Load CSV → extract `ch1` column
2. Load quality CSV if available → log quality stats, reject low-quality chunks
3. z-score normalize raw values
4. Detrend (linear)
5. Bandpass filter 1-70 Hz (scipy butter + filtfilt, using `fs=250`)
6. Notch filter 50 Hz (scipy iirnotch + filtfilt, using `fs=250`)
7. Chunk into 10s segments (2500 samples)
8. Per-chunk z-score normalization
9. Return tensor `(num_chunks, 1, 2500)`

**Quality check approach**: Quality values are 0-99 range (0 likely = not measured, 90+ = good percentage). For basic quality checking:
- Load quality CSV alongside EEG file
- Map quality timestamps to chunk time ranges
- Log quality statistics per file (mean, min, max)
- Reject chunks where quality drops below a configurable threshold (default: `quality_threshold=0` meaning reject chunks with quality=0 i.e. unmeasured, since 0 appears to mean "not measured" while 90+ means good signal)
- **Log `logger.warning()` for every skipped chunk** with reason (bad quality, file naming issue, etc.)
- Make threshold configurable via constructor parameter, easy to adjust once we better understand the metric

**Interface**: Matches CANEDataset exactly — `__getitem__` returns `{"eeg", "label", "subject", "condition", "channels"}`, plus `get_sorted_subjects()`, `get_statistics()`.

### 2. `thesis/data_preparation.py` — Add `prepare_idun_dataset()`

Import `IDUNDataset` and add `prepare_idun_dataset()`:
- Same signature as `prepare_cane_dataset()` (minus artifact removal params, plus quality_threshold)
- Uses `IDUNDataset` instead of `CANEDataset`
- Uses MDD's default STFT params (fs=250, nperseg=256, noverlap=192) — same spectrogram shape (129, 41)
- Uses `dataset_label="cane"` in `split_subjects_into_classes()` so that `get_datasets_for_fold()` works unchanged (IDUN occupies the "cane" slot)

### 3. `main.py` — Swap CANE for IDUN when in-ear

When `channel == "in-ear"`:
- Where `prepare_cane_dataset()` is called (both `dataset_type == "cane"` and `dataset_type == "all"` branches), call `prepare_idun_dataset()` instead
- Log that IDUN real in-ear data is being used instead of CANE synthetic
- Conditions: IDUN uses lowercase ec/eo (same as CANE), no conversion needed

### 4. `thesis/cli.py` — Update help text

Update `--channel in-ear` help text to mention that it uses real IDUN in-ear data instead of synthetic CANE derivation.

## What Stays Unchanged

- `get_datasets_for_fold()` — no changes needed (IDUN uses "cane" dataset_label)
- `create_balanced_folds()` — no changes needed
- `split_subjects_into_classes()` — no changes needed (uses same subject prefix format: "H ", "AX ", "DEP ", "AXDEP ")
- `SpectrogramDataset` — no changes needed (already handles in-ear sign flip)
- MDD and AX_MALIK datasets — still use synthetic T8-T7 when in-ear
- Model architecture — no changes (still receives single-channel input)

## Verification

1. **Quick smoke test**: `poetry run python main.py train --channel in-ear --dataset cane --condition ec --test-mode --checkpoint-dir experiments/test_idun`
   - Should load IDUN data instead of CANE
   - Should log "Using IDUN real in-ear dataset" or similar
   - Should produce spectrograms of shape (129, 41)
   - Should complete one training step without errors

2. **All-dataset test**: `poetry run python main.py train --channel in-ear --dataset all --condition ec --test-mode --checkpoint-dir experiments/test_idun_all`
   - Should load MDD (synthetic in-ear) + IDUN (real) + AX_MALIK (synthetic)
   - Should create balanced folds with subjects from all three

3. **Non-in-ear unchanged**: `poetry run python main.py train --channel all --dataset cane --condition ec --test-mode`
   - Should still use CANE (not IDUN)

4. **Code quality**: `make format`
