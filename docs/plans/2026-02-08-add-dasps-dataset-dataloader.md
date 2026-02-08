# Plan: Add DASPS Dataset Dataloader

## Context

Adding support for the DASPS (Database for Anxious States based on Psychological Stimulation) dataset. DASPS provides EEG recordings from 9 subjects (276 trials) with HAM-D depression scores, recorded on a 14-channel Emotiv EPOC headset at 128 Hz. This dataset differs significantly from existing ones: different channel layout, different file format (HDF5/NPZ), trial-based data instead of continuous recordings, and no EC/EO conditions.

### DASPS Dataset Characteristics
- **9 labeled subjects** (S01-S09), 276 total trials, 15-sec trials (1920 samples at 128 Hz)
- **14 Emotiv EPOC channels**: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
- **HAM-D depression scores**: 7.0-44.0 per trial, varies within subjects
- **Data source**: `DASPS+HAM labels/DASPS+HAM-labels.mat` (HDF5 format, loaded via h5py)
- **Already preprocessed**: FIR band-pass (4-45 Hz), artifact removal (AAR, BSS-CCA)

### Design Decisions (Resolved with User)
1. **Channels**: Map 14 DASPS channels → 8 canonical via approximate 10-20 positions
2. **Conditions**: Not applicable - all 276 trials treated as single pool, `--condition` flag ignored
3. **Labels**: Per-subject mean HAM-D → binary threshold (>=14 = depressed)
4. **Mild depression**: Exclude by default (configurable parameter to include as healthy class 0)

---

## Step 1: Add DASPSDataset Class

**File**: `thesis/dataset.py`

### 1a. Add constants (after line 23, alongside other dataset dirs)

```python
DASPS_DIR = Path("/home/milan/Documents/diplomka/DASPS/")
```

Add channel order constant (after line 34):
```python
DASPS_CHANNEL_ORDER = ["Fp1", "Fp2", "C3", "Cz", "C4", "T7", "T8", "O2"]
```

### 1b. Channel mapping strategy

DASPS Emotiv EPOC channels (by index) mapped to canonical 8:

| Index | EPOC Channel | Canonical | Quality |
|-------|-------------|-----------|---------|
| 0     | AF3         | Fp1       | Approximate (AF3 between Fp1 & F3) |
| 13    | AF4         | Fp2       | Approximate (AF4 between Fp2 & F4) |
| 3     | FC5         | C3        | Approximate (FC5 between F3 & C3) |
| 10    | FC6         | C4        | Approximate (FC6 between F4 & C4) |
| 4     | T7          | T7        | Exact |
| 9     | T8          | T8        | Exact |
| avg(2,11) | mean(F3,F4) | Cz  | Synthetic midline from F3+F4 average |
| 7     | O2          | O2        | Exact |

Implement as class constants:
```python
# Emotiv EPOC channel indices → canonical channel names
CHANNEL_INDEX_MAP: dict[int, str] = {
    0: "Fp1",   # AF3 → Fp1 (approximate)
    13: "Fp2",  # AF4 → Fp2 (approximate)
    3: "C3",    # FC5 → C3 (approximate)
    10: "C4",   # FC6 → C4 (approximate)
    4: "T7",    # T7 → T7 (exact)
    9: "T8",    # T8 → T8 (exact)
    7: "O2",    # O2 → O2 (exact)
}
# Cz: synthetic from mean(F3[idx=2], F4[idx=11])
CZ_SOURCE_INDICES: tuple[int, int] = (2, 11)  # F3 and F4
```

Also define a `CHANNEL_MAPPING` for CLI channel union compatibility:
```python
CHANNEL_MAPPING: dict[str, str] = {
    "AF3": "Fp1", "AF4": "Fp2", "FC5": "C3", "FC6": "C4",
    "T7": "T7", "T8": "T8", "O2": "O2",
    # Cz is synthetic (mean of F3, F4)
}
```

### 1c. DASPSDataset class implementation

Custom class inheriting from `Dataset` (NOT from MDDDataset - data format is fundamentally different).

**Key class constants**:
```python
FS: float = 128  # Hz
CHUNK_SAMPLES: int = int(CHUNK_DURATION_SEC * FS)  # = 1280
STFT_NPERSEG: int = 256
STFT_NOVERLAP: int = 224  # Produces (129, 41) spectrogram shape
HAM_MILD_THRESHOLD: float = 14.0  # HAM-D score below this is "mild"
```

**Constructor** `__init__(self, condition, channel, cache_size=100, test_mode=False, include_mild_as_healthy=False)`:
- `condition`: Ignored for DASPS (no EC/EO), kept for interface compatibility
- `channel`: "all" (8 mapped channels), specific channel name, or "in-ear"
- `include_mild_as_healthy`: If False (default), subjects with mean HAM-D < 14 are excluded. If True, they get label 0 (healthy).
- Loads data from `DASPS+HAM labels/DASPS+HAM-labels.mat` via h5py (reusing loading logic from `/home/milan/Documents/diplomka/DASPS/preprocess.py` `DASPSDataLoader.load_labels_and_data()`)
- Groups trials by subject, computes per-subject mean HAM-D
- Applies channel mapping (14 → 8) during loading
- Creates subject-level entries: each entry = all trials for one subject
- LRU caching pattern (like other datasets, though small enough to fit in memory)

**Data loading flow**:
1. Open `DASPS+HAM-labels.mat` with h5py
2. Dereference HDF5 object references for labels and trials (same approach as `preprocess.py`)
3. Get `eeg_data` (276, 1920, 14), `class_labels` (276,), `subject_ids` (276,), `ham_scores` (276,)
4. Group by subject_id → 9 subject groups
5. Per subject: compute mean HAM-D, assign binary label
6. Filter subjects based on `include_mild_as_healthy` setting
7. Apply channel mapping: select 7 indexed channels + compute synthetic Cz

**Channel extraction** (during preprocessing):
```python
# For each trial's EEG data (1920, 14):
mapped_data = np.zeros((1920, 8))
for idx, canonical in CHANNEL_INDEX_MAP.items():
    col = DASPS_CHANNEL_ORDER.index(canonical)
    mapped_data[:, col] = trial_data[:, idx]
# Synthetic Cz
cz_col = DASPS_CHANNEL_ORDER.index("Cz")
mapped_data[:, cz_col] = (trial_data[:, 2] + trial_data[:, 11]) / 2  # mean(F3, F4)
```

**Chunking**: Each 15-sec trial (1920 samples) → 1 chunk of 10 sec (1280 samples), discard remaining 5 sec.

**Z-score normalization**: Per chunk, per channel (consistent with other datasets).

**`__getitem__(idx)`**: Returns same dict format as MDDDataset:
```python
{"eeg": torch.Tensor,  # (num_chunks, channels, samples) where channels=8 or 1
 "label": int,          # 0 or 1 based on mean HAM-D
 "subject": str,        # e.g., "DEP S1" or "H S4"
 "condition": str,      # Always "ALL" for DASPS
 "channels": list[str]}
```

**Subject ID format**: `"{PREFIX} S{subject_id}"` where:
- `"DEP"` prefix for subjects with mean HAM-D >= 14 (depressed, label=1)
- `"H"` prefix for subjects with mean HAM-D < 14 when `include_mild_as_healthy=True` (healthy, label=0)
- This format works with existing `split_subjects_into_classes()` which checks for `"DEP "` and `"H "` prefixes

**`get_sorted_subjects()`**: Returns sorted list of subject IDs (with condition suffix "ALL").

**`get_statistics()`**: Returns dict with subject count breakdown.

**Test mode**: When `test_mode=True`, load only subject S01 (single subject for quick debugging).

---

## Step 2: Add prepare_dasps_dataset()

**File**: `thesis/data_preparation.py`

Custom prepare function (NOT using `_prepare_dataset_generic` because DASPS ignores conditions):

```python
def prepare_dasps_dataset(
    conditions: list[str],  # Ignored for DASPS, kept for interface consistency
    channel: str,
    rng: np.random.RandomState,
    augmentation: Callable | None = None,
    test_mode: bool = False,
    include_mild_as_healthy: bool = False,
) -> tuple[FlattenedSpectrogramDataset, SubjectClasses]:
```

**Steps**:
1. Instantiate `DASPSDataset(condition="ALL", channel=channel, test_mode=test_mode, include_mild_as_healthy=include_mild_as_healthy)`
2. Wrap with `SpectrogramDataset(dataset, fs=DASPSDataset.FS, nperseg=DASPSDataset.STFT_NPERSEG, noverlap=DASPSDataset.STFT_NOVERLAP, augmentation=augmentation, channel=channel)`
3. Wrap with `FlattenedSpectrogramDataset(spec_dataset)`
4. Assert spectrogram shape is (129, 41)
5. Call `split_subjects_into_classes(subjects, "dasps", rng)`
6. Return `(flat_dataset, subject_classes)`

Add import for `DASPSDataset` at top of file.

---

## Step 3: Update CLI

**File**: `thesis/cli.py`

### 3a. Add "dasps" to dataset choices (line 101)
```python
choices=["mdd", "cane", "dasps", "all"],
```

### 3b. Update help text (lines 102-105)
Add DASPS description to help string.

### 3c. Update channel union (lines 37-39)
Add `DASPSDataset.CHANNEL_MAPPING` to the channel choices union:
```python
set(MDDDataset.CHANNEL_MAPPING.values())
| set(CANEDataset.CHANNEL_MAPPING.values())
| set(DASPSDataset.CHANNEL_MAPPING.values())
```
(In practice, DASPS canonical names are a subset of existing ones, so this adds nothing new, but it's good for completeness.)

---

## Step 4: Integrate into Training

**File**: `main.py`

### 4a. Add imports
```python
from thesis.data_preparation import prepare_dasps_dataset
```

### 4b. Add `elif dataset_type == "dasps"` branch (after line 634)
```python
elif dataset_type == "dasps":
    flat_dataset, subject_classes = prepare_dasps_dataset(
        conditions, channel, rng,
        augmentation=augmentation, test_mode=test_mode,
    )
    normal, anxiety, depression, anxiety_depression = (
        subject_classes.normal, subject_classes.anxiety,
        subject_classes.depression, subject_classes.anxiety_depression,
    )
    logger.info(f"DASPS dataset: {len(normal)} normal, {len(depression)} depression subjects")
```

### 4c. Add DASPS to `--dataset all` mode (after line 679)
```python
# Load DASPS dataset
dasps_flat_dataset, dasps_subject_classes = prepare_dasps_dataset(
    conditions, channel, rng, augmentation=augmentation, test_mode=test_mode,
)
dasps_normal, dasps_anxiety, dasps_depression, dasps_anxiety_depression = (
    dasps_subject_classes.normal, dasps_subject_classes.anxiety,
    dasps_subject_classes.depression, dasps_subject_classes.anxiety_depression,
)
```

### 4d. Update `create_balanced_folds` call (lines 685-690)
Add DASPS subject lists to each class:
```python
create_balanced_folds(
    [mdd_normal, cane_normal, ax_malik_normal, dasps_normal],
    [mdd_anxiety, cane_anxiety, ax_malik_anxiety, dasps_anxiety],
    [mdd_depression, cane_depression, ax_malik_depression, dasps_depression],
    [..., dasps_anxiety_depression],
    n_folds,
)
```

### 4e. Update `get_datasets_for_fold` call and function signature
Add `dasps_flat_dataset` parameter to both the function signature in `data_preparation.py` and the call in `main.py`. Add DASPS index extraction block (following AX_MALIK pattern at lines 504-523).

### 4f. Initialize `dasps_flat_dataset = None` (alongside other datasets, line 596-601)

---

## Step 5: Add h5py Dependency

Verify h5py is available in the poetry environment. If not:
```bash
poetry add h5py
```

---

## Verification

1. **Smoke test**: `poetry run python main.py train --dataset dasps --test-mode --channel all --epochs 2 --n-folds 2`
2. **Spectrogram shape**: Verify (129, 41) assertion passes
3. **Channel mapping**: Check that 8-channel spectrograms are generated correctly
4. **Label assignment**: Log per-subject mean HAM-D and resulting labels
5. **Subject count**: Verify correct number of subjects after mild depression filtering
6. **Combined mode**: `poetry run python main.py train --dataset all --test-mode --channel all --epochs 2 --n-folds 2`

---

## Summary of Files Modified

| File | Changes |
|------|---------|
| `thesis/dataset.py` | Add `DASPSDataset` class (~150 lines), `DASPS_DIR`, `DASPS_CHANNEL_ORDER` constants |
| `thesis/data_preparation.py` | Add `prepare_dasps_dataset()` function, import DASPSDataset |
| `thesis/cli.py` | Add "dasps" to `--dataset` choices, update help text |
| `main.py` | Add DASPS branch in training, add to "all" mode, update `get_datasets_for_fold` call |
| `pyproject.toml` | Add h5py dependency (if not present) |
