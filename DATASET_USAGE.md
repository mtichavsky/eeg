# MDD Dataset Usage Guide

## Summary of Changes

**Before**: Single function processing one hardcoded file
**After**: Complete PyTorch Dataset with HuggingFace-like API

## Quick Start

```python
from thesis.dataset import MDDDataset, create_cross_validation_splits

# For cross-validation (recommended for your use case)
for fold, (train_loader, val_loader, _, _) in enumerate(create_cross_validation_splits(
    condition="EC",       # or "EO", "TASK", or None for all
    n_folds=5,
    batch_size=32,
)):
    print(f"Training fold {fold + 1}/5")

    for batch in train_loader:
        eeg = batch['eeg']      # Shape: (batch_size, 6, 2500)
        labels = batch['label']  # Shape: (batch_size,) - 0=Healthy, 1=MDD
        subjects = batch['subject']
        conditions = batch['condition']

        # Your training code here
        loss = model(eeg, labels)
        # ...
```

## Key Features

### 1. MDDDataset Class
- **Automatic file discovery**: Finds all EDF files in MDD directory
- **Metadata extraction**: Parses labels (H/MDD), subjects, conditions from filenames
- **PyTorch compatible**: Works with DataLoader for batching
- **Flexible filtering**: By condition, subjects, or labels
- **Lazy loading (default)**: Files preprocessed on-demand, not at initialization
- **LRU caching**: Most recently used files kept in memory (configurable)
- **Optional preloading**: Set `preload=True` to preprocess everything upfront

### 2. Cross-Validation Support (NEW!)
- **Subject-level splits**: No data leakage between folds
- **Stratified**: Balanced H/MDD ratio in each fold
- **Easy iteration**: Simple for-loop over folds

### 3. Train/Test Split
- Also available if you prefer single split over CV
- Same subject-level splitting logic

## Dataset Statistics (EC condition only)
- **Total**: 1,621 chunks from 54 files
- **Healthy**: 28 files
- **MDD**: 26 files
- **Subjects**: 54 unique subjects
- **Chunk size**: 6 channels × 2,500 samples (10 seconds at 250 Hz)

## Preload vs Lazy Loading

**Lazy loading (default, `preload=False`):**
- **Fast initialization** (~1 second): Only reads file headers
- **Files preprocessed on-demand**: When first accessed during training
- **Lower memory usage**: Only caches recently used files (default: 10 files)
- **Best for**: Most use cases, especially with limited RAM

```python
dataset = MDDDataset(condition="EC", preload=False, cache_size=10)
# Takes ~1s to initialize, files preprocessed during training
```

**Preloading (`preload=True`):**
- **Slow initialization** (~5-10 minutes): Preprocesses all files upfront
- **Instant access**: All data in memory
- **High memory usage**: All preprocessed data stored
- **Best for**: When you have lots of RAM and want fastest training

```python
dataset = MDDDataset(condition="EC", preload=True)
# Takes ~5-10 minutes to initialize, but training is faster
```

**Performance comparison (EC condition, 54 files):**
- `preload=False`: Init 1s, first access 7.6s per file, cached <0.01s
- `preload=True`: Init ~300s, all accesses <0.01s

## Advanced Usage

### Filter by specific subjects
```python
dataset = MDDDataset(
    condition="EC",
    subjects=["H S1", "H S2", "MDD S1"],  # Only these subjects
)
```

### Use all conditions
```python
dataset = MDDDataset(condition=None)  # EC + EO + TASK
```

### Add custom transforms
```python
def normalize(eeg):
    return (eeg - eeg.mean()) / eeg.std()

dataset = MDDDataset(transform=normalize)
```

### Manual DataLoader creation
```python
from torch.utils.data import DataLoader

dataset = MDDDataset(condition="EC")
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

## Data Format

Each batch is a dictionary with:
- `eeg`: Tensor of shape `(batch_size, channels, samples)` where channels=6, samples=2500
- `label`: Integer tensor `(batch_size,)` - 0=Healthy, 1=MDD
- `subject`: List of subject IDs (e.g., "H S1", "MDD S15")
- `condition`: List of conditions (e.g., "EC", "EO", "TASK")

## Preprocessing Pipeline

Applied to each file:
1. Load EDF file
2. Bandpass filter 1-70 Hz
3. Notch filter at 50 Hz
4. Select 6 channels: Fp1, Fp2, C3, C4, O2, Cz
5. Chunk into 10-second segments
6. Z-score normalization
7. Convert to PyTorch tensors
