# Implementation Plan: In-Ear EEG Channel Option

## Overview

Add `--channel in-ear` option that creates a synthetic in-ear EEG signal by computing the bipolar derivation T8 - T7 (right temporal minus left temporal). This simulates IDUN-style in-ear EEG using conventional scalp electrodes, with 50% sign flip augmentation always applied to handle polarity ambiguity.

## Background

Based on research paper "Estimating cognitive workload using a commercial in-ear EEG headset":
- In-ear surrogate = bipolar derivation between temporal electrodes
- Paper used TP9/TP10, but T7/T8 available in our datasets
- Subtraction: T8 - T7 (right minus left, left as reference)
- Sign flip augmentation handles unknown polarity

## Files to Modify

### 1. `thesis/cli.py` (lines 34-44)
Add "in-ear" to channel choices.

**Current code:**
```python
preproc_group.add_argument(
    "--channel",
    type=str,
    choices=list(
        set(MDDDataset.CHANNEL_MAPPING.values()) | set(CANEDataset.CHANNEL_MAPPING.values())
    )
    + ["all"],
    default="all",
    ...
)
```

**Change:**
- Add "in-ear" to the choices list alongside "all"
- Update help text to explain in-ear creates T8-T7 bipolar derivation

### 2. `thesis/dataset.py` - `load_and_preprocess_edf_file()` (lines 37-88)

Add in-ear channel handling for MDD and AX_MALIK datasets.

**Insert after line 68 (after channel rename), before channel selection:**
```python
if channel == "in-ear":
    # Load both T7 and T8 for bipolar derivation
    raw = raw.pick(["T7", "T8"])
    data = raw.get_data()
    # Bipolar derivation: T8 - T7 (right minus left)
    inear_signal = data[1] - data[0]  # T8 is index 1, T7 is index 0 after pick
    data = inear_signal.reshape(1, -1)  # Shape: (1, n_samples)
    # NOTE: Sign flip is applied in SpectrogramDataset, not here (so it's fresh each epoch)
elif channel == "all":
    ...
```

**Key considerations:**
- Must pick T7 and T8 in correct order (T7 first for consistent indexing)
- Sign flip NOT applied here - it's in SpectrogramDataset for per-epoch augmentation
- Result is single-channel: shape (1, n_samples)

### 3. `thesis/dataset.py` - `load_and_preprocess_cane_raw_file()` (lines 592-710)

Add in-ear channel handling for CANE dataset.

**Insert after line 639 (after channel rename), before channel selection logic:**
```python
if channel == "in-ear":
    # Bipolar derivation: T8 - T7
    inear_signal = df["T8"].values - df["T7"].values
    # NOTE: Sign flip is applied in SpectrogramDataset, not here
    channels_to_use = ["in-ear"]
    processed_channels = [CANEDataset._process_single_channel(
        inear_signal, "in-ear", file_path, artifact_method, artifact_threshold, skip_artifact_removal
    )]
elif channel == "all":
    ...
```

**Alternative approach (cleaner):**
Process T7 and T8 individually first, then subtract after filtering:
- This preserves filter integrity on individual channels
- Subtract after both channels are preprocessed

### 4. `thesis/dataset.py` - Dataset class attributes

Add "in-ear" to channel names where needed.

**MDDDataset (around line 168-171):**
```python
self.channel_names = (
    MDD_CHANNEL_ORDER if channel == "all"
    else ["in-ear"] if channel == "in-ear"
    else ([channel] if channel else [])
)
```

**CANEDataset (around line 396-401):**
```python
if channel == "all":
    self.channel_names = CANE_CHANNEL_ORDER
elif channel == "in-ear":
    self.channel_names = ["in-ear"]
elif channel:
    self.channel_names = [channel]
```

**AX_MALIKDataset (around line 839-841):**
AX_MALIKDataset inherits from MDDDataset and uses `load_and_preprocess_edf_file()`. Need to update channel_names handling:
```python
self.channel_names = (
    AX_MALIK_CHANNEL_ORDER if channel == "all"
    else ["in-ear"] if channel == "in-ear"
    else ([channel] if channel else [])
)
```

### 5. `main.py` (lines 664-676 and 782)

**Cross-dataset channel handling (lines 664-676):**

When `--dataset all` is used, there's T7/T8 → T3/T4 mapping for MDD dataset. Need to handle "in-ear" specially:

```python
# Current logic converts T7→T3 for MDD
if channel in ["T7", "T8"]:
    channel = {"T7": "T3", "T8": "T4"}[channel]
```

**Change needed:** Skip conversion for "in-ear":
```python
if channel in ["T7", "T8"]:  # in-ear doesn't need conversion
    channel = {"T7": "T3", "T8": "T4"}[channel]
# "in-ear" passes through unchanged
```

The preprocessing functions will handle "in-ear" by loading T7/T8 internally.

**in_channels determination (line 782):**
```python
in_channels=8 if channel == "all" else 1,
```
**No change needed** - "in-ear" is not "all", so it correctly gets `in_channels=1`.

### 6. `CLAUDE.md`

Update documentation to explain in-ear channel option.

## Implementation Order

1. **CLI changes** (`thesis/cli.py`) - Add "in-ear" to choices
2. **MDD preprocessing** (`thesis/dataset.py:load_and_preprocess_edf_file`) - Handle in-ear
3. **CANE preprocessing** (`thesis/dataset.py:load_and_preprocess_cane_raw_file`) - Handle in-ear
4. **Dataset class attributes** - Update channel_names handling
5. **Documentation** - Update CLAUDE.md

## Testing

After implementation:
```bash
# Test with MDD dataset
poetry run python main.py train --test-mode \
  --channel in-ear \
  --batch-size 32 \
  --checkpoint-dir=experiments/test_inear \
  --dataset mdd \
  --condition ec

# Verify:
# - Model receives single-channel input (in_channels=1)
# - Spectrograms have shape (1, 129, 41)
# - Training runs without errors
```

## Design Decisions

1. **Subtraction order**: T8 - T7 (right minus left) per advisor recommendation
2. **Sign flip location**: Applied per-access in `SpectrogramDataset` (not at preprocessing/caching), so model sees both polarities during training
3. **Sign flip trigger**: Always applied at 50% probability when channel="in-ear", regardless of --augment-data flag
4. **Where to subtract**: Before filtering for MDD/AX_MALIK (on raw data), after CAR for CANE
5. **Output shape**: Single channel (1, samples), treated same as other single-channel modes

## Sign Flip Implementation Detail

The sign flip should NOT be in preprocessing (which is cached), but in `SpectrogramDataset.convert_to_spectrograms()` so it's applied fresh each epoch. This ensures the model sees both polarities during training.

**In `thesis/dataset.py` - `SpectrogramDataset` class:**

1. Add `channel` parameter to `__init__()` (line 942):
```python
def __init__(
    self,
    dataset: MDDDataset | CANEDataset,
    fs: float = MDDDataset.FS,
    ...
    channel: str = "all",  # NEW
):
    ...
    self.is_inear = (channel == "in-ear")
```

2. Pass `is_inear` to `convert_to_spectrograms()` call (line ~1073):
```python
spectrograms = SpectrogramDataset.convert_to_spectrograms(
    ...
    is_inear=self.is_inear,  # NEW
)
```

3. In `convert_to_spectrograms()` static method (around line 1016-1018):
```python
# Apply sign flip for in-ear (always 50% probability)
if is_inear and np.random.random() < 0.5:
    channel_data = -channel_data

# Apply augmentation to raw EEG before STFT
if augmentation is not None:
    channel_data = augmentation(channel_data)
```

4. Update `data_preparation.py` to pass channel to SpectrogramDataset (lines 93, 181-186):
```python
spec_dataset = SpectrogramDataset(dataset, fs=fs, augmentation=augmentation, channel=channel)
```