# AX_MALIK Dataset Implementation Plan

**Date:** 2026-02-01
**Goal:** Add AX_MALIKDataset class to `thesis/dataset.py` with maximum code reuse from MDDDataset

## Dataset Overview

**AX_MALIK Dataset Characteristics:**
- **Location:** `~/Documents/diplomka/AX_MALIK/`
- **Structure:** `{ec|eo}/C{N}.edf` (21 subjects, 42 files total)
- **Format:** EDF files (same as MDD)
- **Sampling Rate:** 256 Hz (vs MDD's 250 Hz)
- **Duration:** 120 seconds per file
- **Channels:** 30 channels available (includes all required: Fp1, Fp2, C3, Cz, C4, T7, T8, O2)
- **Label:** All files are anxiety class (label_int=1, consistent with CANE anxiety)

## Implementation Strategy

**Approach:** Inheritance from MDDDataset
- Minimize code duplication
- Override only what differs (file discovery, FS, channel mapping)
- Reuse entire preprocessing pipeline

## Changes Required

### 1. Module-Level Constants

Add to the top of `thesis/dataset.py` alongside existing constants:

```python
CANE_DIR = Path("/home/milan/Documents/diplomka/CANE/")
MDD_DIR = Path("/home/milan/Documents/diplomka/MDD/")
AX_MALIK_DIR = Path("/home/milan/Documents/diplomka/AX_MALIK/")  # NEW

# ... existing channel order constants ...

# Add after CANE_CHANNEL_ORDER:
AX_MALIK_CHANNEL_ORDER = ["Fp1", "Fp2", "C3", "Cz", "C4", "T7", "T8", "O2"]
```

### 2. Modify MDDDataset for Inheritance Compatibility

**Change 1:** Update `load_and_preprocess_mdd_raw_file()` signature to accept `fs` parameter:

```python
@staticmethod
def load_and_preprocess_mdd_raw_file(
    file_path: Path,
    channel: str,
    fs: float,  # NEW: Dataset-specific sampling rate
) -> torch.Tensor:
    """
    Load an EDF file from disk, preprocess it, and return chunks.

    :param Path file_path: Path to the EDF file to preprocess.
    :param str channel: Channel to use: specific channel name or "all".
    :param float fs: Sampling frequency in Hz (e.g., 250 for MDD, 256 for AX_MALIK).
    :return: Tensor of preprocessed EEG chunks.
    :rtype: torch.Tensor
    """
    # Calculate chunk samples from passed fs, not class attribute
    chunk_samples = int(CHUNK_DURATION_SEC * fs)

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw = raw.filter(l_freq=1, h_freq=70, method="iir", verbose=False)
    raw = raw.notch_filter(freqs=50, verbose=False)

    # ... rest of preprocessing ...

    # Use chunk_samples variable instead of MDDDataset.CHUNK_SAMPLES
    for i in range(0, n_samples - chunk_samples + 1, chunk_samples):
        chunk = data[:, i : i + chunk_samples]
        # ...
```

**Change 2:** Update `__getitem__()` to pass `self.FS`:

```python
def __getitem__(self, idx: int) -> dict[str, Any]:
    file_info = self.files[idx]

    chunks = self._load_and_preprocess_mdd_raw_file_cached(
        file_info["path"],
        self.channel,
        self.FS  # NEW: Pass dataset-specific sampling rate
    )

    # ... rest unchanged ...
```

**Change 3:** Update cache setup in `__init__()` to wrap with correct signature:

```python
if cache_size:
    # Wrap to include self.FS in the call
    def _cached_loader(path, channel):
        return MDDDataset.load_and_preprocess_mdd_raw_file(path, channel, self.FS)

    self._load_and_preprocess_mdd_raw_file_cached = lru_cache(maxsize=cache_size)(
        _cached_loader
    )
else:
    self._load_and_preprocess_mdd_raw_file_cached = (
        lambda path, channel: MDDDataset.load_and_preprocess_mdd_raw_file(
            path, channel, self.FS
        )
    )
```

### 3. Implement AX_MALIKDataset Class

Add new class after CANEDataset in `thesis/dataset.py`:

```python
class AX_MALIKDataset(MDDDataset):
    """
    PyTorch Dataset for AX_MALIK EEG data (anxiety detection).

    Inherits preprocessing pipeline from MDDDataset. All subjects are anxiety class.
    """

    FS = 256  # Hz (vs MDD's 250 Hz)
    CHUNK_SAMPLES = int(CHUNK_DURATION_SEC * FS)  # 2560 samples per chunk

    CHANNEL_MAPPING = {
        # AX_MALIK uses standard 10-20 channel names - identity mapping
        "Fp1": "Fp1",
        "Fp2": "Fp2",
        "C3": "C3",
        "Cz": "Cz",
        "C4": "C4",
        "T7": "T7",
        "T8": "T8",
        "O2": "O2",
    }

    def __init__(
        self,
        data_dir: Path = AX_MALIK_DIR,
        condition: Optional[Literal["EC", "EO"]] = None,
        subjects: Optional[list[str]] = None,
        cache_size: Optional[int] = 100,
        transform: Optional[Callable] = None,
        skip_ica: bool = True,
        channel: str = "all",
    ):
        """
        Initialize the AX_MALIK EEG dataset.

        :param Path data_dir: Path to directory containing .edf files.
        :param Optional[Literal["EC", "EO"]] condition: Filter by condition or None for all.
        :param Optional[list[str]] subjects: List of subject IDs to include. None = all.
        :param int cache_size: Number of preprocessed files to cache in memory.
        :param Optional[Callable] transform: Optional transform function to apply to EEG data.
        :param bool skip_ica: If True, skip ICA artifact removal during preprocessing.
        :param str channel: Channel to use: specific channel name or "all" for all 8 channels.
        """
        self.data_dir = Path(data_dir)
        self.condition = condition
        self.transform = transform
        self.skip_ica = skip_ica
        self.files = self._discover_files(
            condition.upper() if condition is not None else None,
            subjects,
            None  # labels parameter ignored
        )

        # Handle channel selection
        self.channel = channel
        self.channel_names = (
            AX_MALIK_CHANNEL_ORDER if channel == "all" else ([channel] if channel else [])
        )

        # Set up caching (same pattern as MDDDataset)
        if cache_size:
            def _cached_loader(path, channel):
                return MDDDataset.load_and_preprocess_mdd_raw_file(path, channel, self.FS)

            self._load_and_preprocess_mdd_raw_file_cached = lru_cache(maxsize=cache_size)(
                _cached_loader
            )
        else:
            self._load_and_preprocess_mdd_raw_file_cached = (
                lambda path, channel: MDDDataset.load_and_preprocess_mdd_raw_file(
                    path, channel, self.FS
                )
            )

    def _discover_files(
        self,
        condition: Optional[Literal["EC", "EO"]],
        subjects: Optional[list[str]],
        labels: Optional[list[str]],  # Ignored - all are anxiety
    ) -> list[dict[str, Any]]:
        """
        Discover all .edf files matching the criteria.

        Files are organized in: {ec|eo}/C{N}.edf

        :param Optional[Literal["EC", "EO"]] condition: Condition filter or None.
        :param Optional[list[str]] subjects: List of subject IDs to include or None for all.
        :param Optional[list[str]] labels: Ignored (all subjects are anxiety).
        :return: List of dictionaries containing file metadata.
        :rtype: list[dict]
        """
        files = []
        pattern = re.compile(r"C(\d+)\.edf")

        for condition_dir in ["ec", "eo"]:
            # Filter by condition if specified
            if condition and condition_dir.upper() != condition:
                continue

            condition_path = self.data_dir / condition_dir
            if not condition_path.exists():
                continue

            for file_path in sorted(condition_path.glob("*.edf")):
                match = pattern.match(file_path.name)
                if not match:
                    logger.warning(f"Skipping file with unexpected name: {file_path.name}")
                    continue

                subject_num = match.group(1)
                subject_id = f"AX S{subject_num} {condition_dir.upper()}"

                # Filter by subjects if specified
                if subjects and subject_id not in subjects:
                    continue

                files.append({
                    "path": file_path,
                    "label": "AX",
                    "subject": subject_id,
                    "condition": condition_dir.upper(),
                    "label_int": 1,  # Anxiety class (consistent with CANE)
                })

        logger.info(f"Discovered {len(files)} AX_MALIK files")
        return files

    def get_statistics(self) -> dict[str, Any]:
        """
        Get dataset statistics.

        :return: Dictionary containing total files, anxiety file count,
                 conditions breakdown, and number of unique subjects.
        :rtype: dict[str, Any]
        """
        conditions: dict[str, int] = {}
        for f in self.files:
            conditions[f["condition"]] = conditions.get(f["condition"], 0) + 1

        return {
            "total_files": len(self.files),
            "ax_files": len(self.files),  # All are anxiety
            "conditions": conditions,
            "subjects": len(self.get_sorted_subjects()),
        }
```

## Integration with SpectrogramDataset

**No changes needed** to SpectrogramDataset wrapper - it will automatically work with AX_MALIKDataset because:

1. AX_MALIKDataset returns same dict structure from `__getitem__()`
2. Spectrogram generation uses dataset-specific STFT parameters to achieve same output shape
3. The 256 Hz vs 250 Hz difference is handled by tuning (nperseg, noverlap, fs) parameters

**STFT Parameter Tuning:**
When creating SpectrogramDataset with AX_MALIKDataset, pass `fs=256` to match the sampling rate:

```python
spec_dataset = SpectrogramDataset(
    ax_malik_dataset,
    fs=256,  # Match AX_MALIK sampling rate
    nperseg=256,
    noverlap=192,  # Tune to achieve same output shape as MDD
)
```

## Methods Inherited Without Override

The following MDDDataset methods work directly with AX_MALIKDataset:

- `__len__()` - Standard implementation
- `__getitem__()` - Inherited as-is (returns correct dict structure)
- `get_sorted_subjects()` - Works with "AX S{N} {EC|EO}" format
- `load_and_preprocess_mdd_raw_file()` - Static method reused completely

## Testing Checklist

After implementation, verify:

1. **File discovery:**
   ```python
   dataset = AX_MALIKDataset(condition="EC")
   print(dataset.get_statistics())
   # Should show 21 EC files
   ```

2. **Data loading:**
   ```python
   item = dataset[0]
   print(item["eeg"].shape)  # Should be (num_chunks, 8, 2560)
   print(item["label"])  # Should be 1 (anxiety)
   print(item["subject"])  # Should be "AX S{N} EC"
   ```

3. **Cross-validation compatibility:**
   ```python
   subjects = dataset.get_sorted_subjects()
   # Should return ["AX S1 EC", "AX S2 EC", ..., "AX S21 EC"]
   ```

4. **Spectrogram generation:**
   ```python
   spec_dataset = SpectrogramDataset(dataset, fs=256)
   spec_list, label, subject = spec_dataset[0]
   print(spec_list[0].shape)  # Should match MDD spectrogram shape
   ```

5. **Multi-dataset training:**
   Test combined training with MDD and/or CANE datasets to ensure label_int=1 consistency works.

## Key Design Decisions

1. **Label consistency:** AX_MALIK uses label_int=1 (anxiety) to match CANE, enabling potential combined anxiety training
2. **Inheritance over composition:** Maximizes code reuse, minimal override needed
3. **Sampling rate handling:** Base dataset preserves 256 Hz; SpectrogramDataset handles output shape normalization via STFT parameters
4. **Subject ID format:** "AX S{N} {EC|EO}" follows existing pattern for easy integration with CV splitting
5. **Channel mapping:** Identity mapping since AX_MALIK already uses standard 10-20 nomenclature

## Future Considerations

- If multi-dataset training with AX_MALIK, update `main.py` to support `--dataset ax_malik` or `--dataset both-anxiety`
- Consider updating `create_balanced_folds()` to handle AX_MALIK alongside MDD/CANE
- Document STFT parameter tuning for 256 Hz to achieve target spectrogram shape
