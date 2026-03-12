# Plan: Add EEGNet as Raw-EEG Baseline Model

## Files to Modify

1. **`thesis/model.py`** — Add `EEGNet` class + `RAW_EEG_MODELS` set + registry entry
2. **`thesis/dataset.py`** — Add `RawChunkDataset` and `FlattenedRawChunkDataset`
3. **`thesis/model_factory.py`** — Add `n_samples` param, branch on `RAW_EEG_MODELS`
4. **`main.py`** — Detect raw EEG model, use raw dataset, pass `n_samples` to factory

## Implementation Steps

### Step 1: `thesis/model.py` — EEGNet class

```python
class EEGNet(nn.Module):
    def __init__(self, input_shape=None, in_channels=1, n_channels=None,
                 n_samples=2500, F1=8, D=2, dropout=0.5, num_classes=2, **kwargs):
        # n_channels falls back to in_channels for factory compatibility
        C = n_channels or in_channels
        F2 = D * F1
        super().__init__()
        # Block 1: temporal filter → depthwise spatial filter
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, n_samples // 2), padding=(0, n_samples // 4), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D * F1, (C, 1), groups=F1, bias=False),  # depthwise
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # Block 2: separable temporal conv
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),  # depthwise
            nn.Conv2d(F2, F2, 1, bias=False),  # pointwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        flat_size = F2 * (n_samples // 32)
        self.classifier = nn.Linear(flat_size, num_classes)

    def forward(self, x):
        # x: (B, C, T) raw EEG
        x = x.unsqueeze(1)          # (B, 1, C, T)
        x = self.block1(x)          # (B, F2, 1, T//4)
        x = self.block2(x)          # (B, F2, 1, T//32)
        x = x.flatten(1)
        return self.classifier(x)
```

Add after class definition:
```python
RAW_EEG_MODELS: set[str] = {"EEGNet"}
```

Add to `MODEL_REGISTRY`:
```python
"EEGNet": (EEGNet, 0),  # rnn_hidden unused
```

### Step 2: `thesis/dataset.py` — Raw chunk datasets

Add `RawChunkDataset` after `SpectrogramDataset`:
- Same constructor signature as `SpectrogramDataset` (accepts `dataset`, `channel`, `fs`)
- `__getitem__(i)` → call `dataset[i]`, return `(list_of_chunks, label, subject)` where each chunk is `(C, T)` tensor (already preprocessed by underlying dataset)
- `get_indices_for_subjects()` identical to `SpectrogramDataset`

Add `FlattenedRawChunkDataset` after `FlattenedSpectrogramDataset`:
- Mirrors `FlattenedSpectrogramDataset` exactly (index build, `label_mapping`, `get_indices_for_subjects`)
- No other changes; `collate_spectrograms` reused as-is

### Step 3: `thesis/model_factory.py`

- Import `RAW_EEG_MODELS` from `thesis.model`
- Add `n_samples: int | None = None` to `create_model()` signature
- Replace unconditional `model_class(input_shape=spec_shape, ...)` with:
```python
if model_name in RAW_EEG_MODELS:
    model = model_class(n_channels=in_channels, n_samples=n_samples,
                        dropout=dropout, num_classes=num_classes).to(device)
else:
    model = model_class(input_shape=spec_shape, in_channels=in_channels,
                        rnn_hidden=rnn_hidden, dropout=dropout, num_classes=num_classes).to(device)
```

### Step 4: `main.py`

- Import `RAW_EEG_MODELS`, `RawChunkDataset`, `FlattenedRawChunkDataset`
- After building per-fold `train_subjects`/`val_subjects`, check `model_name in RAW_EEG_MODELS`:
  - Compute `n_samples` from `train_subjects[0][0].shape[-1]` (first subject's first chunk)
  - Instantiate `RawChunkDataset` + `FlattenedRawChunkDataset` instead of spectrogram equivalents
  - Pass `n_samples=n_samples` to `create_model()`
  - Log warning if `--dataset cane` (500 Hz → 5000 samples, incompatible with MDD-trained EEGNet)
- Training/validation loop unchanged — same `collate_spectrograms`, same loss, same metrics

## Constraints
- CANE (500 Hz, 5000 samples per chunk) will mismatche with MDD (250 Hz, 2500). For `--dataset all` raise a clear error when model is in `RAW_EEG_MODELS`.
- `in_channels=8` with `--channel all` works correctly — depthwise kernel `(C=8, 1)` handles it.
- Depthwise max-norm constraint (‖w‖ ≤ 1) is optional for initial implementation.

## Verification
```bash
poetry run python main.py train --model EEGNet --dataset mdd --condition ec \
  --channel all --test-mode --n-folds 2 --checkpoint-dir experiments/eegnet_test
```
