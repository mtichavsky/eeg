# Plan: EEG Augmentation Refactor

## Goal

Remove poorly-justified augmentations from `EEGAugmentation` and add two
high-potential ones: **Gaussian noise** (raw EEG) and **SpecAugment**
(spectrogram-domain). Keep **magnitude warping** with tightened defaults.

## Motivation

Current `EEGAugmentation` has four transforms, three of which are hard to defend
for resting-state mental health classification:

| Transform | Problem |
|---|---|
| Scaling (0.8–1.2×) | Z-score normalization already removes global amplitude — effect is negligible |
| FT surrogate | Randomizes phases → scrambles temporal structure in the STFT spectrogram; clinical temporal dynamics may be disrupted |
| Time reversal | Flips the spectrogram left-to-right; temporal evolution within 10s windows may carry diagnostic signal |
| Magnitude warping | Defensible — simulates slow within-window amplitude drift not removed by z-score normalization. Keep with σ=0.1 |

Replacements:

**Gaussian noise (raw EEG, before STFT):** Electrode-skin contact quality
varies across subjects and sessions, producing additive measurement noise.
Even after z-score normalization the relative noise floor differs per recording.
Small Gaussian noise (σ=0.02–0.05 of signal std) simulates this. Clearly valid,
very low risk of corrupting the clinical signal.

**SpecAugment (spectrogram domain, after STFT):** Randomly zero out a
contiguous strip of time frames and/or frequency bins. Argument: the model
should not rely on a single time step or narrow band. If the model has learned
an artifact localized to one spectrogram region, masking breaks that.
Applied *after* STFT → raw EEG is untouched. Well-established in speech
recognition; used in several EEG classification papers.

- how large this strip will be? The spectogram is split into 10 second chunks, it cannot be too large so that the model
  still works within the chunk

## Files to Modify

- **`thesis/augmentation.py`** — refactor `EEGAugmentation`, add `SpecAugment`
- **`thesis/dataset.py`** — accept optional `spec_augmentation` in `SpectrogramDataset`
- **`thesis/cli.py`** — add `--spec-augment` flag
- **`main.py`** — wire `--spec-augment` through to `SpectrogramDataset`

## Implementation

### 1. `thesis/augmentation.py`

#### Refactored `EEGAugmentation`

Remove `use_ft_surrogate`, `use_time_reverse`, `use_scaling` and their
implementations. Keep `use_mag_warp` with tighter default sigma (0.1 → 0.1
already fine). Add `use_gaussian_noise` with configurable `noise_std`.

```python
class EEGAugmentation:
    """
    EEG augmentation pipeline applied to raw EEG before STFT conversion.

    Applies augmentations independently with probability p_aug.
    Only includes transforms with clear justification for resting-state EEG.
    """

    def __init__(
        self,
        p_aug: float = 0.5,
        use_gaussian_noise: bool = True,
        noise_std: float = 0.03,
        use_mag_warp: bool = True,
        mag_warp_sigma: float = 0.1,
        mag_warp_knots: int = 4,
    ) -> None:
        """
        :param float p_aug: Probability of applying each augmentation (default 0.5).
        :param bool use_gaussian_noise: Add Gaussian noise to simulate electrode noise.
        :param float noise_std: Std of additive Gaussian noise relative to signal std.
        :param bool use_mag_warp: Smooth amplitude modulation simulating within-window drift.
        :param float mag_warp_sigma: Sigma for magnitude warping curve (default 0.1).
        :param int mag_warp_knots: Cubic spline knots for warping curve (default 4).
        """
```

Remove methods: `_ft_surrogate`, `_time_reverse`, `_scaling`.
Add method: `_gaussian_noise`.

```python
    def _gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian noise scaled to per-channel signal std."""
        std = np.std(x, axis=-1, keepdims=True)
        noise = np.random.normal(0, self.noise_std * std, x.shape)
        return x + noise
```

Keep `_mag_warp` unchanged.

Updated `__call__`:
```python
    def __call__(self, x: np.ndarray) -> np.ndarray:
        is_1d = x.ndim == 1
        if is_1d:
            x = x.reshape(1, -1)
        if self.use_gaussian_noise and np.random.random() < self.p_aug:
            x = self._gaussian_noise(x)
        if self.use_mag_warp and np.random.random() < self.p_aug:
            x = self._mag_warp(x)
        return x.squeeze() if is_1d else x
```

#### New `SpecAugment` class

New standalone class; applied to the spectrogram tensor *after* STFT.

```python
class SpecAugment:
    """
    SpecAugment-style masking applied to EEG spectrograms.

    Randomly zeros out contiguous time and/or frequency strips. Forces the model
    to use distributed time-frequency features rather than relying on a single
    spectrogram region.

    Applied to spectrogram tensors of shape (C, F, T) after STFT conversion.
    """

    def __init__(
        self,
        p_aug: float = 0.5,
        max_time_mask: int = 5,
        max_freq_mask: int = 10,
        n_time_masks: int = 1,
        n_freq_masks: int = 1,
    ) -> None:
        """
        :param float p_aug: Probability of applying each mask type (default 0.5).
        :param int max_time_mask: Max width of time mask in frames (default 5, ~1s at 41 frames).
        :param int max_freq_mask: Max height of frequency mask in bins (default 10, ~5 Hz at 129 bins).
        :param int n_time_masks: Number of time masks to apply (default 1).
        :param int n_freq_masks: Number of frequency masks to apply (default 1).
        """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time and frequency masking.

        :param torch.Tensor x: Spectrogram of shape (C, F, T).
        :return: Masked spectrogram of same shape.
        :rtype: torch.Tensor
        """
        # time masking: zero columns in the T dimension
        # freq masking: zero rows in the F dimension
        # Applied independently per mask with probability p_aug
```

### 2. `thesis/dataset.py` — `SpectrogramDataset`

Add `spec_augmentation` parameter (applied to spectrogram tensors after STFT,
training only). The existing `augmentation` parameter remains for raw EEG.

```python
def __init__(
    self,
    ...
    augmentation: Optional[Callable] = None,       # raw EEG, existing
    spec_augmentation: Optional[Callable] = None,  # spectrogram, new
    channel: str = "all",
):
```

In `convert_to_spectrograms` (or `_get_item`): after building each spectrogram
tensor, apply `spec_augmentation` if set. Caching must be disabled when either
augmentation is active (already handled by the existing `if augmentation is None`
check — extend to `if augmentation is None and spec_augmentation is None`).

Apply `spec_augmentation` inside `_get_item` at the per-chunk level (after STFT,
before returning the tensor) rather than inside `convert_to_spectrograms` — that
way it gets fresh randomness on every `__getitem__` call.

### 3. `thesis/cli.py`

Add argument:
```python
parser.add_argument(
    "--spec-augment",
    type=float,
    default=None,
    metavar="P",
    help="Enable SpecAugment on spectrograms with given probability (e.g. 0.5).",
)
```

### 4. `main.py`

Wire it up alongside the existing `--augment-data` handling:

```python
if args.spec_augment is not None:
    spec_augmentation: SpecAugment | None = SpecAugment(p_aug=args.spec_augment)
    logger.info(f"  SpecAugment: enabled (p={args.spec_augment})")
else:
    spec_augmentation = None
    logger.info("  SpecAugment: disabled")
```

Pass through `train_cv()` → `SpectrogramDataset` construction (same pattern as
existing `augmentation`). Log to `results.txt`.

## Verification

**Smoke test:**
```bash
poetry run python -c "
import numpy as np
import torch
from thesis.augmentation import EEGAugmentation, SpecAugment

# Raw EEG augmentation
aug = EEGAugmentation(p_aug=1.0)
x = np.random.randn(2500)
out = aug(x)
assert out.shape == x.shape, 'shape mismatch'
print('EEGAugmentation OK')

# SpecAugment
sa = SpecAugment(p_aug=1.0)
spec = torch.randn(1, 129, 41)
out = sa(spec)
assert out.shape == spec.shape, 'shape mismatch'
assert (out == 0).any(), 'no masking applied'
print('SpecAugment OK')
"
```

**Training smoke test:**
```bash
poetry run python main.py train --test-mode \
  --channel in-ear --dataset all --model SmallerAttn \
  --condition ec --n-folds 2 \
  --augment-data 0.5 --spec-augment 0.5 \
  --checkpoint-dir experiments/test_augment
```

## Notes

- `--augment-data` and `--spec-augment` are independent and can be combined
- SpecAugment is only active during training (applied inside `FlattenedSpectrogramDataset.__getitem__`, which is called by the training DataLoader; validation uses a Subset of the same dataset but SpectrogramDataset's caching is per-file so augmentation is controlled by the flag, not by split)
  - **Important:** verify that val split does NOT get spec_augmentation applied — SpectrogramDataset should only apply it when `training=True` OR the caller must ensure val dataset has `spec_augmentation=None`. Simplest: pass `spec_augmentation=None` when constructing the val SpectrogramDataset.
- Keep backward compat: `EEGAugmentation` with old arguments (`use_ft_surrogate` etc.) should raise a clear deprecation error or just silently ignore removed kwargs via `**kwargs` — prefer explicit error.
