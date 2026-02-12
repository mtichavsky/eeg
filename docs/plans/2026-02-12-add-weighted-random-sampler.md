# Plan: Add WeightedRandomSampler to Training Pipeline

## Context
The model shows 96% sensitivity but only 52% specificity — it's biased toward predicting "pathological." Class weights in CrossEntropyLoss adjust gradient magnitude but don't fix batch composition. WeightedRandomSampler forces each batch to see ~50/50 class balance during training, which is the standard fix for this pattern.

## Changes

### File: `main.py`

**1. Add import** (line 13)

Add `WeightedRandomSampler` to the existing `torch.utils.data` import:
```python
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
```

**2. Create sampler after class weights computation** (after line 764)

Build per-sample weights from the already-computed `class_weights` tensor, then create the sampler:

```python
# Build per-sample weights for WeightedRandomSampler
sample_weights = []
for idx in range(len(train_dataset)):
    _, label, _ = train_dataset[idx]
    sample_weights.append(class_weights[label].item())
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))
```

Note: `class_weights` is already computed on the line above. We reuse it — `class_weights[label]` gives the weight for that class. `.item()` converts from tensor to float since the sampler expects plain floats.

**3. Replace `shuffle=True` with `sampler=sampler`** (lines 770-777)

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=sampler,          # was: shuffle=True
    num_workers=0,
    pin_memory=pin_memory,
    collate_fn=collate_spectrograms,
)
```

`sampler` and `shuffle=True` are mutually exclusive in PyTorch — the sampler handles randomization.

No changes to val_loader, loss function, or anything else.

## Verification
```bash
poetry run python main.py train --test-mode --dataset mdd --condition ec --checkpoint-dir experiments/test_sampler --channel all
```
Check that training starts without errors and logs show class weights as before.
