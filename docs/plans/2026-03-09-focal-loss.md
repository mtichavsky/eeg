# Plan: Add Focal Loss as Optional Training Objective

## Context

Current training uses `nn.CrossEntropyLoss()` (unweighted) with `WeightedRandomSampler` handling class imbalance at the batch-sampling level. Despite this, specificity remains ~60% — the model is biased toward predicting pathological. Focal loss addresses the complementary problem: it down-weights easy/confidently-correct pathological predictions and forces the model to focus on hard examples (healthy subjects being misclassified). Adding it as an opt-in flag (`--focal-loss`) with configurable gamma lets experiments compare directly.

## Files to Modify

1. **`thesis/loss.py`** — new file, `FocalLoss` class
2. **`thesis/cli.py`** — add `--focal-loss` and `--focal-gamma` flags
3. **`main.py`** — import, extract args, replace criterion instantiation at line 795

## Implementation Steps

### Step 1: Create `thesis/loss.py`

```python
"""Custom loss functions for EEG classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification with optional class-weight alpha.

    Re-weights cross-entropy by (1 - p_t)^gamma so easy examples contribute
    less to the total loss. When alpha is provided it acts as a class-frequency
    prior (same as the weight argument of CrossEntropyLoss).

    :param torch.Tensor | None alpha: Per-class weight tensor of shape (num_classes,).
    :param float gamma: Focusing exponent. 0.0 reduces to weighted cross-entropy.
    :param str reduction: 'mean' (default) or 'sum'.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha: torch.Tensor | None = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        :param torch.Tensor logits: Raw model outputs of shape (N, C).
        :param torch.Tensor targets: Class indices of shape (N,).
        :return: Scalar loss value.
        :rtype: torch.Tensor
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        p_t = F.softmax(logits, dim=1).gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        loss = ((1.0 - p_t) ** self.gamma) * ce_loss
        return loss.mean() if self.reduction == "mean" else loss.sum()
```

`register_buffer("alpha", ...)` means the weight tensor automatically moves with `.to(device)` — no manual device management at the call site.

### Step 2: Add CLI flags in `thesis/cli.py`

After the `--weight-decay` argument block, add:

```python
train_parser.add_argument(
    "--focal-loss",
    action="store_true",
    help="Use Focal Loss instead of cross-entropy. Class weights computed per fold "
    "are passed as alpha. Compatible with WeightedRandomSampler.",
)
train_parser.add_argument(
    "--focal-gamma",
    type=float,
    default=2.0,
    help="Focusing exponent for Focal Loss (only used with --focal-loss). "
    "gamma=0 reduces to weighted cross-entropy. Typical range: 1.0-5.0.",
)
```

### Step 3: Update `main.py`

**Import** (alongside existing `thesis.*` imports):
```python
from thesis.loss import FocalLoss
```

**Extract args** (alongside dropout, weight_decay, etc.):
```python
focal_loss: bool = args.focal_loss
focal_gamma: float = args.focal_gamma
```

**Replace criterion instantiation** at line 795 (inside fold loop, after `class_weights` is already computed at line 747):
```python
# Before:
criterion = nn.CrossEntropyLoss()

# After:
if focal_loss:
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=focal_gamma)
else:
    criterion = nn.CrossEntropyLoss()
```

`class_weights` is available from line 747 (`device="cpu"`), so `.to(device)` moves it to GPU when needed. No changes needed to `train_one_fold`, `train_epoch`, or `eval_epoch` — they accept `criterion: nn.Module` and call `criterion(logits, yb)`.

## Verification

1. Smoke test — runs without error, focal loss logged:
   ```bash
   python main.py train --focal-loss --focal-gamma 2.0 --test-mode --dataset all --n-folds 2
   ```

2. Sanity check — `--focal-gamma 0` should produce nearly identical results to baseline (weighted CE behaviour):
   ```bash
   python main.py train --focal-loss --focal-gamma 0 --test-mode --n-folds 2
   ```

3. Real run to compare specificity:
   ```bash
   python main.py train --focal-loss --focal-gamma 2.0 --channel in-ear --dataset all \
     --model Smaller --condition ec+eo --n-folds 6 --dropout 0.1 --weight-decay 1e-4 \
     --checkpoint-dir=experiments/all3-019-inear-binary-smaller-focal
   ```

4. `make format` passes cleanly.