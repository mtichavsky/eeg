# Plan: Structured JSON Logging for Training Metrics

## Overview

Convert training metrics logging from text format to JSON while keeping status messages as plain text. This provides machine-parseable metrics without sacrificing console readability.

## Current State

**Text format (hard to parse):**
```
[2026-02-01 13:03:39 INFO main.train_one_fold] TRAIN CHUNK | Fold 5 | Epoch 018/100 | Loss: 0.1871 | Acc: 0.9303 | Prec: 0.9404 | Recall: 0.9478 | Spec: 0.9017
[2026-02-01 13:03:42 INFO main.train_one_fold] EVAL CHUNK | Fold 5 | Epoch 018/100 | Loss: 0.9405 | Acc: 0.6836 | Prec: 0.7273 | Recall: 0.7470 | Spec: 0.5912
[2026-02-01 13:03:42 INFO main.train_one_fold]   Confusion: TP=496, TN=269, FP=186, FN=168
[2026-02-01 13:03:42 INFO main.train_one_fold] EVAL SUBJECT | Fold 5 | Epoch 018/100 | Loss: 0.9405 | Acc: 0.7027 | ...
[2026-02-01 13:03:42 INFO main.train_one_fold]   Confusion: TP=17, TN=9, FP=6, FN=5
[2026-02-01 13:03:42 INFO main.train_one_fold] EVAL EC vs EO | Fold 5 | Epoch 018/100 | EO chunk acc: 0.7000 | EC chunk acc: 0.7059
```

**Parsed via regex in** `plot_training_curves.py:38-130`

## Target State

**Console (pretty JSON for metrics, text for status):**
```
[2026-02-01 13:03:39 INFO] Starting 10-fold CV training
{
  "t": "2026-02-01T13:03:39.828",
  "phase": "train",
  "fold": 5,
  "epoch": 18,
  "max_epoch": 100,
  "loss": 0.1871,
  "chunk": {"acc": 0.9303, "prec": 0.9404, "rec": 0.9478, "spec": 0.9017}
}
{
  "t": "2026-02-01T13:03:42.778",
  "phase": "eval",
  "fold": 5,
  "epoch": 18,
  "max_epoch": 100,
  "loss": 0.9405,
  "chunk": {"acc": 0.6836, "prec": 0.7273, "rec": 0.7470, "spec": 0.5912, "tp": 496, "tn": 269, "fp": 186, "fn": 168},
  "subject": {"acc": 0.7027, "prec": 0.7391, "rec": 0.7727, "spec": 0.6000, "tp": 17, "tn": 9, "fp": 6, "fn": 5},
  "condition": {"eo_acc": 0.7000, "ec_acc": 0.7059}
}
```

**File (compact single-line JSON):**
```
{"t":"2026-02-01T13:03:39.828","phase":"train","fold":5,"epoch":18,"max_epoch":100,"loss":0.1871,"chunk":{"acc":0.9303,"prec":0.9404,"rec":0.9478,"spec":0.9017}}
{"t":"2026-02-01T13:03:42.778","phase":"eval","fold":5,"epoch":18,"max_epoch":100,"loss":0.9405,"chunk":{"acc":0.6836,...},"subject":{...},"condition":{...}}
```

## JSON Schema

```python
# Training metrics (phase="train")
{
    "t": str,           # ISO timestamp
    "phase": "train",
    "fold": int,
    "epoch": int,
    "max_epoch": int,
    "loss": float,
    "chunk": {
        "acc": float,
        "prec": float,
        "rec": float,
        "spec": float      # binary only
        # OR for 4-class:
        "prec_normal": float, "rec_normal": float, ...
    }
}

# Evaluation metrics (phase="eval")
{
    "t": str,
    "phase": "eval",
    "fold": int,
    "epoch": int,
    "max_epoch": int,
    "loss": float,
    "chunk": {
        "acc": float, "prec": float, "rec": float, "spec": float,
        "tp": int, "tn": int, "fp": int, "fn": int  # confusion matrix
    },
    "subject": {
        "acc": float, "prec": float, "rec": float, "spec": float,
        "tp": int, "tn": int, "fp": int, "fn": int
    },
    "condition": {       # optional, only when EC+EO combined
        "ec_acc": float,
        "eo_acc": float
    }
}
```

## Implementation

### 1. Create `thesis/json_logging.py` (new file)

```python
"""Structured JSON logging for training metrics."""
import json
import logging
from datetime import datetime
from typing import Any

def log_metrics_json(
    logger: logging.Logger,
    phase: str,  # "train" or "eval"
    fold: int,
    epoch: int,
    max_epoch: int,
    loss: float,
    chunk_metrics: dict,
    subject_metrics: dict | None = None,
    condition_metrics: dict | None = None,
    num_classes: int = 2,
    pretty: bool = False,
) -> None:
    """Log training/eval metrics as JSON."""
    ...

def metrics_to_json_dict(
    metrics: dict,
    num_classes: int,
    include_confusion: bool = False,
) -> dict:
    """Convert metrics dict to compact JSON-friendly dict."""
    ...
```

### 2. Modify `main.py:setup_logging()` (lines 85-124)

- Add parameter `json_metrics: bool = True`
- Store flag globally or return config object
- Console handler: keep text formatter (JSON printed directly via print())
- File handler: keep text formatter (JSON appended directly)

### 3. Modify `main.py:train_one_fold()` (lines 300-360)

Replace current logging:
```python
# BEFORE (lines 307-310)
metrics_str = format_metrics_for_logging(train_metrics, num_classes)
logger.info(f"TRAIN CHUNK | Fold {fold + 1} | Epoch {epoch:03d}/{num_epochs} | ...")

# AFTER
log_metrics_json(
    logger, "train", fold + 1, epoch, num_epochs,
    train_metrics["loss"], train_metrics, num_classes=num_classes
)
```

Consolidate eval logging (currently 5 separate logger.info calls) into single call:
```python
# BEFORE (lines 328-357): 5 separate log calls
# AFTER: 1 call with all data
log_metrics_json(
    logger, "eval", fold + 1, epoch, num_epochs,
    eval_metrics["loss"], chunk_metrics,
    subject_metrics=subject_metrics,
    condition_metrics=condition_metrics,
    num_classes=num_classes
)
```

### 4. Modify `plot_training_curves.py:parse_log_file()` (lines 38-130)

Add JSON parsing alongside existing regex:

```python
def parse_log_file(log_path: Path) -> Dict[int, Dict[str, Any]]:
    """Parse training log file (supports both JSON and legacy text format)."""
    data = {}

    with open(log_path) as f:
        for line in f:
            line = line.strip()

            # Try JSON first
            if line.startswith("{"):
                try:
                    record = json.loads(line)
                    _process_json_record(record, data)
                    continue
                except json.JSONDecodeError:
                    pass

            # Fall back to regex for legacy logs
            # ... existing regex code ...

    return data

def _process_json_record(record: dict, data: dict) -> None:
    """Process a JSON metrics record."""
    fold = record["fold"]
    epoch = record["epoch"]
    loss = record["loss"]

    if fold not in data:
        data[fold] = {"train": [], "eval": [], "best_epoch": None, "best_loss": None}

    if record["phase"] == "train":
        data[fold]["train"].append((epoch, loss))
    elif record["phase"] == "eval":
        acc = record["chunk"]["acc"]
        data[fold]["eval"].append((epoch, loss, acc))
```

### 5. Update `thesis/metrics.py:format_metrics_for_logging()` (lines 138-163)

Keep for backward compatibility but add deprecation note. The function is still useful for non-JSON contexts.

## Files to Modify

| File | Changes |
|------|---------|
| `thesis/json_logging.py` | **NEW** - JSON logging utilities |
| `main.py` | Modify `train_one_fold()` to use JSON logging |
| `plot_training_curves.py` | Add JSON parsing support |
| `thesis/metrics.py` | Minor: add helper for metrics dict extraction |

## Verification

1. **Run training with new logging:**
   ```bash
   poetry run python main.py train --test-mode --checkpoint-dir=experiments/test_json_logging --dataset mdd --condition ec
   ```

2. **Verify console output** is readable (pretty JSON for metrics, text for status)

3. **Verify log file** has compact single-line JSON for metrics

4. **Test plot_training_curves.py** parses new format:
   ```bash
   poetry run python plot_training_curves.py experiments/test_json_logging/training_*.log
   ```

5. **Test backward compatibility** - run on old log file to ensure regex still works

## Notes

- No new dependencies required (uses stdlib `json`)
- Backward compatible - old logs still parseable via regex fallback
- Multi-class (4-class) mode supported with per-class metrics
- Confusion matrix included in eval JSON (no separate line needed)
