# Plan: Add TSception to the EEG Classification Project

## Context

Add the TSception architecture (IJCNN 2020) as two new model options. TSception uses multi-scale temporal inception convolutions + asymmetric hemisphere spatial convolutions — a complementary approach to the project's existing CNN-LSTM and Transformer families. The user wants:
- `TSception`: faithful adaptation of the original paper (paper's training params as defaults)
- `TSceptionS`: reduced-size variant matching the project's smaller models in param count

Both operate on **raw EEG** (bypass the STFT spectrogram pipeline), fitting naturally into `RAW_EEG_MODELS`.

---

## Architecture Overview

### Forward pass (original TSception)

```
Input: (batch, channels, num_time)        ← project's standard raw-EEG shape
  └─ reshape → (batch, 1, channels, time)  ← TSception's native 4D input

Temporal inception (3 parallel branches, concat along dim=-1 / time axis):
  Tception1: Conv2d(1, num_T, (1, k1)) → LeakyReLU → AvgPool(1, 8)
  Tception2: Conv2d(1, num_T, (1, k2)) → LeakyReLU → AvgPool(1, 8)   k_i = int(fraction_i × fs)
  Tception3: Conv2d(1, num_T, (1, k3)) → LeakyReLU → AvgPool(1, 8)   fractions = [0.5, 0.25, 0.125]
  → BatchNorm2d(num_T)

Asymmetric spatial (2 branches, concat along dim=2 / channel axis):
  Sception1: Conv2d(num_T, num_S, (C, 1))          → LeakyReLU → AvgPool(1, 2)   C = num_channels (global)
  Sception2: Conv2d(num_T, num_S, (C//2, 1), stride=(C//2, 1)) → LeakyReLU → AvgPool(1, 2)  (hemisphere)
  → BatchNorm2d(num_S)

Flatten → FC: Linear(flat, hidden) → ReLU → Dropout → Linear(hidden, num_classes)
```

Kernel sizes for 250 Hz, 2500-sample chunks (10 s):
- k1=125, k2=62, k3=31

Time widths after temporal pool (per branch): 297 + 304 + 308 = **909** → after spatial pool: **454**
Flat feature size: num_S × 3 × 454 = 6 × 3 × 454 = **8172** (with 8 channels)

### Two variants

| Model | num_T | num_S | hidden | ~Params | Comparable to |
|-------|-------|-------|--------|---------|---------------|
| **TSception** | 9 | 6 | 128 | ~1.05 M | CNNCatLSTM (~1.0 M) |
| **TSceptionS** | 9 | 6 | 32 | ~264 k | SmallerAll (~283 k) |

TSception hidden=128 matches the paper's `Train.py` defaults.  
TSceptionS hidden=32 matches the paper's own comment: *"we also suggest T=S=15 and hidden_node=32 when applying TSception to other datasets."*

---

## Paper Training Hyperparameters (defaults for TSception)

From `TSception/Train.py`:

| Param | Value | Notes |
|-------|-------|-------|
| Optimizer | Adam | no weight-decay in paper |
| Learning rate | **1e-3** | (project default is lower — use `--lr 0.001`) |
| Batch size | 128 | |
| Epochs | 200 | |
| Dropout | 0.3 | FC layers only |
| Early stopping | patience=4 | (project uses different early-stop, this is just FYI) |
| L1 lambda | **1e-6** | new `--l1-lambda` CLI arg added to training loop |

---

## Files Changed

| File | Action | Key change |
|------|--------|------------|
| `thesis/tsception.py` | **CREATED** | TSception class (CBCR License) + TSceptionWrapper (MIT) |
| `thesis/model.py` | modified | Import wrapper; add to RAW_EEG_MODELS + MODEL_REGISTRY |
| `thesis/model_factory.py` | modified | TSceptionConfig/TSceptionSConfig dataclasses; add to _RAW_EEG_DEFAULT_CONFIGS |
| `thesis/cli.py` | modified | TSception arg group + --l1-lambda |
| `main.py` | modified | Config building, l1_lambda threading into training loop |
| `LICENSE` | modified | Added tsception.py to THIRD-PARTY EXCEPTIONS |

---

## Verification

```bash
# Quick smoke test — single fold, 1 file per class
poetry run python main.py train --test-mode \
  --model TSception \
  --channel all \
  --dataset mdd \
  --condition ec \
  --lr 0.001 \
  --l1-lambda 1e-6 \
  --checkpoint-dir experiments/mdd_test_tsception_all_ec

poetry run python main.py train --test-mode \
  --model TSceptionS \
  --channel all \
  --dataset mdd \
  --condition ec \
  --checkpoint-dir experiments/mdd_test_tsceptionS_all_ec

# Verify param counts in results.txt: TSception ~1.05M, TSceptionS ~264k
```
