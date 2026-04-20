# Plan: Add LGGNet Architecture to EEG Classification Project

## Context

Implement the LGGNet architecture (TNNLS 2023, Ding et al.) from the reference code at
`/home/milan/Documents/diplomka/LGG/` as two new model variants in the project. LGGNet
uses a local-global graph convolutional approach: a multi-scale temporal CNN block extracts
per-channel features, a local graph layer aggregates channels into anatomical brain regions,
and a global GCN learns inter-region connectivity via a trainable adjacency mask. The design
is well-suited for small EEG channel sets (including 8-channel) and has architectural
analogues to depression/anxiety neuroscience (frontal asymmetry, hemispheric lateralization).

Two variants:
- **LGGNet**: full model (num_T=64) → ~1.17M params (between Deformer 1.78M and DeformerS 588K)
- **LGGNetS**: smaller model (num_T=32) → ~585K params (matches DeformerS ~588K)

Both are raw-EEG models (bypass spectrogram pipeline), following the Deformer pattern.

---

## Graph Topology for 8 Channels

Canonical order: `Fp1(0), Fp2(1), T7(2), T8(3), C3(4), C4(5), Cz(6), Oz(7)`

| Graph type | Channel reorder (indices) | idx_graph | Regions |
|------------|--------------------------|-----------|---------|
| `"frontal"` | `[0,1,2,3,4,5,6,7]` (no reorder) | `[2,2,3,1]` | Fp1+Fp2 / T7+T8 / C3+C4+Cz / Oz |
| `"hemisphere"` | `[0,2,4,1,3,5,6,7]` | `[3,3,2]` | L:Fp1+T7+C3 / R:Fp2+T8+C4 / Mid:Cz+Oz |

Default: `"hemisphere"` (matches paper default `hem`).

---

## Architecture Details (250 Hz, 2500 samples, 8 channels)

**Forward pass**: `(B, 8, 2500)` input

1. `x = x[:, channel_order, :].unsqueeze(1)` → `(B, 1, 8, 2500)`
2. Three parallel `Tception` layers (Conv2d + PowerLayer):
   - Kernels: `(1, 125)`, `(1, 62)`, `(1, 31)` at 250 Hz
   - PowerLayer: `log(avg_pool(x²))`, pool=32, stride=8
   - Outputs concatenated along time dim → `(B, num_T, 8, ~900)`
3. `BN_t(num_T)` → `OneXOneConv(num_T→num_T, 1×1)` + LeakyReLU + AvgPool(1,2) → `BN_t_`
4. Permute + reshape → `(B, 8, features)` where features = num_T × ~450
5. Local filter: `ReLU(x ⊙ weight - bias)` element-wise per channel
6. `Aggregator`: mean-pool channels by brain region → `(B, num_regions, features)`
7. `get_adj`: self-similarity + learnable mask + D^(-½)AD^(-½) normalization
8. `BN(num_regions)` → `GraphConvolution(features → out_graph)` → `BN_(num_regions)`
9. Flatten → Dropout → Linear → logits

**Param counts** (hemisphere graph, 2-class):
- LGGNet: local_filter 230K + GCN 922K + temporal/misc 18K ≈ **1.17M**
- LGGNetS: local_filter 115K + GCN 461K + temporal/misc 9K ≈ **585K**

**Hyperparameters from paper** (used as defaults):
- Adam, lr=1e-3 (already project default)
- batch_size=64 (already project default)
- dropout=0.5 (already project default)
- pool=32 (paper: 16 at 128 Hz → scaled: 32 at 250 Hz)
- pool_step_rate=0.25

Two-stage training from paper is NOT implemented (use standard cosine LR annealing like Deformer).

---

## Critical Reference Code Notes

- `BN_t` is `num_T` channels (NOT `3*num_T`): concatenation is along time dim (dim=-1), not filter dim
- `Aggregator` in reference is a plain class, not `nn.Module` — promote it to `nn.Module` in our port
- `get_adj` uses global `DEVICE` — fix to `torch.eye(n, device=x.device)` for device independence
- Reference `input_size = (1, num_chan, num_time)` tuple → refactor to separate `num_chan` and `num_time` params (matching Deformer pattern)
- Reference uses `dropout_rate` parameter → rename to `dropout` so create_model() dispatch is uniform

---

## Files Changed

### NEW: `thesis/lggnet.py`

Port from `/home/milan/Documents/diplomka/LGG/networks.py` and `layers.py`.
CBCR License 1.0 (same as `thesis/deformer.py`). See `LICENSE` for third-party entry.

Key classes: `PowerLayer`, `GraphConvolution`, `Aggregator` (promoted to `nn.Module`), `LGGNet`.

### MODIFIED: `thesis/model_factory.py`

- Added `LGGNetConfig` and `LGGNetSConfig` dataclasses
- Renamed `_DEFORMER_DEFAULT_CONFIGS` → `_RAW_EEG_DEFAULT_CONFIGS` (added LGGNet entries)
- Renamed `deformer_config` parameter → `raw_eeg_config` in `create_model()`

### MODIFIED: `thesis/model.py`

- Added `from thesis.lggnet import LGGNet`
- Extended `RAW_EEG_MODELS` frozenset with `"LGGNet"`, `"LGGNetS"`
- Added entries to `MODEL_REGISTRY`

### MODIFIED: `thesis/cli.py`

- Added `LGGNet hyperparameters` argument group with 5 flags:
  `--lggnet-num-t`, `--lggnet-out-graph`, `--lggnet-pool`, `--lggnet-pool-step-rate`, `--lggnet-graph-type`

### MODIFIED: `main.py`

- Updated import to use `_RAW_EEG_DEFAULT_CONFIGS` and `LGGNetConfig`
- Renamed `deformer_config` → `raw_eeg_config` in `train_cross_validation()`
- Added LGGNet config-building block in `train()`
- Added 5 LGGNet params to logger.info preamble and results.txt

### MODIFIED: `LICENSE`

- Added `thesis/lggnet.py` entry to THIRD-PARTY EXCEPTIONS section

---

## Verification Commands

```bash
# Smoke test
poetry run python -c "
from thesis.model_factory import create_model
import torch
m = create_model('LGGNet', (129,41), 0.5, 2, torch.device('cpu'), in_channels=8)
print(m.count_parameters())  # (1170815, 1170815)
print(m(torch.randn(4,8,2500)).shape)  # torch.Size([4, 2])
"

# CLI smoke test
poetry run python main.py train --model LGGNet \
  --dataset mdd --condition ec --channel all \
  --skip-ica --test-mode \
  --checkpoint-dir experiments/mdd_999_all_ec_lggnet_test

# Smaller variant
poetry run python main.py train --model LGGNetS \
  --dataset mdd --condition ec --channel all \
  --skip-ica --test-mode \
  --checkpoint-dir experiments/mdd_999_all_ec_lggnet_s_test
```
