# Model Card

All models are defined in `thesis/model.py`. Parameter counts are computed with the default
input shape **(129, 41)** (STFT output: 129 frequency bins × 41 time frames).

# Dataset Overview

> **CANE and IDUN are paired recordings from the same subjects** — cap EEG (CANE) and
> in-ear EEG (IDUN) were collected simultaneously in the same study.
> When `--channel in-ear` is used, CANE is automatically replaced by IDUN.

---

## Technical Specifications

| Property              | MDD                              | CANE                             | IDUN                     | SAD                                             |
|-----------------------|----------------------------------|----------------------------------|--------------------------|-------------------------------------------------|
| Target condition      | Major Depression                 | Anxiety / Dep. / Comorbid        | ← same as CANE           | Social Anxiety                                  |
| Paper                 | Mumtaz et al. 2017               | —                                | —                        | Al-Ezzi et al. 2023                             |
| Device                | Brain Master Systems             | —                                | IDUN 'Guardian' (in-ear) | egosports amplifier + ANT Neuro shielded cap    |
| Sampling rate         | 256 Hz                           | 500 Hz                           | 250 Hz                   | 2048 → 256 Hz                                   |
| Electrodes (total)    | 19-ch cap                        | 8-ch cap                         | 1 (in-ear)               | 30-ch cap                                       |
| Channels used         | 8 scalp (10–20)                  | 8 scalp (10–20)                  | 1 in-ear                 | 8 scalp (10–20)                                 |
| Channels used (names) | Fp1, Fp2, C3, Cz, C4, T7, T8, O2 | Fp1, Fp2, C3, Cz, C4, T7, T8, O2 | in-ear                   | Fp1, Fp2, C3, Cz, C4, T7, T8, O2                |
| Reference             | Linked-ear → infinity (IR)       | —                                | —                        | reference CPz, ground AFz                       |
| File format           | EDF                              | CSV                              | CSV + quality file       | EDF                                             |
| Recording paradigm    | Resting state                    | Resting state                    | Resting state            | Resting state                                   |
| Duration              | 5 min EC + 5 min EO              | 5 min EC + 5 min EO              | —                        | ~4–6 min (~120 s used)                          |
| Diagnosis tool        | DSM-IV · BDI-II · HADS           | —                                | —                        | Structured Clinical Interview for DSM-IV + SIAS |

> I have received only a subset of the full SAD dataset.
> SAD has POz which I'd rather use as substitue for Oz, but given MDD uses O2, I'm sticking with that.

---

## Subject & Chunk Counts

| Class                | MDD                | CANE                   | IDUN        | SAD     |
|----------------------|--------------------|------------------------|-------------|---------|
| Healthy / Normals    | H: 28              | Normals: 19 EC / 18 EO | Normals: 19 | HC: 21  |
| Anxiety              | —                  | AX: 18                 | AX: 20      | SAD: 21 |
| Depression           | MDD: 30 EC / 32 EO | DEP: 1                 | DEP: 1      | —       |
| Comorbid             | —                  | COM: 26 EC / 25 EO     | COM: 15     | —       |
| **Total files EC**   | **58**             | **64**                 |             | **42**  |
| **Total files EO**   | **60**             | **62**                 |             | **42**  |
| **Chunks EC (10 s)** | **1 741**          |                        |             |         |
| **Chunks EO (10 s)** | **1 794**          |                        |             |         |

TODO double check normals count for MDD

---

## Parameter Counts

| Model                  | Params   | Input channels                                      | Temporal aggregation                                   | RNN hidden / d\_model |
|------------------------|----------|-----------------------------------------------------|--------------------------------------------------------|-----------------------|
| `CNN_LSTM_DepCap`      | 798 K    | 1                                                   | LSTM                                                   | 100                   |
| `Smaller`              | 257 K    | 1                                                   | LSTM                                                   | 64                    |
| `SmallerAttn`          | 265 K    | 1                                                   | Self-attention (Transformer)                           | 128                   |
| `SmallerAll`           | 283 K    | 8 (Conv3d)                                          | LSTM                                                   | 64                    |
| `SmallerAllAttn`       | 290 K    | 8 (Conv3d)                                          | Self-attention (Transformer)                           | 128                   |
| `SmallerAllV2`         | 346 K    | 8 (per-channel shared CNN + cross-channel attn)     | LSTM                                                   | 64                    |
| `SmallerAllV2Attn`     | 354 K    | 8 (per-channel shared CNN + cross-channel attn)     | Self-attention (Transformer)                           | 128                   |
| `SmallerAllV3`         | 1.00 M   | 8 (per-channel shared CNN + full concat projection) | LSTM                                                   | 100 (default)         |
| `Deformer`             | 1.78 M   | 8 (raw EEG, 2500 samples @ 250 Hz)                  | Dense CNN-Transformer (depth=4)                        | heads=16, dim_head=16 |
| `DeformerS`            | 588 K    | 8 (raw EEG, 2500 samples @ 250 Hz)                  | Dense CNN-Transformer (depth=3)                        | heads=4,  dim_head=16 |
| `DeformerS` (5 s)      | 390 K    | 8 (raw EEG, 1250 samples @ 250 Hz)                  | Dense CNN-Transformer (depth=3)                        | heads=4,  dim_head=16 |
| `TSception`            | ~1.05 M  | 8 (raw EEG, 2500 samples @ 250 Hz)                  | Multi-scale temporal inception + asymmetric spatial CNN | —                    |
| `TSceptionS`           | ~264 K   | 8 (raw EEG, 2500 samples @ 250 Hz)                  | Multi-scale temporal inception + asymmetric spatial CNN | —                    |
| `TSceptionS` (4 s) †   | ~102 K   | 8 (raw EEG, 1000 samples @ 250 Hz)                  | Multi-scale temporal inception + asymmetric spatial CNN | —                    |
| `LGGNet`               | 1.17 M   | 8 (raw EEG, 2500 samples @ 250 Hz)                  | Multi-scale Tception → local filter → GCN (hemisphere) | out_graph=32         |
| `LGGNetS`              | 585 K    | 8 (raw EEG, 2500 samples @ 250 Hz)                  | Multi-scale Tception → local filter → GCN (hemisphere) | out_graph=32         |

† The paper (TAFFC 2022) downsampled DEAP from 512 Hz to **128 Hz**, giving 4 s × 128 Hz = **512 samples**.
MAHNOB-HCI (the paper's second dataset) was recorded at **256 Hz** and was not explicitly downsampled.
This project's data is at **250 Hz and was not resampled**, so 4 s chunks contain **1000 samples** —
2× the DEAP input length but comparable to MAHNOB-HCI conditions.

## Architecture Summary

### `CNN_LSTM_DepCap`
Baseline architecture inspired by the reference paper.
- Conv2d(1→64, 10×10, stride=2) + MaxPool + Dropout2d
- Conv2d(64→32, 5×5) + MaxPool + Dropout2d
- LSTM (hidden=100)
- Dense 64 → 32 → *num\_classes*

### `Smaller`
Reduced version of `CNN_LSTM_DepCap` (~68% fewer parameters).
- Conv2d(1→32, 10×10, stride=2) + MaxPool + Dropout2d
- Conv2d(32→16, 5×5) + MaxPool + Dropout2d
- LSTM / GRU / Transformer (configurable)
- Dense 32 → 16 → *num\_classes*

### `SmallerAttn`
`Smaller` with temporal self-attention (TransformerEncoder) instead of LSTM.

### `SmallerAll`
Multi-channel (8-channel) variant of `Smaller` using a 3D convolution to process all EEG
channels simultaneously. Conv3d kernel spans the full channel depth before collapsing to 2D.

### `SmallerAllAttn`
`SmallerAll` with temporal self-attention instead of LSTM.

### `SmallerAllV2`
Per-channel weight-shared CNN + cross-channel self-attention + LSTM.
Each EEG channel is processed independently through the `Smaller` CNN (shared weights).
A TransformerEncoder then computes soft channel-importance weights before merging into the
LSTM path. `chan_d_model=64` by default.

### `SmallerAllV2Attn`
`SmallerAllV2` with temporal self-attention replacing LSTM.

### `SmallerAllV3`
Per-channel weight-shared CNN + **full channel concatenation** + LSTM.

Key difference from all previous multi-channel models: all 8 channels' CNN features are concatenated
at each time step and projected to `chan_d_model`, so the LSTM always sees every channel simultaneously.
No channel is discarded or soft-selected before the recurrent layer.

Architecture stages:
1. **Per-channel CNN** (shared weights): each of 8 channels processed by the same Conv2d stack → `(B*8, 16, H', W')`
2. **Concat + project**: reshape to `(B, W', 8×16×H')` → `Linear(6912→128)` + ReLU → `(B, W', 128)`
3. **LSTM(128→100)**: temporal sequence modelling
4. **Classifier** (→64→32→num_classes)

`chan_d_model=128`, `rnn_hidden=100` by default. Note: the 3a design (per-frame attention, rnn_hidden=330,
~1.53M params) was used in experiment 021; the current 4b concat design was introduced for experiment 022.

### `Deformer`
EEG-Deformer (Ding et al., J-BHI 2024). Dense convolutional transformer operating directly on raw EEG
(no STFT). Takes `(batch, 8, 2500)` as input.

Architecture stages:
1. **Shallow CNN encoder**: Conv2d(1→64, 1×25) + depthwise Conv2d(64→64, 8×1) + BN + ELU + MaxPool → `(B, 64, 1, 1250)`
2. **Patch embedding**: rearrange to `(B, 64, 1250)` + learned positional embedding
3. **Dense Transformer** (depth=4): each layer produces a coarse-grained path (multi-head attention, 16 heads × 16 dim) and a fine-grained path (Conv1d + BN + ELU + MaxPool); both are fused and the fine-grained log-power summary is concatenated into a dense feature vector
4. **MLP head**: Linear(out_size → num_classes)

Sub-module parameter breakdown: CNN encoder 34,624 · Transformer 1,651,692 · MLP head 10,498.

Defined in `thesis/deformer.py` (verbatim from the original repo, CBCR License 1.0).
Configured via `DeformerConfig` in `thesis/model_factory.py`.

### `DeformerS`
Same architecture as `Deformer`, ~1/3 the size (587,526 params). Uses `DeformerSConfig` by default:
`num_kernel=48, depth=3, heads=4, dim_head=16, mlp_dim=16, temporal_kernel=25`.

Key differences from `Deformer`:
- **num_kernel 64→48**: narrower CNN encoder and fewer transformer tokens
- **depth 4→3**: one fewer hierarchical level; final sequence is 156 samples (~0.6 s) instead of 78 (~0.3 s)
- **heads 16→4**: inner attention dim 256→64, the largest single saving (~67% reduction in attention params)

Parameter count scales with `num_time` because: (a) the learned positional embedding is
`(1, num_kernel, num_time // 2)` and (b) the Dense Transformer accumulates a coarser-grained
log-power vector at each depth level whose size depends on the input sequence length; both feed
the final MLP head. At `num_time=1250` (5 s) → **390,314 params**; at `num_time=2500` (10 s) → **587,526 params**.

### `LGGNet`
Local-Global Graph Network for raw EEG classification (Ding et al., TNNLS 2023).
Implemented in `thesis/lggnet.py` (CBCR License 1.0). Configured via `LGGNetConfig`.

Architecture stages:
1. **Channel reorder**: channels permuted to graph-topology order, unsqueezed → `(B, 1, 8, 2500)`
2. **Multi-scale Tception** (3 parallel `Conv2d + PowerLayer`): kernels `(1, 125)`, `(1, 62)`, `(1, 31)`;
   PowerLayer: `log(avg_pool(x²).clamp(min=1e-6))`, pool=32, stride=8; outputs concatenated along time dim
   → `(B, num_T, 8, ~899)`
3. **BN + 1×1 Conv + AvgPool(1,2) + BN**: → `(B, num_T, 8, ~449)`
4. **Permute + reshape**: → `(B, 8, num_T × ~449)` = `(B, 8, features)`
5. **Local filter**: `ReLU(x ⊙ weight − bias)`, per-channel element-wise, weight `(8, features)` learnable
6. **Aggregator**: mean-pools channels into brain-region nodes → `(B, num_regions, features)`
7. **Adjacency**: `get_adj` computes self-similarity + learnable mask `global_adj` + D^(−½)AD^(−½) norm
8. **GCN** (`GraphConvolution`): `ReLU(A · (X·W − b))`, `features → out_graph=32` → `(B, num_regions, 32)`
9. **BN + Flatten + Dropout + Linear** → logits

Graph topologies (8-channel, canonical order Fp1,Fp2,T7,T8,C3,C4,Cz,Oz):
- `hemisphere` (default): Left {Fp1,T7,C3} / Right {Fp2,T8,C4} / Midline {Cz,Oz} — 3 regions
- `frontal`: {Fp1,Fp2} / {T7,T8} / {C3,C4,Cz} / {Oz} — 4 regions

Parameter breakdown (hemisphere, 2-class, num_T=64): temporal 18K · local_filter 230K · GCN 922K · total **1,170,815**.

### `LGGNetS`
Same architecture as `LGGNet` with **num_T 64→32** as the only change.

Parameter breakdown (hemisphere, 2-class, num_T=32): temporal 9K · local_filter 115K · GCN 461K · total **584,511**.
Matches DeformerS (~588K) in parameter count; much simpler architecture (no attention).

Configured via `LGGNetSConfig` in `thesis/model_factory.py`.

### `TSception`
Multi-scale temporal inception + asymmetric spatial CNN for raw EEG (Ding et al., TAFFC 2022).
Takes `(batch, 8, 2500)` as input. Implemented in `thesis/tsception.py` (CBCR License 1.0).

Architecture stages:
1. **Temporal inception** (3 parallel branches): `Conv2d(1→num_T, (1, k_i))` + LeakyReLU + AvgPool(1,8),
   with kernel fractions [0.5, 0.25, 0.125]×fs → k = [125, 62, 31] at 250 Hz; concatenated along time axis
2. **BatchNorm2d(num_T)**
3. **Asymmetric spatial** (2 branches): global branch `Conv2d(num_T→num_S, (C,1))` + hemisphere branch
   `Conv2d(num_T→num_S, (C//2,1), stride=(C//2,1))` + AvgPool(1,2); concatenated along channel axis
4. **Flatten → FC**: Linear(flat→hidden) + ReLU + Dropout + Linear(hidden→num_classes)

With 8 channels and 2500 samples: flat feature size = num_S×3×454 = **8172**; FC1 alone (8172×128)
accounts for 99.7% of TSception's 1.05 M parameters. This FC dominance is an expected consequence of
the paper's architecture scaling to longer inputs.

Default config: `num_T=9, num_S=6, hidden=128, sampling_rate=250, num_time=2500`.
Configured via `TSceptionConfig` in `thesis/model_factory.py`.

### `TSceptionS`
Reduced-parameter variant of `TSception` (~264 K), matching `SmallerAll` in scale.

Same architecture as `TSception` with a single change: **hidden 128→32** (FC1: 8172×32 instead of 8172×128),
following the paper's own cross-dataset recommendation:
*"we also suggest T=S=15 and hidden_node=32 when applying TSception to other datasets."*

Default config: `num_T=9, num_S=6, hidden=32, sampling_rate=250, num_time=2500`.
Configured via `TSceptionSConfig` in `thesis/model_factory.py`.

---

## References

**MDD** — Mumtaz W. et al. "Electroencephalogram (EEG)-based computer-aided technique to diagnose
major depressive disorder (MDD)." *Biomedical Signal Processing and Control* 31 (2017) 108–115.
DOI: 10.1016/j.bspc.2016.07.006

**SAD** — Al-Ezzi A. et al. "Machine Learning for the Detection of Social Anxiety Disorder Using
Effective Connectivity and Graph Theory Measures." *IEEE Trans. Neural and Rehab. Eng.* (2023).

**Surrogate in-ear channel creation** — Tremmel C. et al. "Estimating cognitive workload using a commercial in-ear EEG
headset." *Journal of Neural Engineering* 21 (2024) 066022. DOI: 10.1088/1741-2552/ad8ef8
*(cited as methodological basis for in-ear EEG; the clinical CANE/IDUN data is from a separate study)*

**TSception / TSceptionS** — Ding Y. et al. "TSception: Capturing Temporal Dynamics and Spatial Asymmetry
from EEG for Emotion Recognition." *IEEE Trans. Affective Computing* (2022). DOI: 10.1109/TAFFC.2022.3169001
Source: https://github.com/yi-ding-cs/TSception (CBCR License 1.0)

**LGGNet / LGGNetS** — Ding Y. et al. "LGGNet: Learning From Local-Global-Graph Representations for
Brain-Computer Interface." *IEEE Trans. Neural Networks and Learning Systems* (2023).
DOI: 10.1109/TNNLS.2023.3236635
Source: https://github.com/yi-ding-cs/LGG (CBCR License 1.0)