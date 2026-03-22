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

| Model              | Input channels                                  | Temporal aggregation         | RNN hidden / d\_model | Parameters |
|--------------------|-------------------------------------------------|------------------------------|-----------------------|------------|
| `CNN_LSTM_DepCap`  | 1                                               | LSTM                         | 100                   | 798,306    |
| `Smaller`          | 1                                               | LSTM                         | 64                    | 256,770    |
| `SmallerAttn`      | 1                                               | Self-attention (Transformer) | 128                   | 265,218    |
| `SmallerAll`       | 8 (Conv3d)                                      | LSTM                         | 64                    | 283,266    |
| `SmallerAllAttn`   | 8 (Conv3d)                                      | Self-attention (Transformer) | 128                   | 289,794    |
| `SmallerAllV2`     | 8 (per-channel shared CNN + cross-channel attn) | LSTM                         | 64                    | 345,667    |
| `SmallerAllV2Attn` | 8 (per-channel shared CNN + cross-channel attn) | Self-attention (Transformer) | 128                   | 354,115    |
| `SmallerAllV3`     | 8 (per-channel shared CNN + full concat projection) | LSTM              | 100 (default)         | 1,001,522  |

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