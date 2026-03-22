# Model Architecture & Training Evolution

This document traces the full design progression: model architectures from the Sharma et al. baseline through SmallerAllV3, and training techniques from plain cross-entropy to focal loss. It is the source of truth for Chapter 4.3 of the thesis.

**Last updated:** 2026-03-25
**Current best binary model:** SmallerAllV3 (3b) + focal loss (021) — 78.84% chunk, 80.22% subject
**Current best balanced model:** SmallerAllV2Attn + focal loss d=0.2 (020) — 76.99% chunk, spec 70.80%

---

## Part 1 — Model Architectures

### Generation 0 — `CNN_LSTM_DepCap` (Sharma et al. port)

**Core idea:** Direct port of the architecture from Sharma et al. (2021), adapted for our
dataset dimensions. The paper used 254×342 spectrograms; we used 129×41 (limited by
the 250 Hz / 10-second window).

```
(B, 1, H, W)
→ Conv2d(1→64, 10×10) + MaxPool + Dropout2d
→ Conv2d(64→32, 5×5) + MaxPool + Dropout2d
→ permute → (B, W', H'×32)
→ LSTM(hidden=100)
→ Dense(64→32→num_classes)
```

**Parameters:** ~800K
**Results:** Not formally benchmarked in cross-validation (paper claims >99% — data leakage,
no subject-level splits).
**Why superseded:** Massively overparameterised for ~100 subjects / 8000 chunks. Heavily
overfits. Paper's CV scheme leaked subject data across folds.

---

### Generation 1 — `Smaller`

**Core idea:** Halve filter counts and LSTM hidden size to reduce overfitting on a small dataset.

```
(B, 1, H, W)
→ Conv2d(1→32, 10×10) + MaxPool + Dropout2d    ← 64 filters → 32
→ Conv2d(32→16, 5×5) + MaxPool + Dropout2d     ← 32 filters → 16
→ permute → (B, W', H'×16)
→ LSTM(hidden=64)                              ← 100 → 64
→ Dense(32→16→num_classes)
```

**Parameters:** 256,770
**Key insight:** ~3× fewer parameters with nearly identical accuracy. The original model was
wildly overparameterised for this dataset size. This reduction is the single largest gain
in the whole project.

**Used in experiments:** 014b (in-ear), 018 (in-ear, weighted sampler)

---

### Generation 1a — `SmallerAttn`

**Core idea:** Replace LSTM with a TransformerEncoder for temporal modelling. The
spectrogram time axis (W'=10 frames) is treated as a sequence of tokens.

```
(B, 1, H, W)
→ Same Conv2d backbone as Smaller
→ permute → (B, W', H'×16)
→ Linear(H'×16 → d_model=128) + positional embeddings
→ TransformerEncoder(d_model=128, nhead=8, 1 layer)
→ mean pool over W' → (B, 128)
→ Dense(64→32→num_classes)
```

**Parameters:** 265,218
**What improves:** +0.82pp chunk, +1.52pp subject vs Smaller on in-ear binary — marginal
but consistent. Attention allows the model to weight which time frames are most informative,
rather than trusting the LSTM's last hidden state.

**Used in experiments:** 018 (in-ear), 019 (in-ear + focal), 020 (in-ear + focal, dropout grid)

---

### Generation 2 — `SmallerAll` / `SmallerAllAttn`

**Core idea:** Extend to 8 EEG channels by treating channels as the *depth* dimension of
a 3D volume, collapsed by a single Conv3d kernel.

```
(B, 8, H, W)
→ unsqueeze → (B, 1, 8, H, W)
→ Conv3d(1→32, kernel=(8,10,10))     ← depth=8 collapses all channels
→ AdaptiveMaxPool3d((1,None,None))   ← depth → 1
→ squeeze → (B, 32, H', W')
→ Conv2d(32→16) → MaxPool → Dropout2d
→ (B, W', H'×16)
→ LSTM(hidden=64) / TransformerEncoder(d_model=128)   ← Attn variant
→ Dense(32→16→num_classes)
```

**Parameters:** 283,266 (LSTM) / 289,794 (Attn)

**What it gets right:** Simple. One 3D kernel can learn a linear combination of all 8 channel
spectra.

**The problem — rigid all-or-nothing spatial filter:**
1. One fixed set of channel weights, shared across all frequencies and time frames.
2. After the first layer, the 8-channel structure is destroyed — all subsequent computation
   is on a single merged map.
3. No weight sharing across channels: the frontal theta detector learned for Fp1 cannot be
   reused for Fp2.

**Used in experiments:** 014b (8ch), 018 (8ch 4-class, weighted sampler)

---

### Generation 3 — `SmallerAllV2` / `SmallerAllV2Attn`

**Core idea:** Decouple spectral feature extraction from spatial channel mixing using a
weight-shared per-channel CNN followed by learned cross-channel attention.

```
(B, 8, H, W)
→ reshape → (B×8, 1, H, W)          ← process each channel independently
→ Shared CNN (same Conv2d weights for all 8 channels)
→ (B×8, 16, H', W')
→ reshape → (B, 8, 16, H', W')

Cross-channel attention (static — one set of weights per clip):
  mean over W' → (B, 8, 16, H')     ← TIME AXIS COLLAPSED HERE
  flatten → (B, 8, 16×H')
  Linear → (B, 8, chan_d_model=64)
  TransformerEncoder (d_model=64, nhead=4)
  Gate + softmax → weights (B, 8, 1)
  Weighted sum → (B, 16, H', W')    ← channels merged

→ permute → (B, W', H'×16)
→ LSTM(64) / TransformerEncoder(128)   ← V2 vs V2Attn
→ Dense(32→16→num_classes)
```

**Parameters:** 345,667 (LSTM) / 354,115 (Attn)

**What improves:**
- **Weight sharing:** one universal spectral detector for all electrodes. Depression biomarkers
  are spectral phenomena; the detector should be electrode-agnostic.
- **Learnable channel importance:** the TransformerEncoder computes data-driven channel
  weights instead of the fixed linear combination of Conv3d.

**The remaining problem — static channel weights:**
`mean(dim=4)` collapses W' *before* attention → one set of channel weights for the entire
10-second clip. An artifact on Fp1/Fp2 for 0.5 s cannot be selectively down-weighted; the
model has no per-frame awareness.

**Used in experiments:** 020 (8ch binary + 4-class, weighted sampler; 8ch binary + focal d=0.1/0.2/0.3)

---

### Generation 4a — `SmallerAllV3` (initial design, superseded)

> **Status: superseded.** Used in experiments `all3-021-all-binary-smallerAllV3-focal`
> and `all3-021-all-binary-smallerAllV3-focal_hsf` (both rnn_hidden=330, confirmed via
> `nhead=8` in training logs). Achieved 78.84% chunk — statistically identical to V2Attn
> but with higher fold variance and epoch-1 best in several folds. Abandoned; see 4b.

**Core idea:** Preserve the time axis *through* cross-channel attention so the model
computes per-time-frame channel importance.

```
(B, 8, H, W)
→ reshape → (B×8, 1, H, W)
→ Shared CNN → (B×8, 16, H', W')
→ reshape → (B, 8, 16, H', W')

Time-aware cross-channel attention:
  permute → (B, W', 8, 16, H')
  reshape → (B×W', 8, 16×H')        ← FOLD W' INTO BATCH, NOT MEAN-POOL
  Linear → (B×W', 8, chan_d_model=128)
  TransformerEncoder (d_model=128, nhead=8)
  Gate + softmax → (B×W', 8, 1)     ← different weights per time frame
  Weighted sum → (B×W', 128)        ← still collapses 8 channels to 1
  reshape → (B, W', 128)

→ LSTM(input=128, hidden=330)       ← oversized hidden
→ Dense(64→32→num_classes)
```

**Parameters:** 1,531,442 (rnn_hidden=330)

**Why abandoned — weighted-sum still collapses channels:**
V3a fixed V2's static-weight problem but kept the weighted *sum* to merge channels.
Softmax over 8 channels with a single gate means the model *selects* the "best" channel
at each time step and discards the rest. If the gate weights one channel at 0.7 and each
other at 0.04, the LSTM is back to seeing one channel. This explains why every
attention-based model in this line matched single-channel in-ear performance (~78–79%):
they all converged to channel selection, not channel combination. The oversized LSTM
(hidden=330) also contributed to the epoch-1-best overfitting pattern seen in loss curves.

---

### Generation 4b — `SmallerAllV3` (current design)

> **Used in experiment `all3-022-all-binary-smallerAllV3-focal`** (rnn_hidden=100, confirmed
> via `concat_dim=6912` in training log — the distinguishing marker from the 3a design).

**Core idea:** Abandon attention-based channel mixing entirely. Concatenate all 8 channels'
CNN features at each time step so the LSTM always sees every channel simultaneously.

```
(B, 8, H, W)
→ reshape → (B×8, 1, H, W)
→ Shared CNN → (B×8, 16, H', W')
→ reshape → (B, 8, 16, H', W')

Concat all channels per time step:
  permute → (B, W', 8, 16, H')
  reshape → (B, W', 8×16×H')        ← ALL CHANNELS VISIBLE AT EVERY STEP
  Linear(8×16×H' → chan_d_model=128) + ReLU   ← projection

→ LSTM(input=128, hidden=100)
→ Dense(64→32→num_classes)
```

**Parameters:** ~1M (chan_proj linear layer dominates: 8×16×54×128 ≈ 885K)

**Why this is fundamentally different:**
The LSTM input at each time step `t` is a vector containing CNN features from *all 8 channels*
at that moment. No channel is discarded before the recurrent layer. The LSTM can learn patterns
like "Fp1 alpha suppression AND T7 theta increase → depressed" — genuine multi-channel
interaction that was impossible in all previous versions.

**Known weakness:** The chan_proj linear layer (~885K params) is responsible for >80% of the
model's total parameters. With dropout=0.1 and ~7600 training chunks per fold, this creates
a bottleneck that promotes overfitting. Reducing `chan_d_model` from 128 to 64 would cut the
model to ~556K total and should be tried if further experimentation is done.

**Used in experiments:** 021 (8ch binary, focal, dropout=0.1, rnn_hidden=100), 022 (same
config, replicate run)

---

## Summary Table — Architectures

| Version | Channel mixing | Channels visible to LSTM | Params | Status |
|---------|---------------|--------------------------|--------|--------|
| `CNN_LSTM_DepCap` | None (1 ch) | 1 | 798,306 | superseded |
| `Smaller` | None (1 ch) | 1 | 256,770 | superseded by SmallerAttn |
| `SmallerAttn` | None (1 ch) | 1 | 265,218 | best single-channel model |
| `SmallerAll` | Conv3d depth kernel (rigid) | 1 (merged at conv) | 283,266 | superseded |
| `SmallerAllAttn` | Same as SmallerAll + temporal attn | 1 | 289,794 | superseded |
| `SmallerAllV2` | Per-channel CNN + global attn (static) | 1 (weighted sum) | 345,667 | superseded |
| `SmallerAllV2Attn` | Same as V2 + temporal attn | 1 (weighted sum) | 354,115 | superseded |
| `SmallerAllV3` (4a) | Per-channel CNN + per-frame attn (select) | 1 (weighted sum) | 1,531,442 | superseded |
| `SmallerAllV3` **(4b)** | Per-channel CNN + full concat + projection | **8** | **1,001,522** | **current** |

---

## Part 2 — Training Technique Evolution

### Baseline (014b): No class balance handling

Plain cross-entropy + no sampling correction. With ~63% pathological samples, the model
predicts pathological nearly always:
- Sensitivity ~93–94%, Specificity ~54–55%
- High accuracy is misleading — essentially a majority-class predictor

### Weighted Random Sampler (018)

`WeightedRandomSampler` oversamples minority (healthy) class to produce balanced batches.

**Effect:** Sensitivity ↓ ~89%, Specificity ↑ ~62%. Better balanced but still biased.
Helps most on MDD (balanced dataset) and least on SAD/AX_MALIK (no healthy controls —
sampler can't help what isn't there).

### Focal Loss (019–022)

`FocalLoss(gamma=2.0)` with per-fold class weights as alpha. Modulates the cross-entropy
by `(1−p)^gamma` to down-weight confident-and-correct predictions, forcing the model to
focus on hard examples.

**Effect vs weighted sampler alone:**
- Sensitivity ↓ further ~84–87%, Specificity ↑ to ~68–71%
- Balanced accuracy (avg of sens + spec / 2): ~76.3% — best achieved
- Focal loss + weighted sampler are complementary: sampler handles count imbalance,
  focal loss handles difficulty imbalance within each class

**Focal loss with V3 (021/022):** Instability increased. The combination of:
1. Very large chan_proj layer (~885K params)
2. Low dropout (0.1)
3. Gamma=2.0 (aggressive focus on hard examples)

...causes many folds to pick best model at epoch 1–2 and then immediately overfit. The
focal loss amplifies the cost of early overconfident wrong predictions on the validation
set, making val loss curves very spiky.

### Spatial Dropout (`nn.Dropout2d`)

Added after each CNN pooling layer. Drops entire feature maps rather than individual neurons,
forcing the model to not rely on any single filter. Provides better regularisation than
scalar dropout for convolutional layers.

### L2 Regularisation (Weight Decay)

`weight_decay=1e-4` throughout. Prevents weights from growing large. Combined with dropout,
is the primary regularisation strategy (BatchNorm was avoided: small batch sizes during
EEG training cause noisy normalisation statistics).

---

## Part 3 — Full Experiment Results

### Binary Classification (chronological)

| Exp | Model | Ch | Dropout | Loss | Chunk Acc | Subj Acc | Sensitivity | Specificity | Bal Acc |
|-----|-------|----|---------|------|-----------|----------|-------------|-------------|---------|
| 014b | Smaller | in-ear | 0.1 | CE | 80.79% ±4.77% | 82.95% ±4.55% | 94.53% ±2.36% | 55.48% ±13.5% | 75.01% |
| 014b | SmallerAll | 8ch | 0.1 | CE | 79.72% ±3.51% | 82.26% ±3.98% | 93.56% ±5.03% | 54.75% ±10.1% | 74.16% |
| 016 | Smaller | in-ear | 0.1 | CE | 76.86% ±2.84% | 77.79% ±2.74% | 86.23% ±4.29% | 63.90% ±4.66% | 75.07% |
| 018 | Smaller+WS | in-ear | 0.1 | CE | 77.44% ±4.64% | 75.87% ±6.78% | 88.87% ±7.11% | 61.96% ±8.59% | 75.42% |
| 018 | SmallerAttn+WS | in-ear | 0.1 | CE | 78.26% ±4.92% | 77.39% ±7.05% | 89.41% ±7.02% | 63.11% ±8.76% | 76.26% |
| 019 | SmallerAttn+FL | in-ear | 0.1 | FL γ2 | 77.76% ±4.64% | 78.74% ±7.50% | 84.21% ±6.97% | **68.78%** ±6.26% | **76.50%** |
| 020 | V2Attn+WS | 8ch | 0.1 | CE | **79.10%** ±4.90% | 79.83% ±5.98% | 89.43% ±6.78% | 63.42% ±11.2% | 76.43% |
| 020 | V2Attn+FL d=0.2 | 8ch | 0.2 | FL γ2 | 76.99% ±5.16% | 76.75% ±6.61% | 80.84% ±7.26% | **70.80%** ±9.98% | **75.82%** |
| 020 | SmallerAttn+FL d=0.3 | in-ear | 0.3 | FL γ2 | 77.30% ±5.29% | 76.77% ±5.94% | 84.84% ±9.51% | 66.54% ±14.6% | 75.69% |
| 021 | V3(4a)+FL | 8ch | 0.1 | FL γ2 | 78.84% ±6.39% | 80.22% ±6.94% | 83.74% ±11.0% | 70.18% ±13.4% | 76.96% |
| 021 | V3(4a)+FL (hsf/EC only) | 8ch | 0.2 | FL γ2 | 78.68% ±4.98% | 79.24% ±8.41% | 85.15% ±8.56% | 68.62% ±13.4% | 76.89% |
| **022** | **V3(4b)+FL** | **8ch** | **0.1** | **FL γ2** | **78.56%** ±3.54% | **79.39%** ±6.35% | 86.68% ±6.10% | 66.56% ±9.59% | **76.62%** |

> WS = WeightedRandomSampler. FL = FocalLoss. Bal Acc = (sensitivity + specificity) / 2.
> All: dataset=all, condition=ec+eo, 10-fold CV (016 = 6-fold). "hsf" = V3 3a design
> (rnn_hidden=330), trained on half the data (EC only), included for comparison.

### 4-Class Classification (chronological)

| Exp | Model | Ch | Chunk Acc | Subj Acc | Healthy rec. | Anxiety rec. | Depr. rec. | Comorbid rec. |
|-----|-------|----|-----------|----------|--------------|--------------|------------|---------------|
| 016 | Smaller | in-ear | 63.62% ±4.45% | 64.29% ±7.65% | — | — | — | — |
| 018 | SmallerAll+WS | 8ch | 63.46% ±9.60% | 64.57% ±11.4% | 54.64% | 57.69% | 86.06% | 61.78% |
| 018 | SmallerAllAttn+WS | 8ch | 64.76% ±8.05% | 66.09% ±8.23% | 60.36% | 45.98% | 85.04% | 70.34% |
| 019 | SmallerAttn+FL | in-ear | 66.51% ±7.32% | 67.10% ±9.13% | 50.39% | 60.90% | 89.44% | 89.31% |
| **020** | **V2Attn+WS** | **8ch** | **65.91%** ±6.63% | **68.49%** ±6.68% | 55.87% | 42.18% | **90.35%** | **79.63%** |

> No V3 4-class run exists. Experiment B from ENDGAME.md would add this row.

### Per-Dataset Chunk Accuracy (best models)

| Dataset | SmallerAttn+FL (019) | V2Attn+WS (020) | V3+FL (021) |
|---------|---------------------|-----------------|-------------|
| MDD | 89.03% | 90.20% | 88.47% |
| CANE/IDUN | 69.56% | 72.55% | 72.21% |
| SAD/AX_MALIK | 62.62% | 68.08% | 73.02% |

---

## Part 4 — Key Findings

### Architecture findings

1. **The biggest single gain: DepCap → Smaller.** ~5× fewer parameters, same accuracy.
   The original model was massively overparameterised for ~100 subjects / 8000 chunks.
   This is the primary lesson: dataset size determines the capacity ceiling.

2. **Multi-channel models never exceeded in-ear.** SmallerAll through V3(4a) all collapsed
   to effective single-channel operation (Conv3d merge or softmax gating). They were
   functionally equivalent to the single-channel models. V3(4b) is the first model that
   genuinely presents all 8 channels to the LSTM — but has only been tested in 2 runs.

3. **Attention vs LSTM: marginal (+0.82pp chunk, +1.52pp subject).** Not worth the
   parameter cost in the multi-channel setting. In the single-channel setting the
   SmallerAttn is strictly better and more compact than Smaller.

4. **V3(4b) is the first true multi-channel model, but overfits badly.** The chan_proj
   layer (~885K params) is too large. With dropout=0.1, best model is picked at epoch 1–2
   in ~4/10 folds. Reducing chan_d_model to 32–64 and/or raising dropout to 0.3–0.5 is
   the correct next step before concluding V3 has no gain over V2.

5. **In-ear matches 8-channel.** T7–T8 bipolar derivation achieves parity with all 8
   electrodes — validating the in-ear EEG hypothesis. The IDUN real in-ear data performs
   similarly to synthetic T7–T8.

### Training technique findings

6. **Class imbalance trajectory: sensitivity trades off with specificity.**
   Baseline (014b): sens=93%, spec=55%. After focal loss (021): sens=84%, spec=70%.
   Best balanced accuracy ~76.96% with V3+FL. No technique achieved both >85% sensitivity
   and >70% specificity simultaneously.

7. **Focal loss reduces sensitivity but improves balanced accuracy.** Gamma=2.0 provides
   the best specificity but creates training instability with large models (V3). Lower
   gamma (0.5–1.0) should be tried with V3.

8. **AX_MALIK is the hardest dataset** (SAD accuracy: 62–73%). All subjects are anxiety-class
   with no healthy controls, so the model never sees a healthy/pathological contrast in that
   dataset — it can only learn to predict what other datasets suggest healthy vs pathological
   looks like, then generalise.

9. **4-class is stuck at 63–67%.** ~3× random chance (25%). The 1 depression-only subject
   in CANE/IDUN makes stable depression-only learning impossible. Anxiety vs comorbid is
   also hard: spectral differences between anxiety alone and anxiety+depression are subtle.

---

## Part 5 — Current State & What to Do Next

### What has been established

- Binary classification ceiling: ~79% chunk accuracy, ~80% subject accuracy, ~77% balanced
  accuracy. This appears to be a dataset-size / signal-quality ceiling, not an architecture
  ceiling — the best and worst models differ by only ~2pp.
- The V3(4b) concat architecture is the right approach for true multi-channel modelling but
  has not been fairly evaluated due to overfitting from chan_proj.
- In-ear EEG is viable: parity with 8-channel validates the wearable hypothesis.

### Remaining experiments (from ENDGAME.md, priority order)

**Experiment A — EEGNet baseline (HIGH, ~3 days)**
The most-cited compact EEG network (Lawhern et al. 2018). Works on raw EEG (no STFT).
Without it, reviewers will ask "why not EEGNet?". Expected ~73–78% — confirms spectrogram
CNN-LSTM is competitive. Implement in `thesis/model.py`, train binary in-ear and 8ch.

**Experiment B — V3(4b) 4-class + focal loss (MEDIUM, ~1 day)**
The best binary model is V3+FL. No 4-class V3 run exists. Run same config as 021 with
`--class-mode 4`. Expected ~67–70% chunk. Closes the 4-class table.

**Experiment C — V3 with reduced chan_d_model (OPTIONAL)**
Run V3(4b) with `chan_d_model=32` and `dropout=0.3` to properly evaluate true multi-channel
benefit without overfitting artefacts. If this shows a gain over V2Attn, it becomes the
strongest architecture finding of the thesis.

### Design principles (for thesis Chapter 4.3)

1. **Separate spectral extraction from spatial mixing.** CNN answers "what pattern is
   present?"; channel mixing answers "how do electrodes combine for the decision?".
2. **Weight sharing across channels.** EEG biomarkers are spectral phenomena; the detector
   should be electrode-agnostic.
3. **LSTM for temporal consistency.** Transient events are clinically less meaningful than
   sustained patterns.
4. **Don't discard channels before the recurrent layer.** Any aggregation that collapses N
   channels to 1 before the LSTM is a channel *selector*, not a multi-channel model. The
   LSTM input must contain all channels to enable cross-channel reasoning.
5. **Dataset size sets the capacity ceiling.** ~100 subjects / 8000 chunks cannot support
   models above ~300–400K parameters without aggressive regularisation (dropout ≥0.3, L2).