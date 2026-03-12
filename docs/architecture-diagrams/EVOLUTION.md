# Multi-Channel Model Architecture Evolution

This document traces the design decisions behind the multi-channel EEG models:
`SmallerAll`, `SmallerAllV2`/`SmallerAllV2Attn`, and `SmallerAllV3`.

`SmallerAllV3` went through two distinct designs. The first (attention-based)
was a natural extension of V2 but turned out to share its fundamental flaw.
The second (concat-based, current) takes a different direction entirely and is
the version in the codebase today.

---

## Generation 1 — `SmallerAll` / `SmallerAllAttn`

**Core idea:** treat the 8 EEG channels as the *depth* dimension of a 3D
volume and collapse them with a single Conv3d kernel.

```
(B, 8, H, W)
→ unsqueeze → (B, 1, 8, H, W)
→ Conv3d(1→32, kernel=(8,10,10))     ← kernel depth = 8 = all channels
→ AdaptiveMaxPool3d((1,None,None))   ← collapse depth to 1
→ squeeze → (B, 32, H', W')
→ Conv2d(32→16) → MaxPool → Dropout
→ (B, W', H'×16)  ← time sequence
→ LSTM(rnn_hidden=64) / Attention
→ classifier (→32→16→num_classes)
```

**What it gets right:**
Simple, efficient. The 3D conv can in principle learn *which linear combination*
of the 8 channel spectra is most informative.

**The problem — rigid all-or-nothing spatial filter:**
The Conv3d kernel spans *all 8 channels* with a fixed depth of 8. This means:

1. The model learns exactly **one** set of channel weights, shared across all
   frequency bins and time frames. A channel that's useful only in the alpha
   band (8–13 Hz) contributes the same weight as one that's uniformly
   informative.
2. After the first layer the 8-channel structure is **destroyed** — all
   subsequent computation is on a single merged feature map. There is no
   further cross-channel reasoning.
3. No weight sharing across channels: the detector learned for frontal theta
   in Fp1 cannot be reused for Fp2, even though the spectral signature is the
   same.

**Parameters:** ~148K–290K (rnn_hidden-dependent).

---

## Generation 2 — `SmallerAllV2` / `SmallerAllV2Attn`

**Core idea:** decouple *spectral feature extraction* from *spatial channel
mixing* using a weight-shared per-channel CNN followed by learned cross-channel
attention.

```
(B, 8, H, W)
→ reshape → (B×8, 1, H, W)          ← process each channel independently
→ Shared CNN (same weights for all 8 channels)
→ (B×8, 16, H', W')
→ reshape → (B, 8, 16, H', W')

Cross-channel attention (static — one set of weights per clip):
  mean over W' → (B, 8, 16, H')     ← TIME AXIS COLLAPSED HERE
  flatten → (B, 8, 16×H')
  Linear → (B, 8, chan_d_model=64)
  TransformerEncoder (d_model=64, nhead=4)
  Gate + softmax → weights (B, 8, 1)
  Weighted sum → (B, 16, H', W')    ← merged back

→ permute → (B, W', H'×16)
→ LSTM(64) / Attention(128)
→ classifier (→32→16→num_classes)
```

**What improves:**
- **Weight sharing:** the CNN learns one universal spectral detector and applies
  it to every electrode. Frontal theta is detected the same way in Fp1 and Fp2.
- **Learnable channel importance:** the TransformerEncoder computes a data-
  driven set of channel weights, replacing the fixed linear combination of
  Conv3d depth.
- **Ongoing cross-channel reasoning** (within the attention block).

**The remaining problem — static channel weights:**
`mean(dim=4)` collapses W' *before* attention, producing **one** set of channel
weights for the entire 10-second clip. A blink artifact contaminates Fp1/Fp2
for ~0.5 s but the model cannot down-weight those channels only during the
contaminated frames. A channel informative only in a specific frequency×time
tile gets the same global weight as one that's consistently informative.

**Parameters:** ~345K (LSTM) / ~354K (Attention).

---

## Generation 3a — `SmallerAllV3` (initial design, superseded)

> **Status: superseded.** Experiment `all3-021-all-binary-smallerAllV3-focal`
> used this design. It achieved 78.84% ± 6.39% chunk accuracy — statistically
> identical to V2Attn but with higher variance and 3 folds best-at-epoch-1/2,
> indicating overfitting. The design was abandoned; see Generation 3b below.

**Core idea:** preserve the time axis *through* the cross-channel attention
step so the model can compute **per-time-frame channel importance**.

```
(B, 8, H, W)
→ reshape → (B×8, 1, H, W)
→ Shared CNN (identical to V2)
→ (B×8, 16, H', W')
→ reshape → (B, 8, 16, H', W')

Time-aware cross-channel attention:
  permute → (B, W', 8, 16, H')
  reshape → (B×W', 8, 16×H')        ← FOLD W' INTO BATCH, NOT MEAN-POOL
  Linear → (B×W', 8, chan_d_model=128)
  TransformerEncoder (d_model=128, nhead=8)
  Gate + softmax → (B×W', 8, 1)     ← different weights per time frame
  Weighted sum → (B×W', 128)        ← STILL COLLAPSES 8 CHANNELS TO 1
  reshape → (B, W', 128)

→ LSTM(input=128, hidden=330)
→ classifier (→64→32→num_classes)
```

**Why it was abandoned — the weighted-sum still collapses channels:**
V3a fixed V2's static-weight problem (attention now per time frame) but kept
the weighted *sum* to merge channels. Softmax over 8 channels with a single
gate means the model effectively selects the "best" channel at each time step
and discards the rest. If the gate learns to put 0.7 weight on Fp1 and 0.04
on each other channel, the LSTM is back to seeing one channel. This is why
every attention-based model in this line matched single-channel in-ear
performance (~78–79%): they all converged to channel selection, not channel
combination. Additionally, LSTM(hidden=330) was oversized for the dataset,
contributing to the epoch-1-best overfitting pattern.

**Parameters:** ~890K (before LSTM shrink).

---

## Generation 3b — `SmallerAllV3` (current design)

**Core idea:** abandon attention-based channel mixing entirely. Instead,
concatenate all 8 channels' CNN features at each time step so the LSTM
*always* sees every channel simultaneously and can learn any cross-channel
combination.

```
(B, 8, H, W)
→ reshape → (B×8, 1, H, W)
→ Shared CNN (identical to V2/V3a)
→ (B×8, 16, H', W')
→ reshape → (B, 8, 16, H', W')

Concat all channels per time step:
  permute → (B, W', 8, 16, H')
  reshape → (B, W', 8×16×H')        ← ALL CHANNELS VISIBLE AT EVERY STEP
  Linear(8×16×H' → chan_d_model=128) + ReLU

→ LSTM(input=128, hidden=100)        ← all 8 channels feed in together
→ classifier (→64→32→num_classes)
```

**Why this is fundamentally different:**
The LSTM input at time step `t` is a single vector containing CNN features
from *all 8 channels* at that moment. No channel is discarded or down-weighted
before the recurrent layer. The LSTM can learn patterns like "Fp1 shows alpha
suppression *and* T7 shows theta increase → depressed" — genuine multi-channel
interaction that was impossible in all previous versions.

The architecture diagram (`SmallerAllV3_architecture.drawio`) reflects the 3a
design; it has not been updated to 3b.

**Parameters:** ~1M (chan_proj dominates at ~885K due to 8× wider input).
Reducing `chan_d_model` from 128 to 64 cuts this to ~556K total.

---

## Summary Table

| Version | Channel mixing | Channels visible to LSTM | Params | Status |
|---------|---------------|--------------------------|--------|--------|
| `SmallerAll` | Conv3d depth kernel (rigid) | 1 (merged at conv) | ~148–290K | superseded |
| `SmallerAllV2` | Per-channel CNN + global attn (static weights) | 1 (weighted sum) | ~345K | superseded |
| `SmallerAllV2Attn` | Same as V2 + temporal attention | 1 (weighted sum) | ~354K | superseded |
| `SmallerAllV3` (3a) | Per-channel CNN + per-frame attn (soft select) | 1 (weighted sum) | ~890K | superseded |
| `SmallerAllV3` (3b) | Per-channel CNN + full concat + projection | **8** | ~1M | **current** |

---

## Design Principles

1. **Separate spectral extraction from spatial mixing.** The CNN answers "what
   spectral pattern is present?"; the channel mixing stage answers "how do all
   electrodes combine to produce the decision?".
2. **Weight sharing across channels.** Depression/anxiety biomarkers are
   spectral phenomena; the detector should be electrode-agnostic.
3. **LSTM for temporal consistency.** Transient spectral events are clinically
   less meaningful than sustained patterns. The LSTM captures whether a pattern
   persists across the 10-second window.
4. **Don't discard channels before the recurrent layer.** Any aggregation that
   collapses N channels to 1 before the LSTM is functionally a channel selector,
   not a multi-channel model. The LSTM input must contain all channels.
