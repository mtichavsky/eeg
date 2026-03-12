# Multi-Channel Model Architecture Evolution

This document traces the design decisions behind the three generations of
multi-channel EEG models: `SmallerAll`, `SmallerAllV2`/`SmallerAllV2Attn`,
and `SmallerAllV3`.

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

## Generation 3 — `SmallerAllV3`

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
  Weighted sum → (B×W', 128)
  reshape → (B, W', 128)            ← time axis restored!

→ LSTM(input=128, hidden=330)       ← temporal aggregation
→ classifier (→64→32→num_classes)   ← DepCap-style wider head
```

**What improves over V2:**
- Channel weights are computed **per time step**: the model can learn that
  Fp1/Fp2 should be down-weighted during blink frames and up-weighted during
  clean frames, automatically, without explicit artifact detection.
- Frontal alpha asymmetry (Fp1 vs Fp2), a key depression biomarker, is a
  *time-varying* phenomenon — the model now has the inductive bias to detect it.
- The `B×W'` fold is a batching efficiency trick, not semantic mixing: the
  Transformer attends only within each `(b, t)` row across 8 channel tokens.
  Rows at different time steps are independent. After the weighted sum the
  temporal structure is fully restored.

**Trade-off:** larger model (~890K params). The bulk of the increase comes from
the wider LSTM hidden size (330 vs 64) and wider classifier head, not from the
attention mechanism itself.

---

## Summary Table

| Version | Channel mixing | Time-aware? | Params | Key limitation |
|---------|---------------|-------------|--------|----------------|
| `SmallerAll` | Conv3d depth kernel (rigid) | No | ~148–290K | One fixed spatial filter; no ongoing cross-channel reasoning |
| `SmallerAllV2` | Per-channel CNN + global cross-channel attn | No | ~345K | Channel weights are static (W' mean-pooled before attention) |
| `SmallerAllV3` | Per-channel CNN + per-frame cross-channel attn | **Yes** | ~890K | Larger model; higher memory/compute |

---

## Design Principles (all three share)

1. **Separate spectral extraction from spatial mixing.** The CNN answers "what
   spectral pattern is present?"; the channel mixing stage answers "which
   electrode is most relevant right now?".
2. **Weight sharing across channels.** Depression/anxiety biomarkers are
   spectral phenomena; the detector should be electrode-agnostic.
3. **LSTM for temporal consistency.** Transient spectral events are clinically
   less meaningful than sustained patterns. The LSTM captures whether a pattern
   persists across the 10-second window.
