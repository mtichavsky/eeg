# Explainability: which channels and bands drive the predictions?

Methodology record for Section VII of the paper. Implemented in `thesis/explain/`, driven by
`python main.py explain <run-dir>`, tested in `tests/test_explainability.py`.

## Geometry (verify before trusting any result)

An earlier draft of this note had two facts wrong. Both would corrupt every result silently,
so they are stated here with their source of truth:

| Fact | Value | Source |
|---|---|---|
| Spectrogram shape | `(72, 41)` — **not** `(129, 41)` | `EXPECTED_SPECTROGRAM_SHAPE`, `thesis/stft.py:72` |
| Channel order | `Fp1, Fp2, C3, Cz, C4, T7, T8, O2/Oz` | `CANONICAL_CHANNEL_ORDER`, `thesis/dataset.py:39` |

So **channel index → electrode is `0=Fp1, 1=Fp2, 2=C3, 3=Cz, 4=C4, 5=T7, 6=T8, 7=O2/Oz`**.
The temporal pair that the in-ear surrogate derives from is indices **5 and 6**.

Frequency: bin `i` is centred at `i × 0.9766 Hz` (`FREQ_BIN_WIDTH_HZ`). Bands are computed, never
hardcoded — `band_bins()` derives them from the STFT constants. They come out as delta 2–4
(3 bins), theta 5–8 (4), alpha 9–13 (5), beta 14–30 (17), gamma 31–71 (41).

## How AllTransformerV4 handles channels

Input `(B, 8, 72, 41)`. A weight-shared CNN runs per channel, producing **80 tokens = 8 channels
× 10 time frames**; each gets a `chan_embedding[c]` and `time_embedding[t]`; a 2-layer,
4-head transformer attends over all 80; the model **mean-pools every token** and classifies
(`thesis/model.py:744-816`). Token index is **`channel * 10 + time_frame`**.

Two consequences:

- Channels are explicit, separable units, so they can be ablated cleanly.
- The readout is a mean-pool, **not** a CLS token, so there is no single "decision token" whose
  attention can be read off. Attention answers must be aggregated (see rollout below).

## Checkpoint compatibility

`models/*.pth` and everything under `../experiments/` predate the Jul-30 frequency-axis fix:
their `proj.weight` is `(64, 864)` where current code needs `(64, 416)`. **They are unusable.**
`create_model()` loads with `strict=False`, so a mismatch is dropped silently and left randomly
initialised — `load_fold_model()` therefore re-checks and refuses. Valid checkpoints live in
`thesis-text/paper/experiments/`.

One gap: the paper's headline 8-channel binary row comes from the `atv4-lowlr_oex` re-run whose
checkpoints were not kept. Either retrain that config or report ablations against
`binary/8channel/alltransformer-binary` (the earlier `lr=5e-4` run) and say so.

## The experiments

### Channel ablation (primary)

Most defensible for review because it answers "contribution" in the metric the paper already
reports. For each channel, **drop its 10 tokens from the sequence entirely**: the transformer
handles variable-length input and the mean pool renormalises, so the channel genuinely does not
participate. This beats zeroing the spectrogram, which leaves `chan_embedding[c]` and a "this
channel is silent" cue in place. Zeroing is run separately as a robustness check —
`--experiment channel-occlusion`.

Run **per fold** and report mean ± std, not a single checkpoint. That turns "channel X cost 4%"
into a claim with error bars.

**State the redundancy caveat.** EEG channels are spatially correlated: if Fp1 and Fp2 carry
overlapping signal, ablating one alone shows little drop even when the pair is jointly critical.
**Small drop ≠ unimportant.** Grouped region ablation (frontal / central / temporal / occipital)
and leave-one-**in** both run alongside for exactly this reason.

Retraining on reduced montages answers the complementary question — what the channels *carry*,
rather than what this trained model *uses*. Use `--channel T7,T8` and friends.

### Frequency-band occlusion

Occlusion replaces a band with the **per-bin mean of the training folds' spectrograms**, never
with zero: inputs are `log1p|STFT|`, so zero means "no power at all", far outside the data
distribution, and would conflate "this band matters" with "this input is out of distribution".

**Bands differ enormously in width** — gamma is 41 bins, delta 3 — so raw drops are not
comparable. Each band is therefore also compared against width-matched control windows placed
elsewhere in the spectrum. Gamma is a special case: no disjoint 41-bin window fits in a 72-bin
spectrum, so its control list is empty by construction and its drop must be read per-bin.

The channel × band grid doubles as the artifact audit: ocular artifact lives in Fp1/Fp2 at low
frequency, so a frontal × delta hotspot would mean the model reads blinks rather than pathology.

### Hemispheric mirror

Swap Fp1↔Fp2, C3↔C4, T7↔T8; leave Cz and O2/Oz. The montage's pooled power spectrum is
unchanged, so only lateralisation is destroyed — the family frontal alpha asymmetry belongs to.
A drop is evidence of asymmetry use; no drop is evidence of bilateral power use, and both are
reportable.

### Attention rollout

`nn.TransformerEncoderLayer.forward` hardcodes `need_weights=False`, so the weights are never
computed on the normal path and a plain forward hook captures nothing. `capture_attention()`
re-runs attention from a forward-pre-hook, reproducing `layer.norm1(x)` because the model uses
`norm_first=True` (so weights are over LayerNorm'd tokens, not raw features).

With 2 layers, `R = (0.5·A₂ + 0.5·I)(0.5·A₁ + 0.5·I)`. Because the readout is a mean-pool with no
CLS row, per-token influence on the decision is the **column mean of R**, collapsed 80 → 8.

### Integrated Gradients

Hand-rolled (`captum` is not a dependency and the model takes a single tensor input), midpoint
rule, same training-mean baseline as the occlusion reference so the two share a notion of
"absent". The **completeness axiom** — attributions sum to `f(x) − f(baseline)` — is checked
every run; a large residual means too few steps and invalidates the map.

### Caveats to state in the paper

- **Attention is not explanation** (Jain & Wallace 2019). Ablation stays primary; rollout
  corroborates.
- Averaging over heads can hide head specialisation.
- CV folds share training data and are **not independent**, so the naive paired t-test
  over-rejects. Wilcoxon signed-rank is primary; the Nadeau-Bengio corrected t is reported as
  the conservative alternative; Benjamini-Hochberg FDR is applied within each family (channels,
  bands) separately.
- The convincing story is **agreement**: if rollout, integrated gradients, and leave-one-out
  rank the same channels at the top, that is a strong three-method result.

## Verification gate

For every fold, the *unablated* re-evaluation must reproduce the metrics stored inside the
checkpoint (`val_metrics`, written by `train_one_fold`). Fold membership depends on a single
stateful `RandomState` replayed across dataset-preparation calls in a fixed order; if the replay
diverges, the validation set is a different set of subjects and every ablation number is void.
The `explain` command checks this automatically and logs an error per mismatched fold.
