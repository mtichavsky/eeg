# Bipolar surrogate ablation: Fp2−Fp1 vs C4−C3 vs T8−T7

## Context

The paper's in-ear result (`CNN-AttnS`, 77.0% chunk / 79.2% subject, Table II) rests on a surrogate:
for MDD and SAD the "in-ear" channel is the bipolar derivation T8−T7, and only CANE has real in-ear
(IDUN) recordings. The choice of T7/T8 is justified by the literature (Tremmel 2024, Moumane 2024),
not tested.

This experiment tests it: train the same single-channel model on three interhemispheric bipolar
derivations and compare. It tells us whether the temporal pair is special or whether *any*
bipolar channel reaches the same accuracy. Either result is reportable in Section VII:

- T8−T7 clearly best → the surrogate choice is supported by data, not only by citation.
- All three similar → the in-ear result reflects "one bipolar channel is enough", and the
  specific location matters less than the paper implies.
- Fp2−Fp1 best → must be checked against ocular artifact before being interpreted (see below).

**EC and EO are run as separate experiments**, not only combined. Fp1/Fp2 sit directly above the
eyes, so horizontal eye movement dominates Fp2−Fp1 in eyes-open recordings; an eyes-closed vs
eyes-open split is the cheapest way to tell pathology signal from eye movement.

Binary classification only (`--class-mode 2`), `--dataset all`, 10-fold CV (fold count is not
negotiable — cut configurations, never folds).

## Facts established during exploration

| Fact | Where |
|---|---|
| Bipolar derivation is hardcoded as T8−T7 behind `channel == "in-ear"` | EDF: `thesis/dataset.py:132-139`; CANE: `thesis/dataset.py:~754` and `~785` |
| `--channel in-ear` swaps CANE (8-ch Mindrove headset) for IDUN (real in-ear) | `main.py:638` (`--dataset cane`), `main.py:719` (`--dataset all`) |
| MDD/SAD bipolar path: pick T7,T8 → per-channel detrend → subtract. **No CAR** | `thesis/dataset.py:132-139` |
| CANE bipolar path: CAR over all 8 (if `apply_car`) → per-channel filter/artifact removal → subtract. CAR cancels exactly in a difference, so this is equivalent | `thesis/dataset.py:~740-790` |
| Channel names after mapping are the same in all three datasets for Fp1, Fp2, C3, C4, T7, T8 (MDD maps T3→T7, T4→T8) | `MDDDataset.CHANNEL_MAPPING`, `CANEDataset.CHANNEL_MAPPING`, `SADDataset.CHANNEL_MAPPING` |
| `in_channels = 8 if channel == "all" else 1` — a bipolar channel is correctly 1 channel with no change | `main.py:859`, `main.py:1083` |
| `is_inear=(channel == "in-ear")` only drives sign-flip augmentation in `FlattenedRawEEGDataset`; spectrogram models ignore it | `thesis/data_preparation.py:125,260,384`; `thesis/dataset.py:1782` |
| Folds come from one `RandomState(42)` shuffled per class list, in dataset order MDD → CANE → SAD. IDUN has a different subject set from CANE, so **IDUN runs and headset runs get different folds** (SAD's too, since the RNG state has advanced differently) | `main.py:595`, `thesis/data_preparation.py:449` |
| Headline in-ear config (`experiments/binary/in-ear/cnnattns-binary_wve/results.txt`) | `--model CNNAttnS --batch-size 64 --epochs 100 --lr 1e-4 --dropout 0.05 --weight-decay 5e-5 --focal-loss --patience 20 --val-every 1`, cosine LR on, Adam, IDUN quality threshold 30 |
| `main.py` has a `channel in ["T7","T8"] → "T3"/"T4"` remap before MDD loading (`main.py:694`). It matches exact strings only, so bipolar names pass through untouched. Do not extend it | `main.py:692-716` |
| Per-dataset chunk accuracy is stored in each checkpoint: `torch.load(fold_N_best.pth)["val_metrics"]["per_dataset"]` | `main.py:255-267`, `main.py:404-414` |

## Design

### Channel naming

Add three new `--channel` values, each meaning **first electrode minus second**, right minus left
to match the existing T8−T7 convention (sign is irrelevant for spectrograms, but keep it
consistent):

| `--channel` | Derivation | CANE source |
|---|---|---|
| `Fp2-Fp1` | Fp2 − Fp1 | 8-ch headset |
| `C4-C3` | C4 − C3 | 8-ch headset |
| `T8-T7` | T8 − T7 | 8-ch headset |
| `in-ear` (unchanged) | T8 − T7 for MDD/SAD | **IDUN** |

Use an explicit whitelist, not string splitting on `-` — `in-ear` also contains a hyphen.

`T8-T7` on MDD and SAD must produce **bit-identical** chunks to `in-ear`. The only thing that
differs between the `T8-T7` and `in-ear` runs is where CANE comes from (headset T8−T7 vs real
IDUN). That makes the pair a clean "surrogate vs real in-ear on CANE" comparison for free.

### Code changes

1. **`thesis/dataset.py`** — add near `CANONICAL_CHANNEL_ORDER`:

   ```python
   BIPOLAR_CHANNELS: dict[str, tuple[str, str]] = {
       "Fp2-Fp1": ("Fp2", "Fp1"),
       "C4-C3": ("C4", "C3"),
       "T8-T7": ("T8", "T7"),
   }

   def bipolar_pair(channel: Optional[str]) -> Optional[tuple[str, str]]:
       """Return the (minuend, subtrahend) electrodes for a bipolar channel spec.

       ``"in-ear"`` resolves to ``("T8", "T7")``. Returns ``None`` for any non-bipolar spec.
       """
   ```

   Then replace every `channel == "in-ear"` **in the EDF and CANE preprocessing paths** with
   `bipolar_pair(channel)`, using the returned electrodes instead of the hardcoded T7/T8:
   - `load_and_preprocess_edf_file` (~line 132): pick `[a, b]`, detrend each, return `a − b`.
     Keep the "no CAR" behaviour and its comment (CAR over two electrodes is meaningless).
     Note the current code relies on `raw.pick(["T7","T8"])` returning them in that order —
     index by name, not position.
   - `CANEDataset.load_and_preprocess_cane_raw_file` (~754 and ~785): `channels_to_use = [a, b]`,
     subtract `processed[0] − processed[1]`. Update the log line to name the pair.
   - `channel_names` in `MDDDataset.__init__` (~270), `CANEDataset.__init__` (~502),
     `SADDataset.__init__` (~977): use `[channel]` for the new bipolar specs; leave `["in-ear"]`
     for `in-ear`.
   - `IDUNDataset`: no change. It must never receive a bipolar spec other than `in-ear`.

2. **`thesis/data_preparation.py`** — `is_inear=(channel == "in-ear")` at lines 125 and 260 becomes
   `is_inear=bipolar_pair(channel) is not None`. Polarity is as arbitrary for Fp2−Fp1 as for T8−T7,
   so raw-EEG models should get the same sign-flip augmentation. No effect on `CNNAttnS`.

3. **`main.py`** — **no change** to the IDUN swap condition: it stays `channel == "in-ear"` exactly,
   so the new specs load CANE from the headset. Add one `logger.info` line when a bipolar spec is
   used with CANE, stating that CANE comes from the headset, not IDUN. That keeps logs unambiguous.

4. **`thesis/cli.py:40`** — add `list(BIPOLAR_CHANNELS)` to the `--channel` choices. Extend the
   help text: "`Fp2-Fp1`, `C4-C3`, `T8-T7`: bipolar derivations on all datasets, CANE from the
   8-channel headset (unlike `in-ear`, which uses IDUN for CANE)."

5. **Out of scope** — `thesis/inference.py` and `api/`. No API model uses these channels. Don't
   touch them.

6. **`CLAUDE.md`** — add one bullet under "Recent Major Changes" and document the new channel
   values next to `in-ear` in the Datasets / Model Input sections.

### Tests (`tests/test_bipolar_channels.py`)

- `bipolar_pair` returns the right tuples for the three specs and `in-ear`, and `None` for
  `"all"`, `"Fp1"`, `None`.
- CLI parser accepts all three specs.
- **Equivalence test (skipped when data is absent, via `pytest.mark.skipif` on `MDD_DIR.exists()`):**
  for one MDD EDF file, `load_and_preprocess_edf_file(..., channel="T8-T7")` is `torch.equal` to
  `channel="in-ear"`. Same for one SAD file.
- **Derivation test (same skip rule):** for one MDD file, `channel="C4-C3"` equals the difference
  of separately preprocessed C4 and C3 *without CAR*. This needs a small reference computation in
  the test (bandpass → notch → detrend per channel → subtract → chunk → z-score), mirroring the
  function's steps.
- CANE: for one CANE file, `channel="T8-T7"` returns shape `(n, 1, 5000)` with finite values, and
  differs from `channel="Fp2-Fp1"`.

`make format`, `make typecheck`, `make test` must pass.

### Smoke check before GPU time

Locally, with `--test-mode --epochs 1 --n-folds 10`, for each of `Fp2-Fp1`, `C4-C3`, `T8-T7`:
- the log contains no "Using IDUN real in-ear dataset" line;
- the log confirms CANE is loaded from the headset;
- spectrogram shape is `(1, 72, 41)`.

## Runs

All runs: `--dataset all --class-mode 2 --model CNNAttnS --n-folds 10`, plus the headline
hyperparameters, **unchanged and untuned**:

```
--batch-size 64 --epochs 100 --lr 1e-4 --dropout 0.05 --weight-decay 5e-5 \
--focal-loss --patience 20 --val-every 1
```

### Required: 4 montages × 2 conditions = 8 runs

| Dir | `--channel` | `--condition` |
|---|---|---|
| `all_040_fp2-fp1_ec` | `Fp2-Fp1` | `ec` |
| `all_040_fp2-fp1_eo` | `Fp2-Fp1` | `eo` |
| `all_040_c4-c3_ec` | `C4-C3` | `ec` |
| `all_040_c4-c3_eo` | `C4-C3` | `eo` |
| `all_040_t8-t7_ec` | `T8-T7` | `ec` |
| `all_040_t8-t7_eo` | `T8-T7` | `eo` |
| `all_040_inear_ec` | `in-ear` | `ec` |
| `all_040_inear_eo` | `in-ear` | `eo` |

Version `040` avoids the in-flight `all_03x` runs. The `inear` rows are the real-IDUN reference,
rerun per condition so they are comparable to the rest.

### Optional: `ec+eo` for all three headset montages = 3 runs

`all_040_{fp2-fp1,c4-c3,t8-t7}_ec+eo`. These connect directly to Table II: the existing
`cnnattns-binary_wve` run *is* the `in-ear ec+eo` row, so it doesn't need rerunning. Run these
only if GPU time allows after the required eight.

### Launcher

`run-bipolar.sh`, modeled on `run-all-experiments.sh` (same `ml` modules, `EEG_DATA_DIR`,
`launch <gpu> <name> args...` helper, `nohup`, `stdout.log` per run). Launch in waves of one run
per available GPU, with `wait` between waves: EC wave first, then EO, then optional `ec+eo`.

## Analysis

Write `helpers-print/bipolar_ablation_summary.py` to read the 8 (or 11) run directories. For each
run, read `results.txt` and every `fold_N_best.pth` (`val_metrics`), then produce:

1. **Main table** — rows = montage, column groups = EC / EO; cells = chunk acc, subject acc,
   sensitivity, specificity as `mean ± std` over folds.
2. **Per-dataset chunk accuracy** (MDD / CANE / SAD) per run, from `val_metrics["per_dataset"]`.
   This matters more than the pooled number here:
   - on MDD and SAD, T8−T7 and in-ear are identical inputs, so any difference there comes only from
     training on different CANE data;
   - on CANE, `T8-T7` vs `in-ear` is the direct headset-surrogate vs real-in-ear comparison.
3. **Paired statistics across the three headset montages** (`Fp2-Fp1`, `C4-C3`, `T8-T7`). They
   share identical folds, so compare per fold: Wilcoxon signed-rank on the 10 per-fold chunk
   accuracies, reported with the Nadeau–Bengio corrected t-test (folds share training data and
   are not independent). Apply Benjamini–Hochberg correction within each condition over the three
   pairwise comparisons.
4. **`in-ear` rows are unpaired** with the headset rows (different folds — see Facts). Report them
   descriptively, and say so in the text.

**Fold-identity check (do this before any paired test):** confirm the `Val subjects (N): [...]`
log lines match fold by fold across the three headset montages, per condition. They should,
because the subject lists don't depend on which electrodes are picked. If a CANE file raises
`NaNValuesError` for one channel pair but not another, subject lists diverge and pairing breaks.
In that case, stop and report it.

**Eye-movement check for Fp2−Fp1:** if Fp2−Fp1 is competitive, compare its EC and EO results. A
gain that appears only in EO, or a much larger EO gain, points to eye movement rather than
pathology. State this whichever way it comes out.

## What gets printed in the paper

- One compact table in Section VII: montage × {EC, EO}, chunk and subject accuracy, with the
  in-ear (IDUN) rows marked as unpaired.
- 2–4 sentences covering: which pair wins and whether the difference is significant; the CANE-only
  headset-T8−T7 vs IDUN comparison; the EC/EO eye-movement check.
- Caveats to state once: the three pairs differ in electrode spacing (T7–T8 spans ~80% of the
  ear-to-ear arc, C3–C4 ~40%, Fp1–Fp2 much less), which changes the spectral content of the
  difference signal. An interhemispheric bipolar channel also emphasizes asymmetric activity,
  which links to the frontal alpha asymmetry biomarker cited in the introduction.

## Verification

- Unit tests above pass, including the `T8-T7 ≡ in-ear` equivalence on MDD and SAD.
- Smoke runs confirm the CANE source (headset vs IDUN) from logs.
- `make format`, `make typecheck`, `make test` pass.
- After the GPU runs: all 8 run directories have 10 `fold_N_best.pth` files and a `results.txt`.
  The fold-identity check passes for the headset montages.
