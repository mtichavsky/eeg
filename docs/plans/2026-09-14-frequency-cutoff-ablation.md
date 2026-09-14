# Frequency-cutoff ablation: retrain with the spectrogram cut at 30 Hz

## Context

Part of the explainability design for Section VII of the paper. See
`docs/plans/2026-09-14-explainability-design.md` (experiment **B**).

A reviewer can object that the models detect **muscle tension, not brain activity**. Scalp EMG
from jaw, forehead and neck muscles dominates the high-frequency EEG spectrum (Whitham et al.
2007; Goncharova et al. 2003), and anxious subjects plausibly tense more. The current spectrograms
keep everything up to 70 Hz (`FREQ_CUTOFF_HZ = 70.0`, `thesis/stft.py:33`; shape `(72, 41)`).

The test: retrain the two headline models on spectrograms cropped at **30 Hz** (shape `(31, 41)`),
so gamma is never seen during training or validation.

- **Accuracy holds** → the result doesn't depend on the EMG-dominated range.
- **Accuracy drops substantially** → the models relied on content above 30 Hz, which may be
  muscle activity. This has to be reported.

Retraining, not post-hoc occlusion, is deliberate. The model never sees an unnatural input, so the
extrapolation critique of permutation methods (Hooker et al. 2021, ROAR 2019) doesn't apply.

Binary only (`--class-mode 2`), `--dataset all`, `--condition ec+eo`, 10-fold CV (fold count is not
negotiable).

## Facts established during exploration

| Fact | Where |
|---|---|
| The cutoff is a module constant; `NUM_FREQ_BINS` and `EXPECTED_SPECTROGRAM_SHAPE` are derived from it at import time | `thesis/stft.py:33-73` |
| The crop happens in one place: `np.log1p(np.abs(Zxx))[freqs <= FREQ_CUTOFF_HZ]` | `compute_log_spectrogram`, `thesis/stft.py:112-135` |
| `compute_log_spectrogram` is called only from `SpectrogramDataset.convert_to_spectrograms` (static), used by training via `SpectrogramDataset._get_item` (`dataset.py:1554`) and by inference (`inference.py:318`) | `thesis/dataset.py:1485-1540` |
| Spectrogram shape is asserted against `EXPECTED_SPECTROGRAM_SHAPE` in data preparation (generic EDF helper, CANE, IDUN) | `thesis/data_preparation.py:137`, `:274`, `:398` |
| Models are built from `spec_shape` and size their layers with a dummy forward, so a different height propagates automatically | `main.py:794` → `create_model(spec_shape=...)`; `thesis/model.py` `CNNLSTM.__init__` (dummy `torch.zeros(1, in_channels, *input_shape)`) |
| `main.py run` also hardcodes `EXPECTED_SPECTROGRAM_SHAPE` | `main.py:1101` |
| CNN height arithmetic (conv 10×10 stride 2 → pool 2 stride 1 → conv 5×5 → pool 2 stride 1): `H' = floor((H − 10) / 2) − 5`. H=72 → 26 (matches the paper). **H=31 → 5**, so the feature vector is 16×5 = 80 instead of 416. **Minimum H = 22 (cutoff ≥ ~20.5 Hz)**; a 20 Hz cut (H=21) gives H'=0 and cannot be built | `thesis/model.py` CNNLSTM |
| DataLoader workers use forkserver (commit `305a272`): workers re-import modules, so **overriding a module-level constant at runtime would not reach them**. The cutoff must live on the dataset instance, which is pickled to workers | `thesis/dataset.py` `_PicklableLRUCacheMixin` |
| Parameter threading template: `--optimizer` (commit `d50489d`) goes `args` → `train()` → `train_cross_validation(optimizer_name=...)` as an explicit typed parameter | `main.py:531`, `main.py:1021` |
| Torch is not seeded; only fold assignment is (`RANDOM_SEED = 42` for numpy). Two trainings of the same config differ by run-to-run noise, but get **identical folds** | `main.py:61`, `main.py:595` |
| New CLI args are logged to the console and `results.txt` automatically via `vars(args)` | `main.py` `train()` |

## Design

### 1. `thesis/stft.py`: make the cutoff a parameter, keep the constants as defaults

```python
def num_freq_bins(freq_cutoff_hz: float = FREQ_CUTOFF_HZ) -> int:
    """Number of STFT bins whose centre frequency is <= freq_cutoff_hz."""
    return int(freq_cutoff_hz / FREQ_BIN_WIDTH_HZ) + 1

def spectrogram_shape(freq_cutoff_hz: float = FREQ_CUTOFF_HZ) -> tuple[int, int]:
    """Spectrogram (frequency bins, time frames) for a canonical chunk at this cutoff."""
    return (num_freq_bins(freq_cutoff_hz), NUM_TIME_FRAMES)

def compute_log_spectrogram(
    signal: np.ndarray, source_fs: float, freq_cutoff_hz: float = FREQ_CUTOFF_HZ
) -> np.ndarray: ...
```

- Redefine `NUM_FREQ_BINS = num_freq_bins()` and `EXPECTED_SPECTROGRAM_SHAPE = spectrogram_shape()`
  so existing importers (API, `thesis/explain/`, tests) keep working unchanged.
- Keep `FREQ_CUTOFF_HZ = 70.0` and its docstring. Add a module-docstring sentence saying the
  cutoff can be lowered per run for ablations, and that 70 Hz is the default everything else
  assumes.
- **Floating point:** `int(30.0 / 0.9765625) + 1 = 31`, matching `freqs <= 30.0` (bin 30 is
  29.30 Hz, bin 31 is 30.27 Hz). A test must assert `num_freq_bins(c)` equals the row count
  produced by the mask for several cutoffs, so the two can never disagree.

### 2. `thesis/dataset.py`: store the cutoff on the dataset instance

- `SpectrogramDataset.__init__(..., freq_cutoff_hz: float = FREQ_CUTOFF_HZ)`, stored as
  `self.freq_cutoff_hz`.
- `convert_to_spectrograms(tensor, source_fs, augmentation=None, is_inear=False,
  freq_cutoff_hz=FREQ_CUTOFF_HZ)`, passing it to `compute_log_spectrogram`.
- `_get_item` passes `self.freq_cutoff_hz`.
- Update both docstrings. They currently say the cutoff "is not configurable per call site".

### 3. `thesis/data_preparation.py`: thread it through preparation

- Add `freq_cutoff_hz: float = FREQ_CUTOFF_HZ` to the generic helper (line ~72) and to
  `prepare_mdd_dataset`, `prepare_cane_dataset`, `prepare_sad_dataset`, `prepare_idun_dataset`.
  Pass it to every `SpectrogramDataset(...)` construction.
- Replace the three shape asserts with `spectrogram_shape(freq_cutoff_hz)`.
- The raw-EEG branch (`use_raw_eeg=True`) ignores it.

### 4. `thesis/cli.py`: new flag in `add_preprocessing_args`

It goes in the preprocessing group, so both `train` and `run` get it: a 30 Hz checkpoint must be
run with the same cutoff.

```python
preproc_group.add_argument(
    "--freq-cutoff",
    type=float,
    default=70.0,
    help="Highest spectrogram frequency kept, in Hz (default 70, the band-pass upper edge). "
    "Lower it for frequency ablations, e.g. 30 to exclude EMG-dominated gamma. Must be in "
    "[21, 70]: below ~21 Hz the spectrogram CNN has no rows left. Ignored by raw-EEG models.",
)
```

Validate the range in `train()`/`run()` with a clear `ValueError`: above 70 is meaningless after
the band-pass, and below 21 the CNN collapses.

### 5. `main.py`

- `train()`: pass `freq_cutoff_hz=args.freq_cutoff` to `train_cross_validation`. If the model is in
  `RAW_EEG_MODELS` and the cutoff isn't 70, log a warning that it is ignored.
- `train_cross_validation(..., freq_cutoff_hz: float = FREQ_CUTOFF_HZ)`: add a docstring param,
  pass it to every `prepare_*` call (`main.py:~615-760`, all branches), and replace
  `spec_shape = EXPECTED_SPECTROGRAM_SHAPE` (`main.py:794`) with
  `spec_shape = spectrogram_shape(freq_cutoff_hz)`.
- **Checkpoint:** add `"freq_cutoff_hz": freq_cutoff_hz` to the `torch.save` dict in
  `train_one_fold` (`main.py:~407`), so every checkpoint records the geometry it was trained on.
  This takes a `freq_cutoff_hz` parameter on `train_one_fold`. Old checkpoints lack the key, which
  means 70.
- `run` (`main.py:~1101`): use `spectrogram_shape(args.freq_cutoff)` and pass the cutoff to
  `preprocess_and_infer`. If the checkpoint has `freq_cutoff_hz` and it differs from the flag,
  raise. `create_model` loads with `strict=False`, so a mismatched projection layer would
  otherwise be silently dropped.

### 6. `thesis/inference.py`

Add `freq_cutoff_hz: float = FREQ_CUTOFF_HZ` to `preprocess_and_infer` (and `preprocess_file` if
it builds spectrograms), forwarded to `convert_to_spectrograms` at line 318.

### 7. Out of scope

- `api/`: keeps the 70 Hz default. Don't touch.
- `thesis/explain/` (untracked leftovers): don't touch. It keeps importing
  `EXPECTED_SPECTROGRAM_SHAPE`, which is unchanged.

### 8. `CLAUDE.md`

- Add a "Recent Major Changes" bullet for `--freq-cutoff`.
- Amend the Spectrogram Generation section: 70 Hz is the default, overridable per run, with the
  `[21, 70]` constraint.
- Mention that checkpoints now store `freq_cutoff_hz`.

## Tests: `tests/test_freq_cutoff.py`

Synthetic data only; no dataset files needed.

1. **Bin count matches the mask:** for cutoffs `[21, 30, 45.5, 70]` and source rates
   `[250, 256, 500]`, `compute_log_spectrogram(np.random.randn(10 * fs), fs, c).shape ==
   spectrogram_shape(c)`.
2. **Default unchanged:** `spectrogram_shape() == (72, 41)`, `EXPECTED_SPECTROGRAM_SHAPE == (72, 41)`,
   and `compute_log_spectrogram(x, fs)` is `np.array_equal` to the explicit-70 call.
3. **Crop consistency, the key invariant:** `compute_log_spectrogram(x, fs, 30)` is
   `np.array_equal` to `compute_log_spectrogram(x, fs, 70)[:31]`. It guarantees the 30 Hz run
   differs from baseline **only** by the removed rows.
4. **Models build and run at (31, 41):** `create_model("CNNAttnS", spec_shape=(31, 41),
   in_channels=1, ...)` and `create_model("AllTransformerV4", spec_shape=(31, 41), in_channels=8,
   ...)`. A forward pass on `(2, C, 31, 41)` returns `(2, 2)`. Also assert `(22, 41)` builds and
   `(21, 41)` raises.
5. **Validation:** the cutoff check rejects `20` and `80` and accepts `21`, `30` and `70`.
6. **Pickling:** a `SpectrogramDataset` with `freq_cutoff_hz=30` survives `pickle.dumps/loads`
   with the attribute intact. Follow the pattern in `tests/test_dataset_pickle.py`.

`make format`, `make typecheck`, `make test` must pass.

### Smoke check before GPU time

`poetry run python main.py train --test-mode --epochs 1 --freq-cutoff 30 --model CNNAttnS
--channel in-ear --dataset all --condition ec`. The log shows `freq_cutoff: 30.0` and spectrogram
shape `(1, 31, 41)`, and `fold_*_best.pth` contains `freq_cutoff_hz == 30.0`. Repeat for
`AllTransformerV4 --channel all`, expecting `(8, 31, 41)`.

## Runs

**Four runs, not two.** Torch isn't seeded, so a 30 Hz run compared against the old Table II run
would mix the cutoff effect with run-to-run noise and code drift. The Table II runs were made at
commit `454c49e-dirty`, before the picklability and optimizer commits. Rerunning the 70 Hz
baselines on the same code gives identical folds, a paired per-fold comparison, and a direct
measure of how much two same-config trainings disagree.

The 70 Hz AllTransformerV4 rerun also produces the saved checkpoints that experiment **C**
(post-hoc band shuffle) needs; the Table II run didn't keep them.

Hyperparameters are copied from the Table II runs **unchanged**:

| Dir | Model | Common args | Cutoff |
|---|---|---|---|
| `all_041_inear_ec+eo_f70` | CNNAttnS | `--channel in-ear --batch-size 64 --epochs 100 --lr 1e-4 --dropout 0.05 --weight-decay 5e-5 --patience 20` | 70 |
| `all_041_inear_ec+eo_f30` | CNNAttnS | same | 30 |
| `all_041_all_ec+eo_f70` | AllTransformerV4 | `--channel all --batch-size 64 --epochs 200 --lr 1e-4 --dropout 0.1 --weight-decay 1e-4 --patience 50` | 70 |
| `all_041_all_ec+eo_f30` | AllTransformerV4 | same | 30 |

All four also get `--dataset all --condition ec+eo --class-mode 2 --n-folds 10 --val-every 1
--focal-loss`. Sources: `experiments/binary/in-ear/cnnattns-binary_wve/results.txt` (CNNAttnS)
and `experiments/results.txt` (`atv4-lowlr_oex`, AllTransformerV4).

**Launcher:** `run-freq-cutoff.sh`, modeled on `run-all-experiments.sh` (same `ml` modules,
`EEG_DATA_DIR`, `launch <gpu> <name> args...`, `nohup`, per-run `stdout.log`). All four launch in
parallel, one per GPU.

## Analysis

`helpers-print/freq_cutoff_summary.py` reads the four run directories (`results.txt` plus
`fold_N_best.pth` → `val_metrics`) and produces:

1. **Table:** model × cutoff, with chunk acc, subject acc, sensitivity, specificity (mean ± std)
   and parameter count. The count drops at 30 Hz because the projection layer shrinks (416 → 80
   inputs). Verified on current code: CNNAttnS 79,490 → 57,986; AllTransformerV4 113,474 → 91,970.
   Report it.
2. **Paired difference (30 − 70) per fold:** mean and 95% CI, Wilcoxon signed-rank, and the
   Nadeau–Bengio corrected t-test.
3. **Per-dataset chunk accuracy** (MDD / CANE / SAD) at both cutoffs, from
   `val_metrics["per_dataset"]`. A drop confined to CANE or SAD (the anxiety cohorts) would be the
   EMG-shaped pattern.
4. **Reproducibility check:** compare the 70 Hz reruns with Table II (77.0 / 79.2 and 76.5 / 78.6),
   descriptively. A large gap means Table II carries run-to-run luck, which is worth knowing
   whatever the cutoff result is.

**Fold-identity check** before the paired test: the `Val subjects (N): [...]` log lines must match
fold by fold between the 70 and 30 Hz runs of each model.

## What gets printed in the paper

Two or three sentences in Section VII plus one table row pair per model. Things to state once:

- **30 Hz doesn't remove all EMG.** Whitham et al. (2007) show EMG contaminates scalp EEG above
  about 20 Hz, so 20–30 Hz beta still carries some. The 30 Hz cut removes the EMG-*dominated*
  range, not all EMG. A tighter cut isn't possible with the current CNN (minimum ~21 Hz) and would
  also remove beta, a band the anxiety literature cites.
- "Accuracy holds at 30 Hz" means **the result doesn't depend on gamma**, not "the model has no
  artifact sensitivity".
- The two cutoffs change the input size and parameter count. Hyperparameters were not retuned.

## Verification

- The tests above pass, especially the crop-consistency invariant and the default-unchanged test.
- The smoke runs show the right shapes and the checkpoint key.
- `make format`, `make typecheck`, `make test` pass.
- After the GPU runs: four directories, each with 10 `fold_N_best.pth` and a `results.txt`. The
  fold-identity check passes.

## References

- Whitham et al. 2007, "Scalp electrical recording during paralysis: quantitative evidence that
  EEG frequencies above 20 Hz are contaminated by EMG", *Clinical Neurophysiology* 118(8).
  https://pubmed.ncbi.nlm.nih.gov/17574912/
- Goncharova, McFarland, Vaughan & Wolpaw 2003, "EMG contamination of EEG: spectral and
  topographical characteristics", *Clinical Neurophysiology* 114(9).
  https://doi.org/10.1016/S1388-2457(03)00093-2
- Hooker, Mentch & Zhou 2021, "Unrestricted permutation forces extrapolation", *Statistics and
  Computing* 31. https://arxiv.org/abs/1905.03151
- Hooker, Erhan, Kindermans & Kim 2019, "A Benchmark for Interpretability Methods in Deep Neural
  Networks" (ROAR), NeurIPS. https://arxiv.org/abs/1806.10758
- Nadeau & Bengio 2003, "Inference for the Generalization Error", *Machine Learning* 52.
  https://doi.org/10.1023/A:1024068626366
