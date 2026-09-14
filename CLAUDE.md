# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a thesis project for EEG-based depression (MDD) & anxiety detection using deep learning. The project implements a CNN-LSTM model inspired by research papers on EEG depression classification.

**Key Technologies:**
- PyTorch for deep learning
- MNE-Python for EEG signal processing
- Poetry for dependency management
- Python 3.13

## Datasets

The project supports three EEG datasets that can be used independently or combined:

### 1. MDD Dataset
Located at `/home/milan/Documents/diplomka/MDD/`. Files follow the naming pattern: `{H|MDD} S{N} {EC|EO|TASK}.edf`

- **H** = Healthy control
- **MDD** = Major Depressive Disorder
- **Conditions**: EC (Eyes Closed), EO (Eyes Open), TASK - can train on single or combined (ec+eo)
- **Channels**: 8 channels available (Fp1, Fp2, T7, T8, C3, C4, Cz, Oz) - can use all, select specific channel, or use synthetic in-ear channel
- **In-ear channel**: Bipolar derivation T8 - T7 simulating IDUN-style in-ear EEG (use `--channel in-ear`)
- **Bipolar montages**: `--channel Fp2-Fp1`, `--channel C4-C3`, `--channel T8-T7` — interhemispheric bipolar derivations (minuend − subtrahend), no CAR (meaningless over two electrodes). `T8-T7` is bit-identical to `in-ear` on this dataset
- **Channel Naming**: Standardized to 10-20 system (T3→T7, T4→T8 for cross-dataset compatibility)
- **Sampling**: 256 Hz (native rate of the EDF files, verified against all 181 headers), resampled to 250 Hz before the STFT
- **Segments**: 10-second chunks (2560 samples each at the native rate)

### 2. CANE Dataset
Located at `/home/milan/Documents/diplomka/CANE-dataset/`. Files follow pattern: `{H|AX} S{N} {ec|eo}.edf`

- **H** = Healthy control
- **AX** = Anxiety disorder
- **Conditions**: ec (eyes closed), eo (eyes open) - lowercase, can train on single or combined (ec+eo)
- **Channels**: Same 8 channels as MDD (standardized naming), plus synthetic in-ear option
- **Bipolar montages**: `--channel Fp2-Fp1`, `--channel C4-C3`, `--channel T8-T7` — same derivations as MDD/SAD, but sourced from the 8-channel headset, **not** IDUN (unlike `in-ear`). For these specs the initial per-channel z-score and CAR are skipped and the raw electrode voltages are subtracted first (`minuend − subtrahend`), matching MDD/SAD; only the resulting single derived channel then goes through detrend → artifact removal → bandpass → notch
- **Sampling**: 500 Hz
- **Preprocessing**: Artifact removal optional via `--skip-artifact-removal` flag

**IMPORTANT**: When using `--channel in-ear`, CANE is automatically replaced with IDUN real in-ear data (see below). The bipolar montages (`Fp2-Fp1`, `C4-C3`, `T8-T7`) do **not** trigger this replacement — CANE is always loaded from the headset for those.

### 3. IDUN Dataset
Located at `/home/milan/Documents/diplomka/IDUN_IN_EAR/`. Files follow pattern: `{class_dir}/{subject_id}/eeg_{subject_id}{condition}.csv`

- **Real in-ear EEG recordings** from IDUN device (single-channel CSV format)
- **Classes**: normals (17), anxiety (20), depression (1), comorbid (15) - 53 usable subjects for EC condition
- **Conditions**: ec (eyes closed), eo (eyes open) - case-insensitive in filenames
- **Channel**: Single in-ear channel (no multi-channel support)
- **Sampling**: 250 Hz (already at `MODEL_FS`, so no resampling needed — the only dataset that isn't resampled)
- **Quality Metrics**: Signal quality data in `quality_*.csv` files (0=not measured, 90+=good signal)
- **Edge cases**: Subjects 1001, 1002 have non-standard filenames (skipped automatically)
- **Preprocessing**: Custom pipeline for CSV files (z-score → detrend → bandpass → notch → chunking → quality filtering)

**Usage**:
- Automatically used when `--channel in-ear` is specified with `--dataset cane` or `--dataset all`
- Provides real in-ear EEG instead of synthetic bipolar derivations (T8-T7)
- Uses `dataset_label="cane"` internally for fold compatibility, so logs may report "CANE" but IDUN data is actually used
- Quality threshold (default 0.0) rejects chunks where quality=0 (unmeasured signal)

### 4. SAD Dataset
Located at `/home/milan/Documents/diplomka/SAD/`. Files follow pattern: `{normals|anxious}/{ec|eo}/C{N}.edf`

- **Classes**: normals and anxious (2 classes)
- **Conditions**: EC (eyes closed), EO (eyes open) - can train on single or combined (ec+eo)
- **Channels**: Same 8 channels as MDD (Fp1, Fp2, C3, Cz, C4, T7, T8, O2) - standardized 10-20 nomenclature, plus synthetic in-ear option
- **Sampling**: 256 Hz
- **Duration**: 120 seconds per file
- **Inheritance**: Uses MDDDataset preprocessing pipeline via inheritance for maximum code reuse

### Multi-Dataset Training
The system supports training on:
- `--dataset mdd`: MDD only (2 classes: normal vs depressed)
- `--dataset cane`: CANE only (2 classes: normal vs anxious)
  - **With `--channel in-ear`**: Automatically uses IDUN real in-ear data instead of CANE
- `--dataset all`: Combined (supports both 2-class and 4-class modes)
  - **With `--channel in-ear`**: Uses MDD (synthetic T8-T7) + IDUN (real in-ear) + SAD (synthetic T8-T7)

### Classification Modes
- `--class-mode 2`: Binary classification (healthy vs any-pathological)
- `--class-mode 4`: Multi-class classification (normal/anxiety/depression/comorbid) - requires `--dataset all`

## Common Commands

### Environment Setup
```bash
# Install core + dev dependencies
poetry install --with dev

# Install with API dependencies (for running the inference server)
poetry install --with dev,api

# Or use Makefile shortcut
make venv
```

### Code Formatting and Type Checking

After every implementation, run both:

```bash
make format    # ruff format + ruff check --fix
make typecheck # mypy thesis/ *.py
```

These must pass (or show only pre-existing errors) before considering work complete.

### Running Training
```bash
# Main training script with cross-validation
poetry run python main.py train

# Test mode - load only one file per class for rapid debugging
poetry run python main.py train --test-mode

# Multi-channel training (default, uses all 8 channels)
poetry run python main.py train \ 
  --channel all \
  --batch-size 32 \
  --checkpoint-dir=experiments/mdd_007_all_ec \
  --dataset mdd \
  --condition ec

# Single-channel training example
poetry run python main.py train --skip-ica \
  --channel Fp1 \
  --batch-size 32 \
  --checkpoint-dir=experiments/mdd_007_fp1_ec \
  --dataset mdd \
  --condition ec \
  --n-folds 10 \
  --dropout 0.5 \
  --weight-decay 1e-4 \
  --val-every 1

# In-ear EEG training (synthetic for MDD, real IDUN for CANE)
poetry run python main.py train --skip-ica \
  --channel in-ear \
  --batch-size 32 \
  --checkpoint-dir=experiments/mdd_008_inear_ec \
  --dataset mdd \
  --condition ec

# Real in-ear EEG with IDUN dataset (automatic replacement)
poetry run python main.py train --skip-ica \
  --channel in-ear \
  --batch-size 32 \
  --checkpoint-dir=experiments/idun_001_inear_ec \
  --dataset cane \
  --condition ec

# 4-class classification (requires --dataset all)
poetry run python main.py train --skip-ica \
  --channel all \
  --class-mode 4 \
  --dataset all \
  --condition ec \
  --checkpoint-dir=experiments/all_001_all_ec

# Raw-EEG Deformer model (no spectrograms)
poetry run python main.py train --skip-ica \
  --channel all \
  --model Deformer \
  --chunk-duration 10.0 \
  --dataset mdd \
  --condition ec \
  --checkpoint-dir=experiments/mdd_020_all_ec

# Disable cosine annealing for ablation (constant LR)
poetry run python main.py train --skip-ica \
  --channel all \
  --no-cosine-lr \
  --dataset mdd \
  --condition ec \
  --checkpoint-dir=experiments/mdd_021_all_ec

# AdamW optimizer for a decoupled weight-decay sweep (default optimizer is adam)
poetry run python main.py train --skip-ica \
  --channel all \
  --optimizer adamw \
  --weight-decay 1e-2 \
  --dataset mdd \
  --condition ec \
  --checkpoint-dir=experiments/mdd_022_all_ec
```

### Transfer Learning
```bash
# Fine-tune a pretrained MDD model on CANE dataset
poetry run python main.py train \
  --channel Fp1 \
  --batch-size 32 \
  --checkpoint-dir=experiments/cane_009_fp1_ec_transfer \
  --dataset cane \
  --condition ec \
  --n-folds 6 \
  --pretrained-checkpoint experiments/mdd_006_fp1_ec/fold_1_best.pth \
  --freeze-cnn

# Transfer learning with all layers trainable (no freezing)
poetry run python main.py train \
  --channel Fp1 \
  --checkpoint-dir=experiments/cane_009_fp1_ec_finetune \
  --dataset cane \
  --condition ec \
  --pretrained-checkpoint experiments/mdd_006_fp1_ec/fold_1_best.pth
```

**Transfer Learning Options:**
- `--pretrained-checkpoint <path>`: Load weights from a trained model checkpoint (.pth file)
- `--freeze-cnn`: Freeze CNN layers (conv1, conv2) during training; only LSTM and classifier are trained

**Use Cases:**
- Pre-train on larger MDD dataset, then fine-tune on smaller CANE dataset
- Leverage learned EEG feature representations across related tasks
- Reduce training time when target dataset is small

### Experiment Organization

**IMPORTANT: Experiment Directory Naming Convention**

All experiment directories **MUST** follow this naming pattern:
```
<dataset>_<version>_<channel>_<condition>
```

**Components:**
- `<dataset>`: Dataset identifier (`mdd`, `cane`, `sad`, or `all`)
- `<version>`: Experiment version/iteration (e.g., `006`, `007`)
- `<channel>`: EEG channel used (e.g., `fp1`, `t7`, `t8`, `all` for 8-channel)
- `<condition>`: Recording condition (`ec`, `eo`, or `ec+eo`)

**Examples:**
- `experiments/mdd_006_fp1_ec` - MDD dataset, version 006, Fp1 channel, eyes closed
- `experiments/mdd_008_inear_ec` - MDD dataset, version 008, synthetic in-ear EEG, eyes closed
- `experiments/cane_006_t7_ec+eo` - CANE dataset, version 006, T7 channel, combined conditions
- `experiments/sad_001_all_ec` - SAD dataset, version 001, all 8 channels, eyes closed
- `experiments/all_007_all_ec` - All datasets, version 007, all 8 channels, eyes closed
- `experiments/all_012b_all_ec+eo` - All datasets, multi-channel, combined conditions

**Why this matters:**
- Ensures consistent organization across all experiments
- Makes it easy to identify experiment parameters from directory name
- Prevents confusion between channel names and directory names
- Facilitates automated analysis and result aggregation

**When creating new experiments:**
1. Always use this naming pattern for the `--checkpoint-dir` argument
2. Ensure the channel in the directory name matches the `--channel` argument
3. Document experiments in `EXPERIMENTS.md` following the established format

### Interactive Development
Jupyter notebooks are available for prototyping:
- `prototyping.ipynb` - Main development notebook
- `testing.ipynb` - Testing and experimentation
- `yo.ipynb` - Additional experiments

Run with: `poetry run jupyter notebook`

## Code Architecture

### Module Structure

**`thesis/`** - Main package containing core functionality

1. **`thesis/dataset.py`** - EEG data loading and preprocessing
   - `MDDDataset`: PyTorch Dataset with lazy/preload modes, LRU caching for EDF files
   - `CANEDataset`: PyTorch Dataset for CANE EDF files with optional artifact removal
   - `IDUNDataset`: PyTorch Dataset for IDUN real in-ear CSV files with quality filtering
   - `SADDataset`: PyTorch Dataset for SAD EDF files (inherits from MDDDataset)
   - `create_cross_validation_splits()`: Subject-level stratified K-fold CV
   - `get_preprocessed_chunks()`: Legacy function for single-file processing
   - Preprocessing pipelines:
     - EDF files (MDD/SAD): bandpass filter (1-70 Hz) → notch filter (50 Hz) → average reference (CAR) → linear detrend → 10s chunking
     - EDF files (CANE): CAR → detrend → optional artifact removal → bandpass filter (1-70 Hz) → notch filter (50 Hz) → 10s chunking
     - CSV files (IDUN): z-score → detrend → bandpass filter (1-70 Hz) → notch filter (50 Hz) → 10s chunking → quality-based rejection
   - CANE artifact removal can be skipped with `--skip-artifact-removal` flag

2. **`thesis/model.py`** - Neural network architectures
   - `CNN_LSTM_DepCap`: Main model implementing the paper's architecture (2D convolutions for single-channel)
     - Conv2D layers (64 filters 10x10, 32 filters 5x5) with MaxPool
     - Spatial Dropout (nn.Dropout2d) after each pooling layer for better CNN regularization
     - LSTM/GRU layer (hidden=100) treating spectrograms as time sequences
     - Dense classifier (64 → 32 → num_classes) with standard dropout
   - `CNNLSTM`: Reduced model (~50% fewer parameters) for faster experimentation
   - `CNNLSTMAll`: Multi-channel model with per-channel weight-shared CNN + cross-channel self-attention + LSTM
     - TransformerEncoder computes soft channel weights before temporal merging
   - `CNNAttn`: CNNLSTM variant replacing LSTM with temporal self-attention (single-channel)
   - `CNNAttnAll`: CNNLSTMAll variant replacing LSTM with temporal self-attention (8-channel)
   - `CNNCatLSTM`: Multi-channel model concatenating all channel features at each time step
     - Feeds concatenated (8×features) vectors to LSTM directly (no soft gating)
   - `Deformer`: EEG-Deformer (J-BHI 2024) — raw EEG transformer with shallow CNN + hierarchical transformer layers
     - Accepts raw EEG `(batch, channels, time)` instead of spectrograms
     - Implemented in `thesis/deformer.py` (CBCR License 1.0)
   - `DeformerS`: Smaller Deformer variant (~1/3 params): depth 4→3, heads 16→4, num_kernel 64→48
   - `LGGNet` / `LGGNetS`: Local-Global-Graph Network (TNNLS 2023) — graph neural network over EEG channels
     - Uses predefined adjacency graphs (global, hemisphere, frontal) to aggregate spatial features
     - Implemented in `thesis/lggnet.py` (CBCR License 1.0)
     - `LGGNetS`: smaller variant (num_T 64→32, ~585K params)
   - `TSception` / `TSceptionS`: Temporal-Spatial CNN (IJCNN 2020) — multi-scale temporal + asymmetric spatial filters
     - Implemented in `thesis/tsception.py` (CBCR License 1.0, TSceptionWrapper class only)
     - `TSceptionS`: smaller variant (hidden 128→32, ~264K params; paper's cross-dataset recommendation)
   - **`RAW_EEG_MODELS`** frozenset: `{"Deformer", "DeformerS", "LGGNet", "LGGNetS", "TSception", "TSceptionS"}` — models that consume raw time-series instead of spectrograms

3. **`thesis/model_factory.py`** - Model creation factory
   - `create_model()`: Instantiates models from `MODEL_REGISTRY`, optionally loads pretrained weights, and freezes layers
     - Accepts optional `raw_eeg_config` for raw-EEG models; applies model-appropriate defaults if `None`
   - `DeformerConfig`: Dataclass of Deformer hyperparameters (num_time=2500, temporal_kernel=25, num_kernel=64, depth=4, heads=16, mlp_dim=16, dim_head=16)
   - `DeformerSConfig`: Reduced-parameter variant defaults (num_kernel=48, depth=3, heads=4)
   - `LGGNetConfig`: Dataclass of LGGNet hyperparameters (num_T=64, out_graph=32, pool=16, pool_step_rate=0.25, graph_type="hem")
   - `LGGNetSConfig`: Smaller variant defaults (num_T=32; all other fields identical to LGGNetConfig)
   - `TSceptionConfig`: Dataclass of TSception hyperparameters (num_T=15, num_S=15, hidden=128)
   - `TSceptionSConfig`: Smaller variant defaults (hidden=32; paper's cross-dataset recommendation)
   - `_RAW_EEG_DEFAULT_CONFIGS`: Dict mapping model names to their default config classes
   - Used by both CLI (`main.py`) and API (`api/model_manager.py`)

4. **`thesis/inference.py`** - Shared inference pipeline
   - `preprocess_file()`: Preprocesses EDF/CSV files into EEG chunks
     - Auto-detects EDF channel config via `_detect_edf_channel_config()` (tries MDD then SAD layouts; falls back to identity mapping with a warning)
   - `run_chunk_inference()`: Runs per-chunk model inference
   - `aggregate_predictions()`: Majority-vote aggregation
   - `preprocess_and_infer()`: High-level convenience combining all steps
   - `ChunkResult` / `InferenceResult`: Dataclasses for structured results
   - Used by both CLI (`main.py run`) and API (`api/app.py /predict`)

5. **`thesis/version.py`** - Version management
   - Reads version from `pyproject.toml` (single source of truth)
   - Used by API (`api/app.py`, `api/__init__.py`)

### Training Scripts

**`main.py`** - Primary training script
- Contains `SpectrogramDataset` for converting EEG to spectrograms
- Uses STFT (Short-Time Fourier Transform) for time-frequency representation
- Implements training/evaluation loops with comprehensive metrics (accuracy, precision, recall, specificity, sensitivity)
- Supports training on single condition (EC/EO) or combined (EC+EO)
- Early stopping based on chunk-level accuracy (not combined metric)
- Configurable hyperparameters: dropout (default 0.5), weight decay/L2 (default 1e-4)
- Current setup: 10-fold CV (configurable with --n-folds)
- **Transfer learning support**: Load pretrained weights and optionally freeze CNN layers
- **Model registry**: `MODEL_REGISTRY` dict in `thesis/model.py` maps model names to (class, default_rnn_hidden) tuples
- **Model factory**: `create_model()` in `thesis/model_factory.py` handles model instantiation, weight loading, and layer freezing
- **Optimizer selection**: `--optimizer` chooses `adam` (default, backward-compatible) or `adamw`; instantiated via `create_optimizer()` in `thesis/model_factory.py`, which preserves the `filter(lambda p: p.requires_grad, ...)` behaviour needed for `--freeze-cnn`/`--freeze-lstm`. `adamw` decouples `--weight-decay` from the gradient update (unlike `adam`, where Adam's adaptive scaling largely cancels coupled L2 out)
- **Cosine annealing LR**: Enabled by default for all models (`T_max=num_epochs`, `eta_min=1e-6`); disable with `--no-cosine-lr`
- **LR scheduler support in `train_one_fold()`**: Scheduler stepped once per epoch; scheduler class name and chosen optimizer logged at fold start
- **Parametric chunk duration**: `--chunk-duration` (default 10.0 s) for raw-EEG models; spectrogram models always use 10 s
- **Train accuracy at best val epoch**: Tracked per fold as `train_acc_at_best` and included in CV results
- **Raw-EEG config building**: `train()` computes `num_time = chunk_duration × 250`, loads model defaults from `_RAW_EEG_DEFAULT_CONFIGS`, applies CLI overrides via `dataclasses_replace()`
- **LGGNet CLI overrides**: `--lggnet-num-t`, `--lggnet-out-graph`, `--lggnet-pool`, `--lggnet-pool-step-rate`, `--lggnet-graph-type` (global/hemisphere/frontal)
- **TSception CLI overrides**: `--tsception-num-t`, `--tsception-num-s`, `--tsception-hidden`

**`train_cv_example.py`** - Clean reference implementation
- Shows proper 10-fold cross-validation pattern
- Example `SimpleEEGModel` demonstrating raw EEG input (not spectrograms)
- Useful template for new model architectures

## Data Loading Modes

The `MDDDataset` supports two modes:

**Lazy Loading (default, `preload=False`):**
- Fast initialization (~1s): Only reads file headers
- Files preprocessed on-demand during training
- LRU cache keeps N recently used files (default: 10)
- Best for limited RAM

**Preloading (`preload=True`):**
- Slow initialization (~5-10 min): Preprocesses everything upfront
- All data in memory for instant access
- Best when RAM is abundant and training speed is critical

## Model Input/Output

**Raw EEG Input** (all shapes are post-resampling to `MODEL_FS` = 250 Hz; native chunks are 2560 samples for MDD/SAD and 5000 for CANE):
- Single channel: `(batch_size, 1, 2500)` - 1 channel × 2500 samples
- In-ear channel: `(batch_size, 1, 2500)` - in-ear EEG
  - MDD/SAD: Synthetic bipolar derivation T8-T7
  - IDUN (replaces CANE): Real in-ear recordings from IDUN device
- Bipolar montage (`Fp2-Fp1`, `C4-C3`, `T8-T7`): Same shape `(batch_size, 1, 2500)` as in-ear; CANE always from the headset (never IDUN); 50% sign flip augmentation applied like in-ear
- Multi-channel: `(batch_size, 8, 2500)` - 8 channels × 2500 samples
- Output from datasets as `batch['eeg']`
- **Deformer/DeformerS** consume raw EEG directly `(batch, channels, num_time)` where `num_time = chunk_duration × 250`; they bypass `SpectrogramDataset`

**Spectrogram Input:**
- Single channel (CNN_LSTM_DepCap, CNNLSTM): `(batch_size, 1, H, W)` where H=frequency bins, W=time frames
- In-ear channel: Same as single channel `(batch_size, 1, H, W)` — **no** sign flip augmentation for spectrogram models
  - IDUN or synthetic in-ear depending on dataset
- Bipolar montage (`Fp2-Fp1`, `C4-C3`, `T8-T7`): Same as single channel `(batch_size, 1, H, W)`; CANE always from the headset
- Multi-channel (CNNLSTMAll, CNNAttnAll, CNNCatLSTM): `(batch_size, 8, H, W)` - 8 spectrograms, one per channel
- Current shape: `(72, 41)` — `EXPECTED_SPECTROGRAM_SHAPE` in `thesis/stft.py`
- Created via STFT in `SpectrogramDataset` or preprocessing
- **All STFT parameters live in `thesis/stft.py`** — the single source of truth shared by training and inference. Do not pass per-dataset `nperseg`/`noverlap` anywhere.

**Labels (2-class mode):**
- MDD dataset: 0 = Healthy, 1 = Depressed
- CANE dataset: 0 = Healthy, 1 = Anxious
- IDUN dataset: 0 = Healthy (normals), 1 = Any pathological (anxiety/depression/comorbid)
- SAD dataset: 0 = Healthy, 1 = Anxious
- All datasets: 0 = Healthy, 1 = Any pathological

**Labels (4-class mode, requires --dataset all):**
- 0 = Healthy (normal)
- 1 = Anxiety only
- 2 = Depression only
- 3 = Comorbid (both anxiety and depression)

## Key Implementation Details

**Cross-Validation Architecture:**
- **Subject-level splits**: All chunks from same subject stay together to prevent data leakage
- **Stratified folding**: Maintains class balance (normal/depressed/anxious) across folds
- **Balanced multi-dataset folding**: When using `--dataset all`, each fold contains subjects from all datasets (MDD, CANE, SAD) (not just mixed randomly). This is achieved via `create_balanced_folds()` which stratifies each dataset independently then merges corresponding folds.
- **Standard**: 10-fold CV as per EEG depression literature

**Design Principles:**
1. **Modularity**: Dataset preparation separated into `prepare_mdd_dataset()`, `prepare_cane_dataset()`, `prepare_idun_dataset()`, and `prepare_sad_dataset()` functions
2. **Extensibility**: Adding new datasets requires implementing a prepare function following the same pattern
3. **Data leakage prevention**: Never split chunks from the same subject across train/val
4. **Reproducibility**: Fixed random seed (42) for deterministic fold creation
5. **Dataset independence**: When combining datasets, maintain separate preprocessing pipelines
6. **Channel standardization**: Unified 8-channel ordering (Fp1, Fp2, T7, T8, C3, C4, Cz, Oz) with automatic T3→T7, T4→T8 mapping for cross-dataset compatibility
7. **IDUN/CANE substitution**: When `--channel in-ear` is used, IDUN automatically replaces CANE to provide real in-ear recordings instead of synthetic derivations. Uses `dataset_label="cane"` internally for fold compatibility.

**Preprocessing Pipeline (in `thesis/dataset.py`):**

Applied to each MDD/SAD .edf file (`load_and_preprocess_edf_file`):

1. Load EDF
2. Bandpass filter 1-70 Hz (IIR)
3. Notch filter 50 Hz (remove power line noise)
4. Drop reference electrode (A2-A1 if present)
5. **Average reference** (CAR — subtract mean across all 8 EEG channels at each time point)
6. **Linear detrend** per channel (removes slow DC drifts)
7. Select channels:
   - 8 channels (when `--channel all`)
   - Specific single channel (e.g., `--channel Fp1`)
   - Synthetic in-ear: pick T7+T8, detrend (no CAR — meaningless with 2 electrodes), compute T8-T7
8. Chunk into 10-second segments
9. Per-chunk z-score normalization
10. For in-ear **raw-EEG models only**: Apply 50% sign flip augmentation per chunk during training (in `FlattenedRawEEGDataset.__getitem__`). Spectrogram-based in-ear models do **not** receive sign flip — `SpectrogramDataset` accepts `is_inear` but does not apply the flip.

**Note:** Preprocessing is now more flexible - artifact removal can be skipped for faster iteration. The model can learn to handle artifacts directly from the data.

**Spectrogram Generation (`thesis/stft.py`):**

Most parameters are fixed constants in `thesis/stft.py`, imported by both the training path
(`SpectrogramDataset`) and the inference path (`preprocess_and_infer`). The per-dataset input is
`source_fs`; the frequency cutoff is overridable per run via `--freq-cutoff` (see below).

1. **Resample to `MODEL_FS` (250 Hz)** — CANE (500 Hz) and MDD/SAD (256 Hz) are resampled via
   `scipy.signal.resample`; only IDUN is already at 250 Hz and passes through untouched
2. `scipy.signal.stft()` with `nperseg=256`, `noverlap=192`, Hamming window
3. Log-magnitude: `log1p(abs(Zxx))` for better dynamic range
4. **Crop to `FREQ_CUTOFF_HZ` (70 Hz default)** — drops bins above the cutoff, which carry only
   band-pass roll-off (at 70 Hz) or, at a lower ablation cutoff, frequencies deliberately excluded

Result at the default 70 Hz: `(72, 41)`, with a bin width of 0.977 Hz spanning 0–69.34 Hz **for
every dataset**. The cutoff is overridable per run via `--freq-cutoff` (both `train` and `run`,
range `[21, 70]` — below ~21 Hz the spectrogram CNN has no rows left to build its layers from);
`thesis.stft.spectrogram_shape(freq_cutoff_hz)` gives the resulting shape (e.g. 30 Hz → `(31,
41)`). Every checkpoint's `fold_N_best.pth` records the `freq_cutoff_hz` it was trained with
(missing the key means 70, the pre-ablation default); `main.py run` raises if `--freq-cutoff`
doesn't match a loaded checkpoint's recorded value. This isn't preventing a silent failure —
`create_model` loads weights with `strict=False`, but PyTorch raises a `RuntimeError` on any
shape mismatch regardless of `strict` (`strict=False` only suppresses missing/unexpected
*keys*, not mismatched-shape ones) — the check exists to turn that eventual opaque
"size mismatch for proj.weight: ..." error into an actionable one before we get there.
Raw-EEG models (`RAW_EEG_MODELS`) never build a spectrogram, so they don't record, check, or
print `freq_cutoff_hz` at all — the flag is silently meaningless for them except for a
warning logged when a non-default value is explicitly passed.

**Why resample before the STFT?** `nperseg` fixes the *number* of frequency bins, not their
width — that is `true_fs / nperseg`. Previously each dataset used its own `noverlap` chosen only
to make the output *shape* match (129, 41), so CANE had 1.953 Hz bins spanning 0–250 Hz while
MDD had 0.977 Hz bins spanning 0–125 Hz. Alpha-band energy landed at roughly half the row index
for CANE, and the position of the band-pass dead zone was a dataset fingerprint the model could
exploit as a shortcut in combined (`--dataset all`) training and in MDD→CANE transfer learning.

## Configuration

**Ruff (Code Formatter):**
- Line length: 100 characters
- Auto-fix enabled for imports, errors, and formatting
- Rules: I (import sorting), E (errors), F (pyflakes)

**Git:**
- Main branch: `main`
- Recent commits focus on dataset improvements and model forward pass

## Code documentation structure

Sphinx/reStructuredText (reST) style documentation:
- Parameters: :param type name: description
- Returns: :return: description and :rtype: type
- Full sentences with proper capitalization


## Architecture Philosophy

**Why Two-Level Metrics (Chunk + Subject)?**
- **Chunk-level accuracy**: Measures per-segment classification performance (more samples, lower variance)
- **Subject-level accuracy**: Aggregates chunks via majority voting (clinical relevance, what matters for diagnosis)
- **Combined metric**: `chunk_acc × subject_acc` was used historically but now early stopping uses chunk accuracy
- **Early stopping**: Uses chunk-level accuracy as it provides more stable gradient signal during training
- **Rationale**: A model that's 100% accurate on chunks but only 50% on subjects is overfitting to chunk-level noise

**Why Balanced Fold Stratification?**
When training on combined datasets, naive concatenation + splitting would create folds with only MDD or only CANE subjects. This causes:
- Unrepresentative validation sets
- Dataset-specific overfitting
- Unreliable cross-validation metrics

Solution: Stratify each dataset independently, then merge corresponding folds. Every fold becomes a mini-representation of the full combined dataset.

**Why Subject-Level Splits?**
EEG data has temporal dependencies. Splitting at chunk level would leak information:
- Training chunks at time T could correlate with validation chunks at T+10s from same subject
- Model learns subject-specific patterns instead of generalizable depression markers
- Artificially inflated performance metrics

**Why Separate Dataset Classes?**
- Different preprocessing requirements:
  - CANE uses artifact detection with interpolation/clipping
  - MDD uses simpler pipeline
  - IDUN has custom CSV pipeline with quality-based rejection
- Different file formats (EDF vs CSV for IDUN)
- Different channel names/orderings in raw files
- Different native sampling rates (MDD: 256 Hz, CANE: 500 Hz, IDUN: 250 Hz, SAD: 256 Hz) — all resampled to `MODEL_FS` = 250 Hz before reaching a model
- Different channel naming in raw files (requires mapping to canonical names)
- Allows independent evolution of preprocessing pipelines
- Channel standardization: EDF datasets use unified 8-channel canonical ordering

## Code Evolution Notes

**Recent Major Changes:**
- **Frequency-cutoff ablation** (Sep 2026): `--freq-cutoff` (default 70, range `[21, 70]`) makes the spectrogram's upper frequency bound overridable per run, for testing whether results depend on the EMG-dominated gamma range (e.g. `--freq-cutoff 30`)
  - Motivation: scalp EMG from jaw/forehead/neck muscles dominates EEG above ~20 Hz (Whitham et al. 2007); retraining with gamma cropped out — rather than post-hoc occlusion — tests whether accuracy depends on that band without ever showing the model an unnatural input (avoids the ROAR/Hooker extrapolation critique of permutation methods)
  - `thesis/stft.py`: `num_freq_bins(freq_cutoff_hz)` and `spectrogram_shape(freq_cutoff_hz)` compute geometry for any cutoff; `NUM_FREQ_BINS`/`EXPECTED_SPECTROGRAM_SHAPE` are just these evaluated at the 70 Hz default. `compute_log_spectrogram()` takes `freq_cutoff_hz` as its third parameter
  - `SpectrogramDataset` stores `freq_cutoff_hz` **on the instance**, not as a module constant — `forkserver` DataLoader workers re-import modules, so a module-level override would never reach them; the instance attribute survives pickling via the existing `_PicklableLRUCacheMixin`
  - Threaded through `thesis/data_preparation.py`'s `prepare_*_dataset()` functions and `main.py`'s `train_cross_validation()`/`train_one_fold()` like `chunk_duration`; ignored (with a logged warning if non-default) for `RAW_EEG_MODELS`, which never build spectrograms — raw-EEG checkpoints never record a `freq_cutoff_hz` key, `main.py run` skips the cutoff check entirely for them, and `train()`'s config dump omits `freq_cutoff` from both the console log and `results.txt` for them
  - Every `fold_N_best.pth` checkpoint now records the `freq_cutoff_hz` it was trained with (a missing key means 70, the pre-ablation default); `main.py run` raises if `--freq-cutoff` doesn't match a loaded checkpoint's recorded value. `create_model` loads weights with `strict=False`, but PyTorch raises a `RuntimeError` on any shape mismatch regardless of `strict` (`strict=False` only suppresses missing/unexpected *keys*, not mismatched-shape ones), so this check isn't preventing a silent failure — it replaces an opaque low-level "size mismatch for proj.weight: ..." error with an actionable one
  - `--freq-cutoff` is validated to `[21, 70]`: below ~21 Hz (`spectrogram_shape` height 21) the CNN's conv/pool stack would need a negative dimension and fails to build; above 70 is meaningless after the 1–70 Hz band-pass
  - Tests in `tests/test_freq_cutoff.py` cover bin-count/mask agreement, the 70 Hz default being unchanged, crop-consistency (a lower cutoff is a strict row-prefix of the 70 Hz spectrogram), model construction at reduced/minimum/below-minimum shapes, CLI range validation, and pickling
- **Bipolar surrogate ablation channels** (Sep 2026): `--channel Fp2-Fp1`, `--channel C4-C3`, `--channel T8-T7` add interhemispheric bipolar derivations alongside `in-ear`
  - Motivation: the paper's in-ear result uses T8-T7 as a surrogate for real in-ear EEG on MDD/SAD; this ablation tests whether that specific electrode pair matters or whether any bipolar channel reaches similar accuracy
  - `thesis/dataset.py`: `BIPOLAR_CHANNELS` dict + `bipolar_pair()` resolve a `--channel` spec to its (minuend, subtrahend) electrodes; `"in-ear"` resolves to `("T8", "T7")`. Explicit whitelist, not string-splitting on `-` (`in-ear` also contains a hyphen)
  - `load_and_preprocess_edf_file()` (MDD/SAD) and `CANEDataset.load_and_preprocess_cane_raw_file()` (CANE) both branch on `bipolar_pair(channel) is not None` instead of the old `channel == "in-ear"` hardcode
  - `T8-T7` on MDD/SAD is bit-identical to `in-ear` (same electrodes, same "no CAR" path) — verified in `tests/test_bipolar_channels.py`
  - Unlike `in-ear`, the new bipolar specs do **not** trigger the CANE→IDUN swap in `main.py`: CANE is always loaded from the 8-channel headset for `Fp2-Fp1`/`C4-C3`/`T8-T7`, with a `logger.info` line making this explicit
  - `is_inear` in `thesis/data_preparation.py` (drives raw-EEG sign-flip augmentation) now checks `bipolar_pair(channel) is not None`, so all three montages get the same augmentation as `in-ear`
  - CANE bipolar specs skip the initial per-channel z-score and CAR and subtract raw electrode columns before the single-channel pipeline, so common-mode content cancels exactly (per-channel z-scoring would scale each electrode by its own σ). Covered by synthetic-CSV tests in `tests/test_bipolar_channels.py::TestCANEBipolarCancellation`
- **Selectable optimizer** (Sep 2026): `--optimizer` chooses between `adam` (default, unchanged behaviour) and `adamw` for the upcoming regularization sweep
  - Motivation: plain `Adam` couples `weight_decay` into the gradient, where Adam's per-parameter adaptive scaling largely cancels it out, making `--weight-decay` a weaker knob than its value suggests; `AdamW` decouples it
  - `create_optimizer()` in `thesis/model_factory.py` instantiates the chosen optimizer, preserving the `filter(lambda p: p.requires_grad, ...)` behaviour needed for `--freeze-cnn`/`--freeze-lstm`
  - Threaded through `train()` → `train_cross_validation(optimizer_name=...)` like other hyperparameters (not read from `args` deep inside the CV loop)
  - Logged at fold start in `train_one_fold()`, next to the LR scheduler class name
- **Unified spectrogram frequency axis + 70 Hz crop** (Jul 2026): New `thesis/stft.py` is the single source of truth for all STFT parameters, shared by training and inference
  - All datasets are now resampled to `MODEL_FS` (250 Hz) **before** the STFT, so a spectrogram row means the same frequency everywhere. Previously CANE (500 Hz) had 1.953 Hz bins vs MDD's 0.977 Hz, putting alpha at half the row index and leaving a dataset-identifying dead band
  - **`MDDDataset.FS` corrected from 250 to 256 Hz** — the EDF headers of all 181 MDD recordings say 256. The old value made a "10-second" chunk 2500/256 = 9.77 s and, after the change above, would have wrongly skipped MDD's resample. MDD chunks are now 2560 samples natively
  - `FlattenedRawEEGDataset` now resamples instead of truncating when a chunk is longer than `target_samples`; the old truncation handed 256 Hz data to models expecting 250 Hz. **Raw-EEG checkpoints (Deformer/LGGNet/TSception) are invalidated too**
  - `preprocess_file()` now chunks EDFs at the file's own rate; previously it used a fixed 2500 samples, which produced 40 STFT frames instead of 41 for 256 Hz files — silently, since the LSTM accepts variable sequence lengths
  - Removed `CANEDataset.STFT_NPERSEG` / `STFT_NOVERLAP` and `inference.get_stft_params_for_fs()` / `_STFT_PARAMS_BY_FS` — per-dataset STFT parameters no longer exist
  - `SpectrogramDataset(...)` and `convert_to_spectrograms(...)` now take `source_fs` instead of `fs`/`nperseg`/`noverlap`/`window`
  - Bins above 70 Hz are dropped (the band-pass upper edge): input shape `(129, 41)` → `(72, 41)`, 20–46% fewer model parameters
  - `preprocess_and_infer()` lost its unused `stft_params` argument
  - **All spectrogram checkpoints trained before this change are invalid** and must be retrained, including `models/*.pth` used by the API
- **IDUN real in-ear EEG integration** (Feb 2026): Added `IDUNDataset` for real in-ear recordings
  - 53 subjects (17 normals, 20 anxiety, 1 depression, 15 comorbid) at 250 Hz
  - CSV format with quality metrics for chunk rejection
  - Automatically replaces CANE when `--channel in-ear` is used
  - Uses `dataset_label="cane"` internally for fold compatibility
  - Custom preprocessing pipeline: z-score → detrend → bandpass → notch → quality filtering
- **In-ear EEG channel** (Feb 2026): Added `--channel in-ear` option
  - MDD/SAD: Synthetic bipolar derivation T8 - T7
  - CANE: Real IDUN in-ear recordings (automatic replacement)
  - 50% sign flip augmentation per chunk to handle polarity ambiguity — **raw-EEG models only** (`FlattenedRawEEGDataset`); spectrogram models do not receive sign flip
  - Based on research: "Estimating cognitive workload using a commercial in-ear EEG headset"
- **Multi-channel EEG support** (Jan 2026): Can use all 8 channels (`--channel all`) or single channel
  - Channel standardization: T3→T7, T4→T8 mapping for cross-dataset compatibility
  - Canonical 8-channel ordering: Fp1, Fp2, T7, T8, C3, C4, Cz, Oz
- **4-class classification** (Jan 2026): `--class-mode 4` for normal/anxiety/depression/comorbid classification
  - Requires `--dataset all` for access to all classes
  - Label remapping logic handles dataset-specific class availability
  - Binary mode (`--class-mode 2`) remains available for comparison
- **Simplified preprocessing** (Jan 2026): artifact removal now optional
  - `--skip-artifact-removal` for CANE dataset
  - Model learns to handle artifacts directly from data
- **Test mode** (Feb 2026): `--test-mode` flag for rapid debugging with single file per class
- **Enhanced metrics** (Dec 2025): Added sensitivity/specificity/recall to cv_results.txt
- **Transfer learning**: Load pretrained weights via `--pretrained-checkpoint` and freeze layers with `--freeze-cnn`/`--freeze-lstm`
- **Model registry**: `MODEL_REGISTRY` dict centralizes model class lookups; `create_model()` factory handles instantiation
- **Multi-condition support**: Can now train on combined EC+EO conditions using `--condition ec+eo`
- **Focal Loss** (Mar 2026): `--focal-loss` flag replaces CrossEntropyLoss with FocalLoss from `thesis/loss.py`
  - Class weights computed per fold are passed as alpha (combines count + difficulty imbalance correction)
  - Configurable exponent via `--focal-gamma` (default 2.0); gamma=0 reduces to weighted cross-entropy
  - Complements WeightedRandomSampler: sampler handles count balance, focal modulation handles hard examples
- **Spatial Dropout**: Added `nn.Dropout2d` after CNN pooling layers for better feature map regularization
- **L2 Regularization**: Added weight decay parameter (default 1e-4) for L2 penalty on weights
- **Training visualization**: Added `plot_training_curves.py` for analyzing fold performance
- **CNNLSTMAll / CNNAttnAll** (Apr 2026): Multi-channel models with per-channel weight-shared CNN + cross-channel TransformerEncoder soft gating; CNNAttnAll replaces LSTM with temporal self-attention
- **CNNCatLSTM** (Apr 2026): Redesigned from soft gating to full channel concatenation before LSTM; simpler and often stronger
- **EEG-Deformer / DeformerS** (Apr 2026): Raw-EEG transformer models from J-BHI 2024 paper
  - `Deformer`: full model (~1.78M params); `DeformerS`: reduced variant (~588k params)
  - Implemented in `thesis/deformer.py`; bypass spectrogram pipeline entirely
  - Configured via `DeformerConfig` / `DeformerSConfig` dataclasses in `thesis/model_factory.py`
  - CLI overrides via `--deformer-depth`, `--deformer-heads`, `--deformer-num-kernel`, `--deformer-mlp-dim`, `--deformer-dim-head`, `--deformer-temporal-kernel`
- **LGGNet / LGGNetS + TSception / TSceptionS** (Apr 2026): Additional raw-EEG models from literature
  - LGGNet (TNNLS 2023): graph-based spatial aggregation; `thesis/lggnet.py` (CBCR License)
  - TSception (IJCNN 2020): multi-scale temporal + spatial CNNs; `thesis/tsception.py` (CBCR License)
  - Both bypass spectrogram pipeline; added to `RAW_EEG_MODELS` frozenset
  - Configured via `LGGNetConfig` / `LGGNetSConfig` / `TSceptionConfig` / `TSceptionSConfig`; CLI overrides via `--lggnet-*` / `--tsception-*` flags
- **Parametric chunk duration** (Apr 2026): `--chunk-duration` (default 10.0 s) controls raw-EEG window size for Deformer/DeformerS; spectrogram models always use 10 s
- **Cosine annealing LR extended to all models** (Apr 2026): Previously only for RAW_EEG_MODELS; now default for all architectures
  - Disable with `--no-cosine-lr` for constant-LR ablations
  - `T_max=num_epochs`, `eta_min=1e-6`
- **Train accuracy at best val epoch** (Apr 2026): Now tracked per fold and written to CV results
- **EDF channel auto-detection in inference** (Apr 2026): `_detect_edf_channel_config()` tries known layouts (MDD, SAD) before falling back with a warning; removes hardcoded assumptions
- **API: `classification_task` renamed** (Apr 2026): `"2class"` → `"binary"` throughout API and model manager
- **API: rate limit raised** (Apr 2026): 10 → 50 requests per 10 seconds
- **API: model keys updated** (Apr 2026): `model_in_ear_2class` → `model_in_ear_binary`; `model_8channel_2class` → `model_8channel_binary`; default models upgraded (in-ear: `CNN_LSTM_DepCap` → `CNNLSTM`; 8-channel: `CNNAttnAll`)

**Deprecated Patterns:**
- ~~Direct use of `MDDDataset` for training~~ → Use `prepare_mdd_dataset()`
- ~~Simple `np.array_split()` for combined datasets~~ → Use `create_balanced_folds()`
- ~~Concatenating subject lists without dataset labels~~ → Use tuples with dataset prefix

## Important Notes

- The MDD dataset path is hardcoded in `thesis/dataset.py` as `MDD_DIR`
- STFT dimensions don't match the original paper's target (254, 342); the project uses its own `(72, 41)` geometry defined in `thesis/stft.py`
- Some experimental code in `main.py` is commented out (train loops, old dataset usage)
- Model expects preprocessed spectrograms, not raw EEG directly (unless using the raw EEG example model)
- Use logging for prints, not print statements.
- Add mypy typing to any newly generated code.
- **Parameter logging**: New CLI parameters are logged automatically — `train()` builds a `cfg` dict from `vars(args)` plus derived values (`augment_data`, `device`, `checkpoint_dir`, `log_file`, `git_commit`) and iterates it for both the console logger and `results.txt`. No manual additions needed when adding new args.

## Training Utilities

**`plot_training_curves.py`** - Visualization script for analyzing training results
- Generates loss curves from checkpoint training history
- Creates per-fold plots and combined overview
- Marks best model epochs for easy identification
- Saves plots to `loss_curves/` directory in checkpoint folder
- Usage: `poetry run python plot_training_curves.py <checkpoint_dir>/<log file>`

## Plan Mode

When executing in plan mode, save the generated plan to `docs/plans/` using the filename format:

```
docs/plans/YYYY-MM-DD-kebab-case-title.md
```

Use today's date and a short title derived from the task. Existing plans in that directory follow this convention — match their style.

**Always save the final plan to `docs/plans/` at the end of plan mode.** The `.claude/plans/` file created during planning is ephemeral — the permanent record belongs in `docs/plans/`.

## Vendored Third-Party Code (Licenses)

This project vendors code from external repositories under CBCR License 1.0 (non-commercial). Already vendored: `thesis/deformer.py`, `thesis/lggnet.py`, `thesis/tsception.py`. When adding any new vendored file:

1. **Add the full license header** to the vendored file (see `thesis/deformer.py` for the exact format).
2. **Add an entry to `LICENSE`** under the `THIRD-PARTY EXCEPTIONS` section, including:
   - Filename, license name, copyright holder
   - Source URL and paper DOI
   - Any modifications made
3. **Keep the project's MIT License separate** — vendored files are governed by their own license, not MIT.

The project's main license (MIT) is in `LICENSE`. All CBCR-licensed files are listed there in the THIRD-PARTY EXCEPTIONS section.

## Quick Reference Files
- `docs/plans/` - Saved plan mode outputs, dated and titled
- `EXPERIMENTS.md` - Log of experiment configurations and results
- `.claude/commands/review-experiments.md` - `/review-experiments` slash command: reads EXPERIMENTS.md + experiment dirs, generates actionable insights
- `thesis/dataset.py` - Dataset implementations (MDDDataset, CANEDataset, IDUNDataset, SADDataset, SpectrogramDataset)
- `thesis/data_preparation.py` - Dataset preparation functions (prepare_mdd_dataset, prepare_cane_dataset, prepare_idun_dataset, prepare_sad_dataset)
- `thesis/inference.py` - Shared inference pipeline (preprocess, infer, aggregate) for CLI and API
- `thesis/model_factory.py` - `create_model()` factory used by CLI and API; also `create_optimizer()` (adam/adamw)
- `thesis/version.py` - Single source of truth for version string (reads from pyproject.toml)
- `thesis/labels.py` - Centralized label definitions, display names, and mappings
- `thesis/stft.py` - Canonical STFT/spectrogram configuration shared by training and inference (resampling, parameters, 70 Hz crop, `EXPECTED_SPECTROGRAM_SHAPE`)
- `thesis/deformer.py` - EEG-Deformer and DeformerS raw-EEG transformer implementations
- `thesis/loss.py` - Custom loss functions (`FocalLoss` with alpha/gamma)
- `thesis/cli.py` - CLI argument parser with all training options
- `main.py` - Training orchestration, cross-validation logic, transfer learning support, and CANE/IDUN swap logic
- `plot_training_curves.py` - Training curve visualization utility

## API Configuration
- **API dependencies**: Install with `poetry install --with api`
- **Model loading strategy**: Set `MODEL_LOADING=on_demand` env var to defer model loading until first request (default: `startup` loads all 4 models immediately)
- **Version**: API reads version from `pyproject.toml` via `thesis/version.py`
- **`classification_task` values**: `"binary"` (was `"2class"`) and `"4class"` — update any client code that used the old name
- **Rate limit**: 50 requests per 10 seconds (raised from 10)
- **Config keys**: `model_in_ear_binary`, `model_8channel_binary` (renamed from `*_2class` equivalents)

## Important Implementation Notes
- **IDUN/CANE replacement**: When using `--channel in-ear` with `--dataset cane` or `--dataset all`, CANE is automatically replaced with IDUN real in-ear data
- **Fold labeling**: IDUN uses `dataset_label="cane"` internally for fold compatibility, so validation logs may report "CANE" but IDUN data is actually used
- **Quality threshold**: IDUN's default quality threshold (0.0) rejects only unmeasured chunks (quality=0); some subjects may have all-zero quality values
- **Edge cases**: IDUN subjects 1001 and 1002 have non-standard filenames and are automatically skipped during discovery
