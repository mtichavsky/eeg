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
- **Channels**: 8 channels available (Fp1, Fp2, T7, T8, C3, C4, Cz, Oz) - can use all or select specific channel
- **Channel Naming**: Standardized to 10-20 system (T3→T7, T4→T8 for cross-dataset compatibility)
- **Sampling**: 250 Hz (SFREQ = 1000/4)
- **Segments**: 10-second chunks (2500 samples each)

### 2. CANE Dataset
Located at `/home/milan/Documents/diplomka/CANE-dataset/`. Files follow pattern: `{H|AX} S{N} {ec|eo}.edf`

- **H** = Healthy control
- **AX** = Anxiety disorder
- **Conditions**: ec (eyes closed), eo (eyes open) - lowercase, can train on single or combined (ec+eo)
- **Channels**: Same 8 channels as MDD (standardized naming)
- **Sampling**: 500 Hz
- **Preprocessing**: Artifact removal optional via `--skip-artifact-removal` flag

### 3. AX_MALIK Dataset
Located at `/home/milan/Documents/diplomka/AX_MALIK/`. Files follow pattern: `{ec|eo}/C{N}.edf`

- **All subjects are anxiety class** (no healthy controls in this dataset)
- **Conditions**: EC (eyes closed), EO (eyes open) - can train on single or combined (ec+eo)
- **Channels**: Same 8 channels as MDD (Fp1, Fp2, C3, Cz, C4, T7, T8, O2) - standardized 10-20 nomenclature
- **Sampling**: 256 Hz
- **Duration**: 120 seconds per file
- **Subjects**: 21 subjects (42 files total - one EC and one EO per subject)
- **Inheritance**: Uses MDDDataset preprocessing pipeline via inheritance for maximum code reuse

### Multi-Dataset Training
The system supports training on:
- `--dataset mdd`: MDD only (2 classes: normal vs depressed)
- `--dataset cane`: CANE only (2 classes: normal vs anxious)
- `--dataset all`: Combined (supports both 2-class and 4-class modes)

### Classification Modes
- `--class-mode 2`: Binary classification (healthy vs any-pathological)
- `--class-mode 4`: Multi-class classification (normal/anxiety/depression/comorbid) - requires `--dataset all`

For detailed dataset usage, see DATASET_USAGE.md and context.md.

## Common Commands

### Environment Setup
```bash
# Install dependencies with Poetry
poetry install --with dev

# Or use Makefile shortcut
make venv
```

### Code Formatting

```bash
make format
```

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

# 4-class classification (requires --dataset all)
poetry run python main.py train --skip-ica \
  --channel all \
  --class-mode 4 \
  --dataset all \
  --condition ec \
  --checkpoint-dir=experiments/all_001_all_ec
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
- `<dataset>`: Dataset identifier (`mdd`, `cane`, `ax_malik`, or `all`)
- `<version>`: Experiment version/iteration (e.g., `006`, `007`)
- `<channel>`: EEG channel used (e.g., `fp1`, `t7`, `t8`, `all` for 8-channel)
- `<condition>`: Recording condition (`ec`, `eo`, or `ec+eo`)

**Examples:**
- `experiments/mdd_006_fp1_ec` - MDD dataset, version 006, Fp1 channel, eyes closed
- `experiments/cane_006_t7_ec+eo` - CANE dataset, version 006, T7 channel, combined conditions
- `experiments/ax_malik_001_all_ec` - AX_MALIK dataset, version 001, all 8 channels, eyes closed
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
   - `MDDDataset`: PyTorch Dataset with lazy/preload modes, LRU caching
   - `create_cross_validation_splits()`: Subject-level stratified K-fold CV
   - `get_preprocessed_chunks()`: Legacy function for single-file processing
   - Preprocessing pipeline: bandpass filter (1-70 Hz) → notch filter (50 Hz) → optional artifact removal → 10s chunking
   - CANE artifact removal can be skipped with `--skip-artifact-removal` flag

2. **`thesis/model.py`** - Neural network architectures
   - `CNN_LSTM_DepCap`: Main model implementing the paper's architecture (2D convolutions for single-channel)
     - Conv2D layers (64 filters 10x10, 32 filters 5x5) with MaxPool
     - Spatial Dropout (nn.Dropout2d) after each pooling layer for better CNN regularization
     - LSTM/GRU layer (hidden=100) treating spectrograms as time sequences
     - Dense classifier (64 → 32 → num_classes) with standard dropout
   - `Smaller`: Reduced model (~50% fewer parameters) for faster experimentation
   - `SmallerAll`: Multi-channel variant using Conv3d to process all 8 EEG channels simultaneously
     - Conv3D kernel spans all channels for spatial feature learning
     - Enables learning cross-channel correlations

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
- **Model registry**: `MODEL_REGISTRY` dict maps model names to (class, default_rnn_hidden) tuples
- **Model factory**: `create_model()` handles model instantiation, weight loading, and layer freezing

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

**Raw EEG Input:**
- Single channel: `(batch_size, 1, 2500)` - 1 channel × 2500 samples
- Multi-channel: `(batch_size, 8, 2500)` - 8 channels × 2500 samples
- Output from datasets as `batch['eeg']`

**Spectrogram Input:**
- Single channel (CNN_LSTM_DepCap, Smaller): `(batch_size, 1, H, W)` where H=frequency bins, W=time frames
- Multi-channel (SmallerAll): `(batch_size, 8, H, W)` - 8 spectrograms, one per channel
- Current default: `(129, 41)` with nperseg=256, noverlap=192
- Paper target: `(254, 342)` (may require parameter tuning)
- Created via STFT in `SpectrogramDataset` or preprocessing

**Labels (2-class mode):**
- MDD dataset: 0 = Healthy, 1 = Depressed
- CANE dataset: 0 = Healthy, 1 = Anxious
- AX_MALIK dataset: All subjects are labeled as 1 (Anxious) - no healthy controls
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
- **Balanced multi-dataset folding**: When using `--dataset all`, each fold contains subjects from all datasets (MDD, CANE, AX_MALIK) (not just mixed randomly). This is achieved via `create_balanced_folds()` which stratifies each dataset independently then merges corresponding folds.
- **Standard**: 10-fold CV as per EEG depression literature

**Design Principles:**
1. **Modularity**: Dataset preparation separated into `prepare_mdd_dataset()` and `prepare_cane_dataset()` functions
2. **Extensibility**: Adding new datasets requires implementing a prepare function following the same pattern
3. **Data leakage prevention**: Never split chunks from the same subject across train/val
4. **Reproducibility**: Fixed random seed (42) for deterministic fold creation
5. **Dataset independence**: When combining datasets, maintain separate preprocessing pipelines
6. **Channel standardization**: Unified 8-channel ordering (Fp1, Fp2, T7, T8, C3, C4, Cz, Oz) with automatic T3→T7, T4→T8 mapping for cross-dataset compatibility

**Preprocessing Pipeline (in `thesis/dataset.py`):**

Applied to each .edf file:

1. Load EDF
2. Bandpass filter 1-70 Hz (IIR)
3. Notch filter 50 Hz (remove power line noise)
4. Select channels (8 channels or specific channel)
5. Average reference
6. Optional: Artifact interpolation/clipping for CANE (--skip-artifact-removal to disable)
7. Chunk into 10-second segments
8. Convert to PyTorch tensors

**Note:** Preprocessing is now more flexible - artifact removal can be skipped for faster iteration. The model can learn to handle artifacts directly from the data.

**Spectrogram Generation:**
- Uses `scipy.signal.stft()` with Hamming window
- Log-magnitude: `log1p(abs(Zxx))` for better dynamic range
- Parameters configurable: `nperseg`, `noverlap`, `fs`

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
- Different preprocessing requirements (CANE uses artifact detection with interpolation/clipping, MDD uses simpler pipeline)
- Different STFT parameters (though standardized to same output shape)
- Different channel names/orderings in raw files
- Different sampling rates (MDD: 250 Hz, CANE: 500 Hz)
- Different channel naming in raw files (requires mapping to canonical names)
- Allows independent evolution of preprocessing pipelines
- Channel standardization: Both datasets now use unified 8-channel canonical ordering

## Code Evolution Notes

**Recent Major Changes:**
- **Multi-channel EEG support** (Jan 2026): Can use all 8 channels (`--channel all`) or single channel
  - New `SmallerAll` model with Conv3d for multi-channel spatial feature learning
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
- **Spatial Dropout**: Added `nn.Dropout2d` after CNN pooling layers for better feature map regularization
- **L2 Regularization**: Added weight decay parameter (default 1e-4) for L2 penalty on weights
- **Training visualization**: Added `plot_training_curves.py` for analyzing fold performance

**Deprecated Patterns:**
- ~~Direct use of `MDDDataset` for training~~ → Use `prepare_mdd_dataset()`
- ~~Simple `np.array_split()` for combined datasets~~ → Use `create_balanced_folds()`
- ~~Concatenating subject lists without dataset labels~~ → Use tuples with dataset prefix

## Important Notes

- The MDD dataset path is hardcoded in `thesis/dataset.py` as `MDD_DIR`
- STFT dimensions don't currently match the paper's target (254, 342) - adjust `nperseg`/`noverlap` if needed
- Some experimental code in `main.py` is commented out (train loops, old dataset usage)
- Model expects preprocessed spectrograms, not raw EEG directly (unless using the raw EEG example model)
- Use logging for prints, not print statements.
- Add mypy typing to any newly generated code.

## Training Utilities

**`plot_training_curves.py`** - Visualization script for analyzing training results
- Generates loss curves from checkpoint training history
- Creates per-fold plots and combined overview
- Marks best model epochs for easy identification
- Saves plots to `loss_curves/` directory in checkpoint folder
- Usage: `poetry run python plot_training_curves.py <checkpoint_dir>/<log file>`

## Quick Reference Files
- `DATASET_USAGE.md` - Dataset-specific preprocessing and loading details
- `EXPERIMENTS.md` - Log of experiment configurations and results
- `thesis/dataset.py` - Dataset implementations (MDDDataset, CANEDataset, SpectrogramDataset)
- `thesis/cli.py` - CLI argument parser with all training options
- `main.py` - Training orchestration, cross-validation logic, and transfer learning support
- `plot_training_curves.py` - Training curve visualization utility
