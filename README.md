# EEG-Based Depression Detection

> This thesis project implements a CNN-LSTM deep learning model for EEG-based depression (MDD)
and anxiety detection. The model architecture is inspired by the DepCap research paper and uses
STFT spectrograms as input features derived from preprocessed EEG signals.

## Development Setup

To create a virtual environment with all necessary dependencies, run `make venv`. To use it,
activate the Poetry environment:

```bash
# Install dependencies with Poetry
make venv

# Or manually
poetry install --with dev

# Activate the environment
poetry shell
```

For other helpful targets (such as formatting and type checking), see [Makefile](Makefile).

## Dataset

The project supports three EEG datasets that can be used independently or combined:

### MDD Dataset

Expected location at `../MDD/`. Files follow the naming pattern: `{H|MDD} S{N} {EC|EO|TASK}.edf`

**Dataset Structure:**
- **H** = Healthy control subjects
- **MDD** = Major Depressive Disorder subjects
- **Conditions**: EC (Eyes Closed), EO (Eyes Open), TASK - can train on single or combined (ec+eo)
- **Channels**: 8 channels used (Fp1, Fp2, T7, T8, C3, C4, Cz, Oz) with standardized 10-20 naming
- **Sampling Rate**: 250 Hz (SFREQ = 1000/4)
- **Segments**: 10-second chunks (2,500 samples each)

### CANE Dataset

Expected location at `../CANE/`. Files follow pattern: `{H|AX} S{N} {ec|eo}.edf`

**Dataset Structure:**
- **H** = Healthy control subjects
- **AX** = Anxiety disorder subjects
- **Conditions**: ec (eyes closed), eo (eyes open) - can train on single or combined (ec+eo)
- **Channels**: Same 8 channels as MDD dataset with standardized naming (T3→T7, T4→T8)
- **Sampling Rate**: 500 Hz
- **Preprocessing**: Optional artifact removal via `--skip-artifact-removal` flag

### AX_MALIK Dataset

Expected location at `../AX_MALIK/`. Files follow pattern: `{ec|eo}/C{N}.edf`

**Dataset Structure:**
- **All subjects are anxiety class** (no healthy controls in this dataset)
- **Conditions**: EC (eyes closed), EO (eyes open) - can train on single or combined (ec+eo)
- **Channels**: Same 8 channels as MDD dataset (Fp1, Fp2, C3, Cz, C4, T7, T8, O2)
- **Sampling Rate**: 256 Hz
- **Duration**: 120 seconds per file
- **Subjects**: 21 subjects (42 files total - one EC and one EO per subject)
- **Preprocessing**: Uses MDDDataset preprocessing pipeline via inheritance

### Multi-Dataset Training

- `--dataset mdd`: MDD only (2 classes: normal vs depressed)
- `--dataset cane`: CANE only (2 classes: normal vs anxious)
- `--dataset ax_malik`: AX_MALIK only (anxiety subjects only - no healthy controls in this dataset)
- `--dataset all`: Combined (supports 2-class and 4-class modes)

### Classification Modes

- `--class-mode 2`: Binary classification (healthy vs any-pathological) - works with all datasets
- `--class-mode 4`: Multi-class (normal/anxiety/depression/comorbid) - requires `--dataset all`

## Usage

### Train

Train the CNN-LSTM model using 10-fold cross-validation on the Eyes Closed (EC) condition:

```bash
# Standard training
poetry run python main.py train

# Test mode - loads only one file per class for rapid debugging
poetry run python main.py train --test-mode

# Multi-channel training (default: all 8 channels)
poetry run python main.py train --skip-ica \
  --channel all \
  --batch-size 32 \
  --checkpoint-dir=experiments/mdd_001_all_ec \
  --dataset mdd \
  --condition ec

# Single-channel training (Fp1 channel)
poetry run python main.py train \
  --channel Fp1 \
  --batch-size 32 \
  --checkpoint-dir=experiments/mdd_001_fp1_ec \
  --dataset mdd \
  --condition ec \
  --n-folds 10 \
  --dropout 0.5 \
  --weight-decay 1e-4 \
  --val-every 1

# 4-class classification with multi-channel input
poetry run python main.py train --skip-ica \
  --channel all \
  --class-mode 4 \
  --dataset all \
  --condition ec \
  --checkpoint-dir=experiments/all_001_all_ec_4class
```

Training creates checkpoints in the specified directory with the following structure:

```
checkpoints/
├── fold_1_best.pth          # Best model for fold 1
├── fold_1_final.pth         # Final model for fold 1
├── fold_1_checkpoint_5.pth  # Periodic checkpoint
├── ...
└── cv_results.json          # Cross-validation results summary
```

### Training Analysis

Visualize training and validation loss curves across folds:

```bash
# Generate loss curve plots for an experiment
poetry run python plot_training_curves.py <checkpoint_dir>/<log file>
```

This creates a `loss_curves/` directory with individual plots for each fold and combined overview plot. 

### Model Inference

Run inference on a single EDF file using a trained model:

```bash
# Run inference with trained model, keep same channel and ica setting as when you trained it
poetry run python main.py run <model_path> <EDF file path> --channel Fp1 --skip-ica
```

The inference script outputs:
- Per-chunk predictions (each 10-second segment)
- Subject-level aggregated prediction
- Confidence scores

## Model Architecture

**Available Model Architectures:**

1. **CNN_LSTM_DepCap** (default, single-channel):
   - Input: STFT spectrograms (129 × 41) from 10-second EEG segments
   - Conv2D layers: 64 filters (10×10), 32 filters (5×5) with MaxPooling
   - Spatial Dropout (`nn.Dropout2d`): Applied after each pooling layer
   - LSTM/GRU layer: hidden size = 100
   - Dense classifier: 64 → 32 → num_classes

2. **Smaller** (reduced model, ~50% fewer parameters):
   - Lighter architecture for faster experimentation
   - Same structure as CNN_LSTM_DepCap but with fewer filters

3. **SmallerAll** (multi-channel, uses `--channel all`):
   - Input: 8-channel spectrograms (8 × 129 × 41)
   - Conv3D layers: Process all 8 EEG channels simultaneously
   - Learns spatial correlations between electrode positions
   - Automatically selected when using `--channel all`

**Common Features:**
- Configurable dropout (default 0.5) and L2 weight decay (default 1e-4)
- Supports 2-class or 4-class classification modes
- Transfer learning support via `--pretrained-checkpoint`

## Preprocessing Pipeline

Each EDF file undergoes the following preprocessing (see `thesis/dataset.py`):

1. Load EDF file
2. Bandpass filter: 1-70 Hz (IIR)
3. Notch filter: 50 Hz (remove power line noise)
4. Channel selection: 8 channels (all) or single specific channel
5. Channel name standardization: T3→T7, T4→T8 for cross-dataset compatibility
6. Average reference
8. Optional artifact removal for CANE: (`--skip-artifact-removal` to disable)
9. Segmentation: 10-second chunks
10. STFT transformation: Convert to spectrograms
11. Normalization: Log-magnitude spectrograms

## Development

### Executing tests

```
make test
```

### Formatting and Linting

```bash
# Format code with ruff
make format

# Run type checking with mypy
make typecheck
```


### Viewing Spectrograms

You can visualize the spectrograms generated from preprocessed EEG data using the test scripts:

```bash
# View MDD/CANE dataset spectrograms
poetry run python mdd.py
poetry run python cane.py
```

Spectrograms will be saved into `spectogram_(mdd|cane).png` files.

### Cross-Validation Strategy

The project uses **subject-level stratified K-fold cross-validation** (default K=10):
- Prevents data leakage (all chunks from same subject stay together)
- Maintains class balance across folds (2-class or 4-class stratification)
- For multi-dataset training, ensures each fold contains subjects from both datasets
- Standard approach in EEG depression/anxiety classification literature
- Early stopping based on chunk-level validation accuracy
- Comprehensive metrics: accuracy, precision, recall, specificity, sensitivity

## Important Notes

- Dataset paths are hardcoded in `thesis/dataset.py` as `MDD_DIR`, `CANE_DIR`, and `AX_MALIK_DIR`
- Channel selection: Use `--channel all` (default) for 8 channels or specify single channel (Fp1, T7, etc.)
- Multi-channel mode uses SmallerAll model with Conv3D for spatial feature learning
- 4-class mode requires `--dataset all` (normal/anxiety/depression/comorbid)
- Channel naming standardized: T3→T7, T4→T8 for cross-dataset compatibility
- STFT parameters: `nperseg=256`, `noverlap=192` → output shape (129, 41)
- artifact removal are now optional for faster iteration (`--skip-artifact-removal`)
- Test mode available: `--test-mode` loads one file per class for rapid debugging
- Model expects preprocessed spectrograms, not raw EEG directly
- Checkpoint directories are auto-generated with random suffixes if they exist
- All logging uses Python's `logging` module, not print statements
- Default regularization: dropout=0.5, weight_decay=1e-4 (L2 penalty)
- See `EXPERIMENTS.md` for experiment tracking and results
