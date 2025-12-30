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

The project supports two EEG datasets that can be used independently or combined:

### MDD Dataset

Expected location at `../MDD/`. Files follow the naming pattern: `{H|MDD} S{N} {EC|EO|TASK}.edf`

**Dataset Structure:**
- **H** = Healthy control subjects
- **MDD** = Major Depressive Disorder subjects
- **Conditions**: EC (Eyes Closed), EO (Eyes Open), TASK - can train on single or combined (ec+eo)
- **Channels**: 20 channels available, using only some of them 
- **Sampling Rate**: 250 Hz (SFREQ = 1000/4)
- **Segments**: 10-second chunks (2,500 samples each)

### CANE Dataset

Expected location at `../CANE/`. Files follow pattern: `{H|AX} S{N} {ec|eo}.edf`

**Dataset Structure:**
- **H** = Healthy control subjects
- **AX** = Anxiety disorder subjects
- **Conditions**: ec (eyes closed), eo (eyes open) - can train on single or combined (ec+eo)
- **Channels**: Similar 6 channels as MDD dataset
- **Sampling Rate**: 500 Hz
- **Preprocessing**: Uses artifact detection instead of ICA

### Multi-Dataset Training

- `--dataset mdd`: MDD only (2 classes: normal vs depressed)
- `--dataset cane`: CANE only (2 classes: normal vs anxious)
- `--dataset both`: Combined (3 classes: normal vs depressed vs anxious)

## Usage

### Train

Train the CNN-LSTM model using 10-fold cross-validation on the Eyes Closed (EC) condition:

```bash
# Standard training with ICA artifact removal
poetry run python main.py train

# Faster training without ICA (for debugging/development)
poetry run python main.py train --skip-ica

# Single-channel training (Fp1 channel, automatically skips ICA)
poetry run python main.py train --channel Fp1

# Full example with all major configuration options
poetry run python main.py train --skip-ica \
  --channel Fp1 \
  --batch-size 32 \
  --checkpoint-dir=experiments/my_experiment \
  --dataset mdd \
  --condition ec \
  --n-folds 10 \
  --dropout 0.5 \
  --weight-decay 1e-4 \
  --val-every 1
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

**CNN-LSTM-DepCap Model:**
- Input: STFT spectrograms (129 × 41) from 10-second EEG segments
- Conv2D layers: 64 filters (10×10), 32 filters (5×5) with MaxPooling
- Spatial Dropout (`nn.Dropout2d`): Applied after each pooling layer for CNN regularization
- LSTM/GRU layer: hidden size = 100, treating spectrograms as time sequences
- Standard Dropout: Applied in classifier layers
- Dense classifier: 64 → 32 → num_classes
- Output: Classification (2-class: Healthy/MDD or Healthy/Anxious, 3-class: Healthy/MDD/Anxious)
- Regularization: Configurable dropout (default 0.5) and L2 weight decay (default 1e-4)

## Preprocessing Pipeline

Each EDF file undergoes the following preprocessing (see `thesis/dataset.py`):

1. Load EDF file
2. Bandpass filter: 1-70 Hz (IIR)
3. Notch filter: 50 Hz (remove power line noise)
4. Channel selection: 6 channels or single channel
5. Average reference
6. ICA with ICLabel: Artifact removal (optional, can be skipped)
7. Segmentation: 10-second chunks
8. STFT transformation: Convert to spectrograms
9. Normalization: Log-magnitude spectrograms

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
- Maintains class balance (normal/depressed/anxious) across folds
- For multi-dataset training, ensures each fold contains subjects from both datasets
- Standard approach in EEG depression/anxiety classification literature
- Early stopping based on chunk-level validation accuracy

## Important Notes

- Dataset paths are hardcoded in `thesis/dataset.py` as `MDD_DIR` and `CANE_DIR`
- Single-channel mode automatically skips ICA (ICA requires multiple channels)
- STFT parameters: `nperseg=256`, `noverlap=192` → output shape (129, 41)
- Model expects preprocessed spectrograms, not raw EEG directly
- Checkpoint directories are auto-generated with random suffixes if they exist
- All logging uses Python's `logging` module, not print statements
- Default regularization: dropout=0.5, weight_decay=1e-4 (L2 penalty)
- See `EXPERIMENTS.md` for experiment tracking and results
