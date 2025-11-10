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

The project uses the MDD EEG dataset located at `../MDD/`.
Files follow the naming pattern: `{H|MDD} S{N} {EC|EO|TASK}.edf`

**Dataset Structure:**
- **H** = Healthy control subjects
- **MDD** = Major Depressive Disorder subjects
- **Conditions**: EC (Eyes Closed), EO (Eyes Open), TASK
- **Channels**: 20 channels available, using only some of them 
- **Sampling Rate**: 250 Hz (SFREQ = 1000/4)
- **Segments**: 10-second chunks (2,500 samples each)

**Dataset Statistics (EC condition):**
- Total: 1,621 chunks from 54 files
- Healthy: 28 subjects
- MDD: 26 subjects

## Usage

### Basic Training

Train the CNN-LSTM model using 10-fold cross-validation on the Eyes Closed (EC) condition:

```bash
# Standard training with ICA artifact removal
poetry run python main.py train

# Faster training without ICA (for debugging/development)
poetry run python main.py train --skip-ica

# Single-channel training (Fp1 channel, automatically skips ICA)
poetry run python main.py train --channel Fp1

# Custom checkpoint directory and batch size
poetry run python main.py train --skip-ica --channel Fp1 --batch-size 4 --checkpoint-dir="checkpoints_fp1_001"
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

## Model Inference

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
- LSTM/GRU layer: hidden size = 100, treating spectrograms as time sequences
- Dropout when training
- Dense classifier: 64 → 32 → 2 classes (Healthy/MDD)
- Output: Binary classification (0=Healthy, 1=MDD)

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

### Viewing Spectrograms

You can visualize the spectrograms generated from preprocessed EEG data using the test scripts:

```bash
# View MDD dataset spectrograms
poetry run python mdd.py
# View CANE dataset spectrograms
poetry run python cane.py
```

Spectrograms will be saved into `spectogram_(mdd|cane).png` files.

### Formatting and Linting

```bash
# Format code with ruff
make format

# Run type checking with mypy
make typecheck
```

### Cross-Validation Strategy

The project uses **subject-level stratified 10-fold cross-validation**:
- Prevents data leakage (all chunks from same subject stay together)
- Maintains H/MDD balance across folds
- Standard approach in EEG depression classification literature

## Important Notes

- The MDD dataset path is hardcoded in `thesis/dataset.py` as `MDD_DIR`
- Single-channel mode automatically skips ICA (ICA requires multiple channels)
- STFT parameters: `nperseg=256`, `noverlap=192` → output shape (129, 41)
- Model expects preprocessed spectrograms, not raw EEG directly
- Checkpoint directories are auto-generated with random suffixes if they exist
- All logging uses Python's `logging` module, not print statements