# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a thesis project for EEG-based depression (MDD) & anxiety detection using deep learning. The project implements a CNN-LSTM model inspired by research papers on EEG depression classification.

**Key Technologies:**
- PyTorch for deep learning
- MNE-Python for EEG signal processing
- Poetry for dependency management
- Python 3.13

## Dataset

The project uses the MDD EEG dataset located at `/home/milan/Documents/diplomka/MDD/`. Files follow the naming pattern: `{H|MDD} S{N} {EC|EO|TASK}.edf`

- **H** = Healthy control
- **MDD** = Major Depressive Disorder
- **Conditions**: EC (Eyes Closed), EO (Eyes Open), TASK
- **Channels**: 6 channels used (Fp1, Fp2, C3, C4, O2, Cz)
- **Sampling**: 250 Hz (SFREQ = 1000/4)
- **Segments**: 10-second chunks (2500 samples each)

For detailed dataset usage, see DATASET_USAGE.md.

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

# Faster, debug run, skipping the ICA
poetry run python main.py train --skip-ica
```

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
   - Preprocessing pipeline: bandpass filter (1-70 Hz) → notch filter (50 Hz) → ICA artifact removal → 10s chunking

2. **`thesis/model.py`** - Neural network architectures
   - `CNN_LSTM_DepCap`: Main model implementing the paper's architecture
     - Conv2D layers (64 filters 10x10, 32 filters 5x5) with MaxPool
     - LSTM/GRU layer (hidden=100) treating spectrograms as time sequences
     - Dense classifier (64 → 32 → 2 classes)

### Training Scripts

**`main.py`** - Primary training script
- Contains `SpectrogramDataset` for converting EEG to spectrograms
- Uses STFT (Short-Time Fourier Transform) for time-frequency representation
- Implements training/evaluation loops with metrics (accuracy, precision, recall, specificity)
- Current setup: 10-fold CV on EC condition

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
- Shape: `(batch_size, 6, 2500)` - 6 channels × 2500 samples
- Output from `MDDDataset` as `batch['eeg']`

**Spectrogram Input (for CNN_LSTM_DepCap):**
- Shape: `(batch_size, 1, H, W)` where H=frequency bins, W=time frames
- Current default: `(129, 41)` with nperseg=256, noverlap=192
- Paper target: `(254, 342)` (may require parameter tuning)
- Created via STFT in `SpectrogramDataset` or preprocessing

**Labels:**
- 0 = Healthy (H)
- 1 = MDD

## Key Implementation Details

**Cross-Validation:**
- Subject-level splits prevent data leakage (all chunks from same subject stay together)
- Stratified to maintain H/MDD balance across folds
- Standard: 10-fold CV as per EEG depression literature

**Preprocessing Pipeline (in `thesis/dataset.py`):**
```python
# Applied to each .edf file:
1. Load EDF
2. Bandpass filter 1-70 Hz (IIR)
3. Notch filter 50 Hz (remove power line noise)
4. Select 6 channels
5. Average reference
6. ICA with ICLabel for artifact removal (remove non-brain components)
7. Chunk into 10-second segments
8. Convert to PyTorch tensors
```

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


## Important Notes

- The MDD dataset path is hardcoded in `thesis/dataset.py` as `MDD_DIR`
- STFT dimensions don't currently match the paper's target (254, 342) - adjust `nperseg`/`noverlap` if needed
- Some experimental code in `main.py` is commented out (train loops, old dataset usage)
- Model expects preprocessed spectrograms, not raw EEG directly (unless using the raw EEG example model)
- Use logging for prints, not print statements.
- Add mypy typing to any newly generated code.
