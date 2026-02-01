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
- **Channels**: 8 channels used (Fp1, Fp2, T7, T8, C3, C4, Cz, Oz) with standardized 10-20 naming, plus synthetic in-ear option
- **Sampling Rate**: 250 Hz (SFREQ = 1000/4)
- **Segments**: 10-second chunks (2,500 samples each)

### CANE Dataset

Expected location at `../CANE/`. Files follow pattern: `{H|AX} S{N} {ec|eo}.edf`

**Dataset Structure:**
- **H** = Healthy control subjects
- **AX** = Anxiety disorder subjects
- **Conditions**: ec (eyes closed), eo (eyes open) - can train on single or combined (ec+eo)
- **Channels**: Same 8 channels as MDD dataset with standardized naming (T3→T7, T4→T8), plus synthetic in-ear option
- **Sampling Rate**: 500 Hz
- **Preprocessing**: Optional artifact removal via `--skip-artifact-removal` flag

**Note:** When using `--channel in-ear`, CANE is automatically replaced with the IDUN dataset (see below) for real in-ear EEG recordings.

### IDUN Dataset

Expected location at `../IDUN_IN_EAR/`. Files follow pattern: `{class_dir}/{subject_id}/eeg_{subject_id}{condition}.csv`

**Dataset Structure:**
- **Real in-ear EEG recordings** from IDUN device (single-channel CSV format)
- **Classes**: normals (17), anxiety (20), depression (1), comorbid (15) - 53 usable subjects
- **Conditions**: ec (eyes closed), eo (eyes open) - case-insensitive
- **Channel**: Single in-ear channel (no multi-channel support)
- **Sampling Rate**: 250 Hz (verified from timestamps)
- **Quality Metrics**: Signal quality data available in `quality_*.csv` files (0=not measured, 90+=good)
- **Preprocessing**: z-score → detrend → bandpass 1-70 Hz → notch 50 Hz → 10s chunking → quality-based rejection

**Usage:**
- IDUN is **automatically used** when `--channel in-ear` is specified with `--dataset cane` or `--dataset all`
- Provides real in-ear EEG instead of synthetic bipolar derivations (T8-T7)
- Occupies the "CANE" slot in fold creation, so logs may report "CANE" but IDUN data is actually used
- Quality threshold (default 0.0) rejects chunks with unmeasured signal quality

### AX_MALIK Dataset

Expected location at `../AX_MALIK/`. Files follow pattern: `{ec|eo}/C{N}.edf`

**Dataset Structure:**
- **All subjects are anxiety class** (no healthy controls in this dataset)
- **Conditions**: EC (eyes closed), EO (eyes open) - can train on single or combined (ec+eo)
- **Channels**: Same 8 channels as MDD dataset (Fp1, Fp2, C3, Cz, C4, T7, T8, O2), plus synthetic in-ear option
- **Sampling Rate**: 256 Hz
- **Duration**: 120 seconds per file
- **Subjects**: 21 subjects (42 files total - one EC and one EO per subject)
- **Preprocessing**: Uses MDDDataset preprocessing pipeline via inheritance

### Multi-Dataset Training

- `--dataset mdd`: MDD only (2 classes: normal vs depressed)
- `--dataset cane`: CANE only (2 classes: normal vs anxious)
  - **With `--channel in-ear`**: Automatically uses IDUN real in-ear data instead
- `--dataset ax_malik`: AX_MALIK only (anxiety subjects only - no healthy controls in this dataset)
- `--dataset all`: Combined (supports 2-class and 4-class modes)
  - **With `--channel in-ear`**: Uses MDD (synthetic T8-T7) + IDUN (real in-ear) + AX_MALIK (synthetic T8-T7)

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

# In-ear EEG training
# - MDD/AX_MALIK: synthetic bipolar derivation T8-T7
# - CANE: real IDUN in-ear recordings (automatic replacement)
poetry run python main.py train \
  --channel in-ear \
  --batch-size 32 \
  --checkpoint-dir=experiments/mdd_008_inear_ec \
  --dataset mdd \
  --condition ec

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

**Channel Options:**
- **Single channel** (e.g., `--channel Fp1`): Use a specific EEG electrode
- **Multi-channel** (`--channel all`): Use all 8 electrodes for spatial feature learning
- **In-ear EEG** (`--channel in-ear`): In-ear EEG recordings
  - **MDD/AX_MALIK**: Synthetic bipolar derivation T8 - T7
  - **CANE**: Real IDUN in-ear recordings (automatic replacement)
  - **All datasets**: MDD (synthetic) + IDUN (real) + AX_MALIK (synthetic)
  - Based on research: "Estimating cognitive workload using a commercial in-ear EEG headset"
  - Applies 50% sign flip augmentation per chunk to handle polarity ambiguity

**Common Features:**
- Configurable dropout (default 0.5) and L2 weight decay (default 1e-4)
- Supports 2-class or 4-class classification modes
- Transfer learning support via `--pretrained-checkpoint`

## Preprocessing Pipeline

### MDD/CANE/AX_MALIK Preprocessing
Each EDF file undergoes the following preprocessing (see `thesis/dataset.py`):

1. Load EDF file
2. Bandpass filter: 1-70 Hz (IIR)
3. Notch filter: 50 Hz (remove power line noise)
4. Channel selection:
   - 8 channels (when `--channel all`)
   - Specific single channel (e.g., `--channel Fp1`)
   - Synthetic in-ear: Load T7 and T8, compute bipolar derivation T8 - T7 (when `--channel in-ear`)
5. Channel name standardization: T3→T7, T4→T8 for cross-dataset compatibility
6. Average reference
7. Optional artifact removal for CANE: (`--skip-artifact-removal` to disable)
8. Segmentation: 10-second chunks
9. For in-ear: Apply 50% sign flip augmentation per chunk during training
10. STFT transformation: Convert to spectrograms
11. Normalization: Log-magnitude spectrograms

### IDUN Preprocessing
IDUN CSV files undergo a separate preprocessing pipeline (see `IDUNDataset.load_and_preprocess_idun_file()`):

1. Load CSV file (timestamp, ch1 columns)
2. Z-score normalize raw values
3. Detrend (linear)
4. Bandpass filter: 1-70 Hz (IIR)
5. Notch filter: 50 Hz (remove power line noise)
6. Segmentation: 10-second chunks (2500 samples at 250 Hz)
7. Quality-based chunk rejection (threshold=0.0 by default, rejects unmeasured chunks)
8. Per-chunk z-score normalization
9. Apply 50% sign flip augmentation per chunk during training
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

- Dataset paths are hardcoded in `thesis/dataset.py` as `MDD_DIR`, `CANE_DIR`, `AX_MALIK_DIR`, and `IDUN_DIR`
- **IDUN replacement**: When using `--channel in-ear` with `--dataset cane` or `--dataset all`, CANE is automatically replaced with IDUN real in-ear data. Logs may report "CANE" due to internal fold labeling, but IDUN data is actually used.
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

## Container Deployment (API)

The inference API can be deployed as a container image using Podman (or Docker). The image is
built on Fedora 43 and includes only the dependencies needed for serving predictions.

### Prerequisites

- **Podman** (or Docker) installed on the host
- **nvidia-container-toolkit** for GPU inference (optional — CPU works too)
- **Trained model checkpoints** (`.pth` files from cross-validation experiments)

### Model Files

The API serves 4 model variants. Place checkpoint files in a `models/` directory following this
naming convention:

| File | Electrode Setup | Classification | Architecture |
|------|-----------------|----------------|--------------|
| `model_single_2class.pth` | in-ear          | Binary (healthy vs pathological) | CNN_LSTM_DepCap |
| `model_single_4class.pth` | in-ear          | 4-class (normal/anxiety/depression/comorbid) | CNN_LSTM_DepCap |
| `model_8channel_2class.pth` | All 8 channels  | Binary | SmallerAll |
| `model_8channel_4class.pth` | All 8 channels  | 4-class | SmallerAll |

Copy the best fold checkpoint from your experiment directory:

```bash
mkdir -p models/
cp experiments/mdd_007_fp1_ec/fold_1_best.pth    models/model_single_2class.pth
cp experiments/all_012_fp1_ec/fold_1_best.pth     models/model_single_4class.pth
cp experiments/mdd_007_all_ec/fold_1_best.pth     models/model_8channel_2class.pth
cp experiments/all_012_all_ec/fold_1_best.pth     models/model_8channel_4class.pth
```

> **Note:** Not all 4 models are required. Missing models are reported as unavailable in the
> `/health` endpoint (status becomes `"degraded"`) but the API still serves requests for
> loaded models.

### Build the Image

```bash
podman build -t eeg-api -f api.Containerfile .
```

### Run with GPU

Requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
configured for Podman via the CDI (Container Device Interface) method:

```bash
podman run -d \
    --name eeg-api \
    --device nvidia.com/gpu=all \
    -p 8000:8000 \
    -v ./models:/app/models:ro,Z \
    eeg-api
```

### Run on CPU Only

```bash
podman run -d \
    --name eeg-api \
    -p 8000:8000 \
    -v ./models:/app/models:ro,Z \
    -e DEVICE=cpu \
    eeg-api
```

### Environment Variables

All configuration can be overridden via environment variables (`-e KEY=value`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/app/models` | Path to model checkpoint directory |
| `DEVICE` | `auto` | PyTorch device: `auto`, `cuda`, or `cpu` |
| `MODEL_LOADING` | `startup` | `startup` (load all on start) or `on_demand` (lazy) |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `JSON_PRETTY_PRINT` | `false` | Set `true` for colored human-readable logs; default is JSON output |
| `MAX_FILE_SIZE_MB` | `100` | Maximum upload file size in MB |
| `RATE_LIMIT_TIMES` | `10` | Number of requests allowed per rate-limit window |
| `RATE_LIMIT_SECONDS` | `10` | Rate-limit window duration in seconds |

### Verify the Deployment

```bash
# Health check
curl -s http://localhost:8000/health | python3 -m json.tool

# Run a prediction
curl -X POST http://localhost:8000/predict \
    -F "eeg_recording=@recording.edf" \
    -F "electrode_setup=single" \
    -F "classification_task=2class" \
    -F "sampling_rate=250" \
    -F "request_id=$(uuidgen)" \
    -F "user_id=$(uuidgen)"
```

### HTTPS / TLS

The app itself speaks plain HTTP. To serve HTTPS, put a reverse proxy (e.g. Nginx, Caddy) in
front that handles TLS termination and forwards plain HTTP to the container on port 8000.

If you do this, the rate limiter will see the proxy's IP instead of the real client IP. Fix it
by adding `ProxyHeadersMiddleware` in `api/app.py` and configuring the proxy to set
`X-Forwarded-For` / `X-Forwarded-Proto` headers:

```python
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts="127.0.0.1")
```

### Baking Models into the Image

Instead of mounting models at runtime, you can embed them during build. Place the `.pth` files
in `models/` before building:

```bash
# Copy models, then build — they will be included in the image
cp experiments/...  models/model_single_2class.pth
podman build -t eeg-api -f api.Containerfile .
```

To use baked-in models, simply omit the `-v` volume mount when running.
