# EEG-Based Depression Detection

> This thesis project implements a CNN-LSTM deep learning model for EEG-based depression (MDD)
> and anxiety detection. The model architecture is inspired by the DepCap research paper and uses
> STFT spectrograms as input features derived from preprocessed EEG signals.

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

For other helpful targets (such as formatting and type checking), see [Makefile](Makefile), e.g:

```bash
make test  # Execute unit tests
make format  # Format code with ruff
make typecheck  # Run type checking with mypy
```

## Datasets

The project supports three EEG datasets that can be used independently or combined. For training,
they have to be available at `../MDD/`, `../CANE/`, `../IDUN_IN_EAR/` and `../SAD/` locations.
The parent dir can be overwritten by setting `EEG_DATA_DIR` env var. For more info,
see [docs/DATASETS.md](docs/DATASETS.md)
and [docs/MODEL_CARD.md](docs/MODEL_CARD.md) files.

Details on how are these files preprocessed are available at [docs/PREPROCESSING.md](docs/PREPROCESSING.md).

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

More info about model is present in [docs/MODEL_CARD.md](docs/MODEL_CARD.md). See `--help` for all available options.

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
- Focal loss via `--focal-loss` (with `--focal-gamma`, default 2.0)

## Development

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
- Focal loss: `--focal-loss` replaces CrossEntropyLoss with FocalLoss; class weights passed as alpha so count and difficulty imbalance are both addressed. Use with WeightedRandomSampler (already enabled by default).
- See `EXPERIMENTS.md` for experiment tracking and results

## Container Deployment (API)

The inference API can be deployed as a container image using Podman (or Docker). The image is
built on Fedora 41 and includes only the dependencies needed for serving predictions.

### Prerequisites

- **Podman** (or Docker) installed on the host
- **nvidia-container-toolkit** for GPU inference (optional — CPU works too)
- **Trained model checkpoints** (`.pth` files from cross-validation experiments)

### Model Files

The API serves 4 model variants. Place checkpoint files in a `models/` directory following this
naming convention:

| File                        | Electrode Setup | Classification                               | Architecture    |
|-----------------------------|-----------------|----------------------------------------------|-----------------|
| `model_inear_2class.pth`    | in-ear          | Binary (healthy vs pathological)             | CNN_LSTM_DepCap |
| `model_inear_4class.pth`    | in-ear          | 4-class (normal/anxiety/depression/comorbid) | CNN_LSTM_DepCap |
| `model_8channel_2class.pth` | All 8 channels  | Binary                                       | SmallerAll      |
| `model_8channel_4class.pth` | All 8 channels  | 4-class                                      | SmallerAll      |

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

### Dependency Management

The NVIDIA driver version (`nvidia-smi` command) constraints what maximum CUDA version you can use, which in turn
constraints maximum PyTorch version, which constraints the Python version itself.

The project uses two separate dependency tracks:

- **Development** — managed by Poetry (`pyproject.toml` / `poetry.lock`). Includes dev tools,
  Jupyter, type stubs, etc. Use `make venv` to set up.
- **Container** - TODO

### Build the Image

```bash
# Regenerate pinned requirements (only needed when requirements.in changes)
make container-reqs

# Build
podman build --format docker -t eeg-api -f api.Containerfile .
```

### Publish to Docker Hub and Pull it

```bash
podman login docker.io -u mtichavsky
podman tag eeg-api docker.io/mtichavsky/eeg-classifier-api:latest
podman push docker.io/mtichavsky/eeg-classifier-api:latest
```
The published image is available at:
`docker.io/mtichavsky/eeg-classifier-api`, which you can pull like this:

```bash
podman pull docker.io/mtichavsky/eeg-classifier-api:latest
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

| Variable             | Default       | Description                                                        |
|----------------------|---------------|--------------------------------------------------------------------|
| `MODEL_DIR`          | `/app/models` | Path to model checkpoint directory                                 |
| `DEVICE`             | `auto`        | PyTorch device: `auto`, `cuda`, or `cpu`                           |
| `MODEL_LOADING`      | `startup`     | `startup` (load all on start) or `on_demand` (lazy)                |
| `LOG_LEVEL`          | `INFO`        | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)                |
| `JSON_PRETTY_PRINT`  | `false`       | Set `true` for colored human-readable logs; default is JSON output |
| `MAX_FILE_SIZE_MB`   | `100`         | Maximum upload file size in MB                                     |
| `RATE_LIMIT_TIMES`   | `10`          | Number of requests allowed per rate-limit window                   |
| `RATE_LIMIT_SECONDS` | `10`          | Rate-limit window duration in seconds                              |

### Verify the Deployment

poetry run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload


```bash
# Health check
curl -s http://localhost:8000/health | jq 

# Run a prediction
curl -X POST http://localhost:8000/predict \
    -F "eeg_recording=@../MDD/MDD S1 EO.edf" \
    -F "sampling_rate=256" \
    -F "electrode_setup=in-ear" \
    -F "classification_task=4class" \
    -F "request_id=$(uuidgen)" \
    -F "user_id=$(uuidgen)" | jq
```

http://localhost:8000/docs - docs

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
podman build --format docker -t eeg-api -f api.Containerfile .
```

To use baked-in models, simply omit the `-v` volume mount when running.


## FIT Server Setup

File server: `davs://sc-nas.fit.vutbr.cz:5006/`

```bash
# Load required modules
ml Python/3.12.3-GCCcore-13.3.0
ml CUDA/12.6.0

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install Poetry and project dependencies
pip install poetry
poetry install

# Check GPU and select matching PyTorch wheel
nvidia-smi                                                    # verify GPU is available
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader # get compute capability

# Install PyTorch with CUDA support (adjust URL for your CUDA version)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Point datasets to your data directory (default: /home/milan/Documents/diplomka/)
export EEG_DATA_DIR=/home/xticha09/
```

- Use `CUDA_VISIBLE_DEVICES=2,3` to specify which GPUs the process can use

> **Note:** For running a pre-built PyTorch wheel you don't need to load a CUDA module — PyTorch
> bundles its own CUDA runtime. The module only matters if you're compiling CUDA extensions.
> Loading the matching one is still good practice to avoid surprises.
