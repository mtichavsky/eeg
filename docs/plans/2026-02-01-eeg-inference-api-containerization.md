# EEG Inference API Containerization - Implementation Plan

**Date:** 2026-02-01
**Author:** Milan Tichavský
**Status:** Planning Complete - Ready for Implementation

## Executive Summary

This plan details the implementation of a production-ready FastAPI inference server for EEG-based depression and anxiety classification. The server will support both EDF and CSV file formats, handle single-channel and multi-channel (8-channel) EEG recordings, and provide both 2-class (binary) and 4-class (multi-class) classification modes. The system will be containerized with Docker for deployment, with all 4 trained models baked into the container image.

## Requirements Overview

From the PDF specification and user clarifications:

### API Endpoints

1. **`/predict`** - Main inference endpoint
   - Input: `electrode_setup` (single/all), `eeg_recording` (file), `sampling_rate` (int), `classification_task` (2class/4class), `request_id` (UUID), `user_id` (UUID)
   - File formats: EDF and CSV
   - Output: Subject-level and per-chunk predictions with class distribution

2. **`/health`** - Health check endpoint
   - Returns 200 OK with model availability status

3. **No `/predict-online`** for v1 (deferred)

### The 4 Models to Bake In

1. **Single-channel, 2-class** (binary: healthy vs pathological)
2. **Single-channel, 4-class** (normal/anxiety/depression/comorbid)
3. **8-channel, 2-class** (binary: healthy vs pathological)
4. **8-channel, 4-class** (normal/anxiety/depression/comorbid)

**Model Selection Logic:** Automatic selection based on `electrode_setup` + `classification_task` parameters.

### Architecture Requirements

- FastAPI with async request handling
- Models loaded once at startup (avoid per-request overhead)
- Structured JSON logging with `request_id` and `user_id` in every log
- Rate limiting using slowapi
- Docker containerization with GPU support (nvidia-container-toolkit)
- OpenAPI documentation (auto-generated)
- Uvicorn with --workers flag

## File Structure

```
/home/milan/Documents/diplomka/code--integration/
├── api/
│   ├── __init__.py                    # API package initialization
│   ├── app.py                         # FastAPI application (main entry point)
│   │                                  # Calls main.py and thesis/dataset.py functions directly
│   ├── schemas.py                     # Pydantic request/response models
│   ├── model_manager.py               # Model loading and selection logic
│   ├── config.py                      # API configuration (paths, constants)
│   └── logging_config.py              # Structured JSON logging setup
├── container/
│   ├── Containerfile                  # Multi-stage build with PyTorch CUDA base
│   ├── .containerignore               # Files to exclude from build context
│   └── build_and_push.py              # Build script for containerization (uses podman)
├── models/                            # Directory for model checkpoints (gitignored)
│   ├── model_single_2class.pth       # Single-channel, binary classifier
│   ├── model_single_4class.pth       # Single-channel, 4-class classifier
│   ├── model_multi_2class.pth        # 8-channel, binary classifier
│   └── model_multi_4class.pth        # 8-channel, 4-class classifier
├── thesis/
│   └── dataset.py                     # Updated: Add format-specific preprocessing
│                                      # (load_and_preprocess_edf_file, load_and_preprocess_csv_file)
├── tests/
│   ├── test_api.py                    # API endpoint tests
│   ├── test_inference.py              # Inference logic tests
│   └── test_preprocessing.py          # Preprocessing tests
├── pyproject.toml                     # Updated with API dependencies
├── .env.example                       # Example environment variables
└── README_API.md                      # API deployment documentation
```

## Critical Implementation Files

### 1. `api/config.py` - Configuration Management

**Purpose:** Centralized configuration with environment variable support.

**Key Features:**
- Pydantic-settings for type-safe configuration
- Model paths configurable for dev and production
- Rate limiting settings
- STFT parameters (nperseg=256, noverlap=192)

**Environment Variables:**
```
MODEL_DIR=/app/models
DEVICE=auto  # cuda, cpu, or auto
MAX_FILE_SIZE_MB=100
RATE_LIMIT_TIMES=10
RATE_LIMIT_SECONDS=60
LOG_LEVEL=INFO
JSON_LOGGING=true
```

### 2. `api/logging_config.py` - Structured Logging

**Purpose:** JSON logging with automatic request tracing.

**Key Features:**
- Uses `structlog` for structured logging
- Middleware adds `request_id` and `user_id` to all logs
- Production: JSON output for log aggregation
- Development: Colored human-readable logs

**Example Log Entry:**
```json
{
  "timestamp": "2026-02-01T14:30:45.123Z",
  "level": "INFO",
  "message": "Prediction successful",
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "user_id": "223e4567-e89b-12d3-a456-426614174000",
  "final_prediction": 0,
  "final_class_name": "Healthy"
}
```

### 3. `api/schemas.py` - Pydantic Models

**Purpose:** Request/response validation and OpenAPI schema generation.

**Key Schemas:**
- `PredictRequest`: Validates input parameters
- `PredictResponse`: Subject-level + chunk-level predictions
- `ErrorResponse`: Consistent error format
- `HealthResponse`: Model availability status

**Example Response:**
```json
{
  "final_prediction": 0,
  "final_class_name": "Healthy",
  "total_chunks": 60,
  "class_distribution": {"Healthy": 45, "Pathological": 15},
  "class_percentages": {"Healthy": 75.0, "Pathological": 25.0},
  "chunk_predictions": [...],
  "request_id": "123e4567-e89b-12d3-a456-426614174000",
  "model_used": "single_2class"
}
```

### 4. `thesis/dataset.py` - Updated Preprocessing (Format-Specific)

**Purpose:** Add format-specific preprocessing functions (not dataset-specific).

**New Functions to Add:**
- `load_and_preprocess_edf_file()`: Generic EDF preprocessing (works for any EDF, not just MDD)
  - Uses 250 Hz sampling rate
  - Bandpass 1-70 Hz, notch 50 Hz
  - Returns `torch.Tensor` of shape `(num_chunks, num_channels, 2500)`

- `load_and_preprocess_csv_file()`: Generic CSV preprocessing (works for any CSV, not just CANE)
  - Flexible sampling rate parameter
  - Same filtering pipeline
  - Optional artifact removal
  - Returns `torch.Tensor` of shape `(num_chunks, num_channels, 2500)`

**Existing Functions to Keep:**
- Keep `load_and_preprocess_mdd_raw_file()` and `load_and_preprocess_cane_raw_file()` as wrappers if needed
- API will call the new format-specific functions directly

### 5. `api/model_manager.py` - Model Loading and Selection

**Purpose:** Load 4 models at startup and select based on request parameters.

**Model Selection Table:**

| electrode_setup | classification_task | Model File                  | Architecture    | Channels |
|-----------------|---------------------|-----------------------------|--------------------|----------|
| single          | 2class              | model_single_2class.pth    | CNN_LSTM_DepCap    | 1        |
| single          | 4class              | model_single_4class.pth    | CNN_LSTM_DepCap    | 1        |
| all             | 2class              | model_multi_2class.pth     | SmallerAll         | 8        |
| all             | 4class              | model_multi_4class.pth     | SmallerAll         | 8        |

**Key Methods:**
- `__init__()`: Load all 4 models at startup
- `select_model()`: Return model based on electrode_setup + classification_task
- `get_model_status()`: Check which models are available
- `get_channel_name()`: Map electrode_setup to channel parameter ("Fp1" or "all")

**Graceful Degradation:** If a model fails to load, log error but don't crash (allows partial availability).

### 6. Inference Logic - Reuse from `main.py`

**Purpose:** Reuse existing inference logic from `main.py:run()` - NO duplication.

**Approach:**
- Option 1: Extract minimal wrapper that calls logic from `main.py:run()`
- Option 2: Refactor `main.py:run()` to be callable as a library function
- Option 3: Inline the inference logic in `api/app.py` calling existing functions

**Required Steps (already in main.py):**
1. Convert EEG chunks to spectrograms via `SpectrogramDataset.convert_to_spectrograms()`
2. Run model inference on each chunk
3. Aggregate predictions via majority voting
4. Compute class distribution and percentages

**Class Mappings:**
- 2-class: `{0: "Healthy", 1: "Pathological"}`
- 4-class: `{0: "Healthy", 1: "Anxiety", 2: "Depression", 3: "Comorbid"}`

**Key Point:** API will call `SpectrogramDataset.convert_to_spectrograms()` and reuse inference loop from main.py without copying code.

### 7. `api/app.py` - FastAPI Application

**Purpose:** Main application entry point with endpoints and middleware.

**Lifecycle:**
- **Startup:** Load all 4 models via `ModelManager` (which calls `main.py:create_model()`)
- **Request:** Handle file upload, preprocessing, inference
- **Shutdown:** Cleanup resources

**Middleware:**
- Request context logging (adds request_id and user_id to logs)
- Rate limiting (slowapi)

**Endpoints:**
- `GET /health`: Returns model availability status
- `POST /predict`: Main inference endpoint (multipart form data)

**Error Handling:**
- 400: Validation errors (invalid electrode_setup, etc.)
- 413: File too large
- 422: Preprocessing errors (corrupt file, invalid EEG data)
- 429: Rate limit exceeded
- 500: Internal server error

**File Upload Flow (using existing codebase):**
1. Receive multipart form data (file + metadata)
2. Validate inputs
3. Save file to temporary location
4. Select model based on electrode_setup + classification_task (via `ModelManager`)
5. Detect format (EDF/CSV) from file extension
6. Preprocess file by calling `thesis/dataset.py:load_and_preprocess_edf_file()` or `load_and_preprocess_csv_file()`
7. Convert to spectrograms by calling `SpectrogramDataset.convert_to_spectrograms()`
8. Run inference (reuse logic from `main.py:run()`)
9. Return response
10. Cleanup temporary file

**Key Point:** NO duplicated code - all logic calls existing functions from main.py and thesis/dataset.py

## Container Containerization

### Containerfile (`container/Containerfile`)

**Base Image:** `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`

**Multi-stage Build:**
1. **Stage 1 (base):** Install system dependencies + Poetry
2. **Stage 2 (dependencies):** Install Python dependencies (production only)
3. **Stage 3 (application):** Copy code and models

**Key Features:**
- Models baked into image at build time via `COPY` instructions
- Health check endpoint integration
- Single worker (GPU doesn't benefit from multiple workers)
- Exposed port: 8000

**Build Args:**
```dockerfile
ARG MODEL_SINGLE_2CLASS_PATH
ARG MODEL_SINGLE_4CLASS_PATH
ARG MODEL_MULTI_2CLASS_PATH
ARG MODEL_MULTI_4CLASS_PATH
```

**Entrypoint:**
```dockerfile
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### Build Script (`container/build_and_push.py`)

**Purpose:** Automate container image building with model validation using podman.

**Usage:**
```bash
python container/build_and_push.py \
    --model-single-2class experiments/mdd_006_fp1_ec/fold_1_best.pth \
    --model-single-4class experiments/both_012_fp1_ec/fold_1_best.pth \
    --model-multi-2class experiments/both_007_all_ec/fold_1_best.pth \
    --model-multi-4class experiments/both_015_all_ec/fold_1_best.pth \
    --tag v1.0.0 \
    --registry your-dockerhub-username \
    --push
```

**Features:**
- Validates all model checkpoint paths exist before building
- Uses `podman build -f container/Containerfile` instead of docker
- Passes model paths as build args
- Optional `--push` flag for Docker Hub upload (using `podman push`)
- Provides usage examples after successful build

## Dependencies

Add to `pyproject.toml`:

```toml
[tool.poetry.dependencies]
fastapi = "^0.109.0"
uvicorn = {extras = ["standard"], version = "^0.27.0"}
python-multipart = "^0.0.6"  # File upload support
pydantic = "^2.5.0"
pydantic-settings = "^2.1.0"
slowapi = "^0.1.9"  # Rate limiting
structlog = "^24.1.0"  # Structured logging
```

Install:
```bash
poetry install
```

## Implementation Steps

### Phase 1: Infrastructure Setup (30 min)
1. Create `api/` directory structure
2. Create `api/__init__.py`
3. Implement `api/config.py`
4. Implement `api/logging_config.py`

### Phase 2: Core Logic Updates (1-2 hours)
5. Implement `api/schemas.py`
6. Update `thesis/dataset.py` to add format-specific preprocessing:
   - Add `load_and_preprocess_edf_file()` - generic EDF preprocessing
   - Add `load_and_preprocess_csv_file()` - generic CSV preprocessing
   - Keep existing MDD/CANE functions as wrappers if needed

### Phase 3: Model Management (1 hour)
7. Implement `api/model_manager.py`
   - Use `main.py:create_model()` for model instantiation (no duplication)
   - Load all 4 models at startup
   - Implement model selection logic
8. Test model loading with sample checkpoints

### Phase 4: FastAPI Application (1-2 hours)
9. Implement `api/app.py`
   - Call `thesis/dataset.py` format-specific functions for preprocessing
   - Call `SpectrogramDataset.convert_to_spectrograms()` for STFT
   - Reuse inference logic from `main.py:run()` (no duplication)
10. Add middleware for logging
11. Implement `/health` endpoint
12. Implement `/predict` endpoint

### Phase 5: Local Testing (1 hour)
14. Update `pyproject.toml` dependencies
15. Run `poetry install`
16. Start local server: `uvicorn api.app:app --reload`
17. Test with curl/Postman:
    ```bash
    curl http://localhost:8000/health

    curl -X POST "http://localhost:8000/predict" \
      -F "eeg_recording=@/path/to/test.edf" \
      -F "electrode_setup=single" \
      -F "classification_task=2class" \
      -F "sampling_rate=250" \
      -F "request_id=123e4567-e89b-12d3-a456-426614174000" \
      -F "user_id=223e4567-e89b-12d3-a456-426614174000"
    ```

### Phase 6: Containerization (1-2 hours)
18. Create `container/` directory
19. Implement `container/Containerfile`
20. Create `container/.containerignore`
21. Implement `container/build_and_push.py` (uses podman instead of docker)

### Phase 7: Container Testing (1 hour)
22. Build container image (without push):
    ```bash
    python container/build_and_push.py \
        --model-single-2class <path> \
        --model-single-4class <path> \
        --model-multi-2class <path> \
        --model-multi-4class <path> \
        --tag test \
        --registry your-username
    ```
23. Run container locally:
    ```bash
    podman run --device nvidia.com/gpu=all -p 8000:8000 your-username/eeg-inference-api:test
    ```
24. Test endpoints
25. Verify GPU access: `podman exec -it <container> nvidia-smi`

### Phase 8: Production Deployment (30 min)
26. Create `.env.example`
27. Create `README_API.md` with deployment instructions
28. Build final image with `--push` flag
29. Deploy to production environment

**Total Estimated Time:** 7-10 hours

## Testing Strategy

### Unit Tests

Create `tests/test_api.py`:
```python
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_invalid_electrode_setup(client):
    response = client.post("/predict", data={"electrode_setup": "invalid", ...})
    assert response.status_code == 400
```

Run: `poetry run pytest tests/test_api.py -v`

### Integration Tests

1. Test with real EDF files (MDD dataset)
2. Test with real CSV files (CANE dataset)
3. Verify predictions match `main.py run` output
4. Test all 4 model combinations
5. Test error cases (invalid file, wrong format, etc.)

### Docker Tests

1. Health check returns 200
2. Prediction endpoint works with sample files
3. GPU is accessible (`nvidia-smi` inside container)
4. Rate limiting works (exceed 10 requests/minute)
5. Logs contain request_id and user_id

## Production Deployment Checklist

- [ ] All 4 models trained and validated
- [ ] Model checkpoints available (`.pth` files)
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Docker image builds successfully
- [ ] Health endpoint returns 200 OK
- [ ] Prediction endpoint tested with EDF and CSV
- [ ] GPU access verified in container
- [ ] Rate limiting tested
- [ ] Logs contain request_id and user_id
- [ ] Error responses match schema
- [ ] Docker image pushed to Docker Hub
- [ ] `README_API.md` created with deployment instructions

## Key Design Decisions

### 1. Why Reuse Existing Code?
- Avoid duplication between CLI and API
- Same preprocessing/inference ensures consistent results
- Easier maintenance (one codebase, one source of truth)
- API calls `main.py:create_model()`, `thesis/dataset.py` functions, `SpectrogramDataset.convert_to_spectrograms()`

### 2. Why Format-Specific Preprocessing?
- Current code is dataset-specific (MDD/CANE) but API receives arbitrary EDF/CSV files
- Need generic `load_and_preprocess_edf_file()` and `load_and_preprocess_csv_file()`
- Existing MDD/CANE functions can become thin wrappers

### 3. Why Podman/Containerfile?
- User preference for Containerfile naming
- Podman is daemonless, rootless alternative to Docker
- Compatible with Docker Hub registry
- GPU support via `--device nvidia.com/gpu=all`

### 4. Why Single Worker?
GPU inference doesn't benefit from multiple Uvicorn workers. Use `--workers 1` and rely on async/await for I/O concurrency.

### 5. Why Bake Models into Image?
- Simplifies deployment (no external model storage needed)
- Ensures model versioning matches container version
- Acceptable size increase (~200-400 MB per model = ~1.6 GB total)
- Alternative: Persistent volume claims or Hugging Face hosting (deferred)

### 6. Why Automatic Model Selection?
- Simpler client API (fewer parameters)
- Prevents client errors (selecting wrong model for electrode_setup)
- Server-side logic ensures correct model is used

### 7. Why Structured Logging?
- Enables distributed tracing across services
- Log aggregation systems (ELK, Datadog) prefer JSON
- Request_id tracing required per PDF spec

### 8. Why Skip Artifact Removal?
- Faster inference (artifact removal is slow)
- Model can learn to handle artifacts
- Configurable if needed later

## Future Enhancements (Out of Scope for v1)

- `/predict-online` endpoint for real-time classification
- Batch prediction endpoint (multiple files)
- Model confidence scores in response
- A/B testing between different model versions
- Prometheus metrics for monitoring
- Redis-backed rate limiting (shared across instances)
- Model versioning and rollback
- Persistent volume for external model storage

## References

- PDF Specification: `/home/milan/Documents/diplomka/code--integration/Cloud app integration.pdf`
- Existing inference code: `main.py:921-1021` (run function)
- Model creation: `main.py:422-501` (create_model function)
- Preprocessing: `thesis/dataset.py` (MDDDataset, CANEDataset)
- Model registry: `main.py` (MODEL_REGISTRY)

## Questions for User (Before Implementation)

None - all clarifications received:
- ✅ 4 models: single/all × 2class/4class
- ✅ File formats: EDF and CSV (both implemented)
- ✅ Sampling rate: API parameter
- ✅ Model selection: Automatic based on electrode_setup + classification_task
- ✅ Checkpoints: User has trained models ready
