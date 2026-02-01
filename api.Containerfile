# =============================================================================
# EEG Depression/Anxiety Detection — Inference API
#
# Prerequisites: generate pinned deps locally before building:
#   make container-reqs          # regenerates requirements.txt from requirements.in
#
# Build:  podman build --format docker -t eeg-api -f api.Containerfile .
# Publish: podman tag eeg-api docker.io/mtichavsky/eeg-classifier-api:latest
#          podman push docker.io/mtichavsky/eeg-classifier-api:latest
# Run (GPU):  podman run --device nvidia.com/gpu=all -p 8000:8000 \
#                 -v ./models:/app/models:ro,Z eeg-api
# Run (CPU):  podman run -p 8000:8000 -e DEVICE=cpu \
#                 -v ./models:/app/models:ro,Z eeg-api
# =============================================================================

FROM registry.fedoraproject.org/fedora:41

LABEL maintainer="Milan Tichavský <milantichavsky@seznam.cz>" \
      description="EEG Depression/Anxiety Detection Inference API" \
      version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libgomp is required by PyTorch at runtime (OpenMP threading).
RUN dnf install -y --setopt=install_weak_deps=False \
        python3.12 \
        libgomp && \
    dnf clean all && \
    rm -rf /var/cache/dnf && \
    python3.12 -m ensurepip --upgrade

WORKDIR /app

# All deps — exact versions pinned by pip-compile from requirements.in.
# torch nvidia-* packages bring the CUDA runtime; host provides the driver.
COPY requirements-gpu.txt .
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir -r requirements-gpu.txt
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

# Copy only the code needed for inference
COPY api/ api/
COPY thesis/ thesis/
COPY pyproject.toml .

COPY models/ /app/models

ENV MODEL_DIR=/app/models \
    DEVICE=auto \
    MODEL_LOADING=startup \
    LOG_LEVEL=INFO \
    JSON_PRETTY_PRINT=false \
    NUMBA_CACHE_DIR=/tmp/numba_cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python3.12", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

USER 1001

CMD ["python3.12", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
