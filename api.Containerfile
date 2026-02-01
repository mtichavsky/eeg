# =============================================================================
# EEG Depression/Anxiety Detection — Inference API
#
# Build:  podman build -t eeg-api -f Containerfile .
# Run:    podman run --device nvidia.com/gpu=all -p 8000:8000 \
#             -v ./models:/app/models:ro,Z eeg-api
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Export pinned dependency versions from Poetry lock file
# ---------------------------------------------------------------------------
FROM registry.fedoraproject.org/fedora:43 AS deps

RUN dnf install -y --setopt=install_weak_deps=False python3-pip && \
    dnf clean all && rm -rf /var/cache/dnf

RUN pip install --no-cache-dir poetry

WORKDIR /export
COPY pyproject.toml poetry.lock ./
RUN poetry export --with api --without dev -f requirements.txt -o requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Runtime image
# ---------------------------------------------------------------------------
FROM registry.fedoraproject.org/fedora:43

LABEL maintainer="Milan Tichavský <milantichavsky@seznam.cz>" \
      description="EEG Depression/Anxiety Detection Inference API" \
      version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Python 3.13 ships as the default on Fedora 41.
# libgomp is required by PyTorch at runtime (OpenMP threading).
RUN dnf install -y --setopt=install_weak_deps=False \
        python3 \
        python3-pip \
        libgomp && \
    dnf clean all && \
    rm -rf /var/cache/dnf

WORKDIR /app

# Install Python packages from poetry-pinned versions
COPY --from=deps /export/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Copy only the code needed for inference (api + thesis library + version source)
COPY api/ api/
COPY thesis/ thesis/
COPY pyproject.toml .

# Models directory — mount at runtime or bake in at build time
RUN mkdir -p /app/models

# Default configuration (override via env vars or .env file mounted into /app)
ENV MODEL_DIR=/app/models \
    DEVICE=auto \
    MODEL_LOADING=startup \
    LOG_LEVEL=INFO \
    JSON_PRETTY_PRINT=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]

# Run as non-root for security
USER 1001

CMD ["python3", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
