"""
FastAPI application for EEG-based depression and anxiety classification.

Main entry point for the inference API. Provides endpoints for:
- /health: Health check and model availability
- /predict: EEG file classification (EDF and CSV formats)
"""

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from uuid import UUID

import structlog
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.config import config
from api.logging_config import get_logger, setup_logging
from api.model_manager import ModelManager
from api.schemas import (
    ChunkPrediction,
    ErrorResponse,
    HealthResponse,
    PredictResponse,
)
from thesis.inference import preprocess_and_infer
from thesis.version import __version__

setup_logging()
logger = get_logger(__name__)

# Global model manager (initialized on startup)
model_manager: ModelManager | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Manage application lifespan: initialize ModelManager on startup.

    Loading strategy (startup vs on_demand) is controlled by
    the MODEL_LOADING environment variable.
    """
    global model_manager
    logger.info("Starting up EEG Inference API")
    logger.info(f"Device: {config.get_device()}")
    logger.info(f"Model directory: {config.model_dir}")

    try:
        model_manager = ModelManager()
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ModelManager: {e}", exc_info=True)
        raise

    yield


app = FastAPI(
    title="EEG Inference API",
    description="Depression and anxiety classification from EEG recordings",
    version=__version__,
    lifespan=lifespan,
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    """
    Reject requests exceeding the configured upload size limit.

    Checks the Content-Length header before reading the body so oversized
    requests are rejected without buffering them into memory. Clients using
    chunked transfer encoding (no Content-Length) are caught by the in-endpoint check.
    """
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            size = int(content_length)
        except ValueError:
            pass
        else:
            if size > config.max_file_size_bytes:
                return JSONResponse(
                    status_code=413,
                    content=ErrorResponse(
                        error="HTTP 413",
                        message=f"Request too large: {size} bytes (max: {config.max_file_size_bytes})",
                        detail=None,
                        request_id=None,
                    ).model_dump(),
                )
    return await call_next(request)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Middleware to add generic request context to all logs.

    Clears structlog context vars and logs method/path. Request-specific
    context (request_id, user_id) is bound in the predict() endpoint
    where form data is available.
    """
    structlog.contextvars.clear_contextvars()
    logger.info("Request received", method=request.method, path=request.url.path)

    response = await call_next(request)
    return response


@app.get("/health", response_model=HealthResponse)
@limiter.limit(f"{config.rate_limit_times}/{config.rate_limit_seconds}seconds")
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.

    Returns API status and model availability.

    :param Request _request: FastAPI request object (required by rate limiter).
    :return: Health status response.
    :rtype: HealthResponse
    """
    if model_manager is None:
        return HealthResponse(
            status="unhealthy",
            models_loaded={},
            device=str(config.get_device()),
        )

    models_status = model_manager.get_model_status()
    all_loaded = all(models_status.values())
    some_loaded = any(models_status.values())

    status: Literal["healthy", "degraded", "unhealthy"]
    if all_loaded:
        status = "healthy"
    elif some_loaded:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        models_loaded=models_status,
        device=str(config.get_device()),
    )


@app.post("/predict", response_model=PredictResponse)
@limiter.limit(f"{config.rate_limit_times}/{config.rate_limit_seconds}seconds")
async def predict(
    request: Request,
    eeg_recording: UploadFile = File(...),
    electrode_setup: Literal["in-ear", "8channel"] = Form(...),
    classification_task: Literal["2class", "4class"] = Form(...),
    fs: int | None = Form(None),
    request_id: UUID = Form(...),
    user_id: UUID = Form(...),
) -> PredictResponse:
    """
    Predict depression/anxiety from EEG recording.

    Accepts EDF or CSV files, preprocesses them, and runs inference
    using the appropriate model based on electrode_setup and classification_task.

    :param Request _request: FastAPI request object (for rate limiting).
    :param UploadFile eeg_recording: Uploaded EEG file (EDF or CSV).
    :param Literal["in-ear", "8channel"] electrode_setup: Electrode configuration.
    :param Literal["2class", "4class"] classification_task: Classification mode.
    :param int | None fs: Sampling rate in Hz. Auto-detected from the file if omitted.
    :param UUID request_id: Request identifier for tracing.
    :param UUID user_id: User identifier for tracing.
    :return: Prediction response with subject-level and chunk-level results.
    :rtype: PredictResponse
    :raises HTTPException: On validation, preprocessing, or inference errors.
    """
    # Bind request context to logs
    structlog.contextvars.bind_contextvars(request_id=str(request_id), user_id=str(user_id))

    logger.info(
        "Prediction request received",
        electrode_setup=electrode_setup,
        classification_task=classification_task,
        fs=fs,
        filename=eeg_recording.filename,
    )

    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not initialized")

    # Validate file size
    file_content = await eeg_recording.read()
    if len(file_content) > config.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(file_content)} bytes (max: {config.max_file_size_bytes})",
        )

    # Detect file format
    file_ext = Path(eeg_recording.filename or "").suffix.lower()
    if file_ext not in [".edf", ".csv"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: .edf, .csv",
        )

    # Save to temporary file
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(file_content)
            temp_path = Path(temp_file.name)

        logger.info(f"Saved uploaded file to {temp_path}")

        # Select model
        try:
            model = model_manager.select_model(electrode_setup, classification_task)
            num_classes = model_manager.get_num_classes(electrode_setup, classification_task)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Get channel parameter
        channel = model_manager.get_channel_name(electrode_setup)

        # Run shared inference pipeline
        try:
            result = preprocess_and_infer(
                file_path=temp_path,
                model=model,
                device=model_manager.device,
                channel=channel,
                file_format=file_ext,
                num_classes=num_classes,
                sampling_rate=fs,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

        logger.info(
            "Prediction successful",
            final_prediction=result.final_prediction,
            final_class_name=result.final_class_name,
        )

        # Build response
        model_key = f"{electrode_setup}_{classification_task}"
        return PredictResponse(
            final_prediction=result.final_prediction,
            final_class_name=result.final_class_name,
            total_chunks=result.total_chunks,
            class_distribution=result.class_distribution,
            class_percentages=result.class_percentages,
            chunk_predictions=[
                ChunkPrediction(
                    chunk_index=cr.chunk_index,
                    predicted_class=cr.predicted_class,
                    class_name=cr.class_name,
                )
                for cr in result.chunk_results
            ],
            request_id=request_id,
            model_used=model_key,
        )

    finally:
        # Cleanup temporary file
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
                logger.info(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file: {e}")


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    """
    Custom exception handler for HTTPException.

    Returns error in consistent ErrorResponse format.
    """
    request_id = structlog.contextvars.get_contextvars().get("request_id")

    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"HTTP {exc.status_code}",
            message=exc.detail or "An error occurred",
            detail=None,
            request_id=request_id,
        ).model_dump(),
    )
