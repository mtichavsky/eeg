"""
Pydantic models for API request/response validation.

Provides request/response validation and OpenAPI schema generation.
"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ChunkPrediction(BaseModel):
    """
    Per-chunk prediction information.
    """

    chunk_index: int = Field(..., description="Chunk number (0-indexed)")
    predicted_class: int = Field(..., description="Predicted class label (0-3)")
    class_name: str = Field(..., description="Human-readable class name")


class PredictResponse(BaseModel):
    """
    Response schema for /predict endpoint.

    Includes subject-level aggregated prediction and per-chunk details.
    """

    final_prediction: int = Field(..., description="Final subject-level prediction (majority vote)")
    final_class_name: str = Field(..., description="Human-readable class name")
    total_chunks: int = Field(..., description="Total number of EEG chunks processed")
    class_distribution: dict[str, int] = Field(..., description="Count of chunks per class")
    class_percentages: dict[str, float] = Field(..., description="Percentage of chunks per class")
    chunk_predictions: list[ChunkPrediction] = Field(
        ..., description="Per-chunk prediction details"
    )
    request_id: UUID = Field(..., description="Request identifier (echo from request)")
    model_used: str = Field(..., description="Model identifier used for inference")


class HealthResponse(BaseModel):
    """
    Response schema for /health endpoint.
    """

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ..., description="Overall system health status"
    )
    models_loaded: dict[str, bool] = Field(..., description="Availability status for each model")
    device: str = Field(..., description="Device being used (cuda/cpu)")


class ErrorResponse(BaseModel):
    """
    Standard error response format.
    """

    error: str = Field(..., description="Error type or category")
    message: str = Field(..., description="Human-readable error message")
    detail: str | None = Field(None, description="Additional error details")
    request_id: UUID | None = Field(None, description="Request ID if available")
