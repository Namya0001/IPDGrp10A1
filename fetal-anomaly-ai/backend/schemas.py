"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel
from typing import List, Optional


class ClassProbability(BaseModel):
    class_name: str
    class_index: int
    probability: float


class PredictionResult(BaseModel):
    class_name: str
    class_index: int
    confidence: float


class GradCAMResult(BaseModel):
    heatmap_url: str
    overlay_url: str


class PredictResponse(BaseModel):
    success: bool
    prediction: PredictionResult
    probabilities: List[ClassProbability]
    gradcam: GradCAMResult
    low_confidence: bool
    confidence_threshold: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    num_classes: int


class ModelInfoResponse(BaseModel):
    model: str
    task: str
    input_size: str
    num_classes: int
    gradcam_layer: str
    preprocessing: dict


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
