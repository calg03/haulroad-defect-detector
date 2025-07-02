"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from enum import Enum


class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"


class DefectClass(str, Enum):
    """Defect class enumeration"""
    BACKGROUND = "background"
    POTHOLE = "pothole"
    CRACK = "crack"
    PUDDLE = "puddle"
    DISTRESSED_PATCH = "distressed_patch"
    MUD = "mud"


class ImageInfo(BaseModel):
    """Image information"""
    name: str = Field(..., description="Image filename")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    size_bytes: int = Field(..., description="File size in bytes")


class DefectCounts(BaseModel):
    """Defect detection counts by class"""
    pothole: int = Field(default=0, description="Number of pothole pixels")
    crack: int = Field(default=0, description="Number of crack pixels")
    puddle: int = Field(default=0, description="Number of puddle pixels")
    distressed_patch: int = Field(default=0, description="Number of distressed patch pixels")
    mud: int = Field(default=0, description="Number of mud pixels")


class InferenceResult(BaseModel):
    """Single image inference result"""
    image_name: str = Field(..., description="Image filename")
    image_shape: List[int] = Field(..., description="Image dimensions [height, width, channels]")
    road_coverage: float = Field(..., description="Percentage of image that is road (0.0-1.0)")
    total_defect_pixels: int = Field(..., description="Total number of defect pixels")
    defect_counts: DefectCounts = Field(..., description="Defect counts by class")
    mean_confidence: float = Field(..., description="Average confidence of defect predictions")
    processing_status: ProcessingStatus = Field(..., description="Processing status")
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    
    # File paths (optional, only when saving to disk)
    overlay_path: Optional[str] = Field(default=None, description="Path to overlay image")
    defect_mask_path: Optional[str] = Field(default=None, description="Path to defect mask image")
    road_mask_path: Optional[str] = Field(default=None, description="Path to road mask image")


class BatchInferenceResult(BaseModel):
    """Batch inference result"""
    total_images: int = Field(..., description="Total number of images processed")
    successful: int = Field(..., description="Number of successfully processed images")
    failed: int = Field(..., description="Number of failed images")
    results: List[InferenceResult] = Field(..., description="Individual image results")


class InferenceRequest(BaseModel):
    """Inference request parameters"""
    save_outputs: bool = Field(default=True, description="Whether to save output images")
    overlay_alpha: float = Field(default=0.6, ge=0.0, le=1.0, description="Overlay transparency")
    confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Confidence threshold")


class ModelInfo(BaseModel):
    """Model information"""
    status: str = Field(..., description="Model initialization status")
    architecture: Optional[str] = Field(default=None, description="Model architecture")
    device: Optional[str] = Field(default=None, description="Device (cuda/cpu)")
    classes: Optional[List[str]] = Field(default=None, description="Defect classes")
    num_classes: Optional[int] = Field(default=None, description="Number of classes")
    confidence_threshold: Optional[float] = Field(default=None, description="Current confidence threshold")
    road_model_path: Optional[str] = Field(default=None, description="Road model file path")
    defect_model_path: Optional[str] = Field(default=None, description="Defect model file path")


class HealthCheck(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    model_status: str = Field(..., description="Model initialization status")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")


class UploadResponse(BaseModel):
    """File upload response"""
    filename: str = Field(..., description="Uploaded filename")
    size_bytes: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="File content type")
    upload_path: str = Field(..., description="Server file path")


# API Response models with metadata
class APIResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[Any] = Field(default=None, description="Response data")
    timestamp: str = Field(..., description="Response timestamp")


class InferenceAPIResponse(APIResponse):
    """Inference API response"""
    data: Optional[InferenceResult] = Field(default=None, description="Inference result")


class BatchInferenceAPIResponse(APIResponse):
    """Batch inference API response"""
    data: Optional[BatchInferenceResult] = Field(default=None, description="Batch inference results")


class ModelInfoAPIResponse(APIResponse):
    """Model info API response"""
    data: Optional[ModelInfo] = Field(default=None, description="Model information")