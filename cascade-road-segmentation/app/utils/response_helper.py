"""
Response helper utilities for consistent API responses
"""

import time
from datetime import datetime
from typing import Any, Optional
from fastapi import HTTPException
from fastapi.responses import JSONResponse

try:
    from ..models.schemas import APIResponse, ErrorResponse
except ImportError:
    from models.schemas import APIResponse, ErrorResponse


def create_success_response(
    message: str,
    data: Optional[Any] = None,
    status_code: int = 200
) -> JSONResponse:
    """
    Create a successful API response
    
    Args:
        message: Success message
        data: Response data
        status_code: HTTP status code
        
    Returns:
        JSONResponse with standardized format
    """
    response_data = APIResponse(
        success=True,
        message=message,
        data=data,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response_data.model_dump()
    )


def create_error_response(
    message: str,
    error_type: str = "error",
    details: Optional[dict] = None,
    status_code: int = 400
) -> JSONResponse:
    """
    Create an error API response
    
    Args:
        message: Error message
        error_type: Type of error
        details: Additional error details
        status_code: HTTP status code
        
    Returns:
        JSONResponse with error format
    """
    error_data = ErrorResponse(
        error=error_type,
        message=message,
        details=details
    )
    
    response_data = APIResponse(
        success=False,
        message=message,
        data=error_data.model_dump(),
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=status_code,
        content=response_data.model_dump()
    )


def handle_inference_error(error: Exception) -> JSONResponse:
    """
    Handle inference errors with appropriate status codes
    
    Args:
        error: Exception that occurred
        
    Returns:
        JSONResponse with error details
    """
    error_message = str(error)
    
    # Determine appropriate status code based on error type
    if "file not found" in error_message.lower() or "no such file" in error_message.lower():
        status_code = 404
        error_type = "file_not_found"
    elif "out of memory" in error_message.lower() or "cuda" in error_message.lower():
        status_code = 507  # Insufficient Storage
        error_type = "resource_exhausted"
    elif "invalid" in error_message.lower() or "corrupt" in error_message.lower():
        status_code = 400
        error_type = "invalid_input"
    else:
        status_code = 500
        error_type = "inference_error"
    
    return create_error_response(
        message=f"Inference failed: {error_message}",
        error_type=error_type,
        status_code=status_code
    )


def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to native Python types
    """
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def format_inference_result(result: dict) -> dict:
    """
    Format inference result for API response
    
    Args:
        result: Raw inference result from cascade_inference
        
    Returns:
        Formatted result suitable for API response
    """
    # Convert numpy types and remove large arrays
    formatted = convert_numpy_types(result.copy())
    
    # Remove large arrays that shouldn't be in API response
    arrays_to_remove = [
        'overlay_array', 
        'defect_mask_array', 
        'road_mask_array', 
        'confidence_map_array'
    ]
    
    for array_key in arrays_to_remove:
        formatted.pop(array_key, None)
    
    return formatted


class TimingContext:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
    
    @property
    def duration(self) -> float:
        """Get duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def duration_ms(self) -> float:
        """Get duration in milliseconds"""
        return self.duration * 1000


def validate_confidence_threshold(threshold: float) -> float:
    """
    Validate and clamp confidence threshold
    
    Args:
        threshold: Confidence threshold value
        
    Returns:
        Valid threshold value (0.0-1.0)
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError("Confidence threshold must be a number")
    
    return max(0.0, min(1.0, float(threshold)))