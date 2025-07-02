"""
FastAPI endpoints for road defect segmentation
"""

import os
import time
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import cv2
import numpy as np

try:
    from ..core.config import settings
    from ..core.inference_engine import inference_engine
    from ..models.schemas import (
        InferenceResult, 
        BatchInferenceResult, 
        ModelInfo, 
        HealthCheck,
        InferenceRequest,
        DefectCounts
    )
    from ..utils.file_handler import (
        process_upload, 
        create_output_directory, 
        safe_remove_file,
        get_file_info
    )
    from ..utils.response_helper import (
        create_success_response, 
        create_error_response, 
        handle_inference_error,
        format_inference_result,
        TimingContext,
        validate_confidence_threshold
    )
except ImportError:
    # Fallback for direct execution
    from core.config import settings
    from core.inference_engine import inference_engine
    from models.schemas import (
        InferenceResult, 
        BatchInferenceResult, 
        ModelInfo, 
        HealthCheck,
        InferenceRequest,
        DefectCounts
    )
    from utils.file_handler import (
        process_upload, 
        create_output_directory, 
        safe_remove_file,
        get_file_info
    )
    from utils.response_helper import (
        create_success_response, 
        create_error_response, 
        handle_inference_error,
        format_inference_result,
        TimingContext,
        validate_confidence_threshold
    )

# Create router
router = APIRouter()

# Track service start time for uptime calculation
SERVICE_START_TIME = time.time()


@router.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    try:
        model_info = inference_engine.get_model_info()
        model_status = model_info.get("status", "unknown")
        
        health_data = HealthCheck(
            status="healthy" if model_status == "initialized" else "degraded",
            timestamp=datetime.utcnow().isoformat(),
            version=settings.app_version,
            model_status=model_status,
            uptime_seconds=time.time() - SERVICE_START_TIME
        )
        
        return health_data
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about loaded models"""
    try:
        model_info = inference_engine.get_model_info()
        return ModelInfo(**model_info)
    except Exception as e:
        logging.error(f"Failed to get model info: {e}")
        return create_error_response(
            message="Failed to retrieve model information",
            error_type="model_error",
            status_code=500
        )


@router.post("/predict/single")
async def predict_single_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file to process"),
    save_outputs: bool = Form(default=True, description="Save output images to disk"),
    overlay_alpha: float = Form(default=0.6, description="Overlay transparency (0.0-1.0)"),
    confidence_threshold: float = Form(default=0.6, description="Confidence threshold (0.0-1.0)")
):
    """
    Run defect segmentation on a single image
    
    - **file**: Image file (jpg, png, etc.)
    - **save_outputs**: Whether to save result images to disk
    - **overlay_alpha**: Overlay transparency (0.0 = transparent, 1.0 = opaque)
    - **confidence_threshold**: Minimum confidence for defect detection
    """
    temp_files = []  # Track files for cleanup
    
    try:
        # Validate parameters
        overlay_alpha = max(0.0, min(1.0, overlay_alpha))
        confidence_threshold = validate_confidence_threshold(confidence_threshold)
        
        # Process upload
        with TimingContext("file_upload") as upload_timer:
            file_path, file_info = await process_upload(file)
            temp_files.append(file_path)
        
        logging.info(f"File uploaded in {upload_timer.duration_ms:.1f}ms: {file_info['original_filename']}")
        
        # Update inference settings
        inference_engine._inference.confidence_threshold = confidence_threshold
        
        # Create output directory if saving outputs
        output_dir = None
        if save_outputs:
            output_dir = create_output_directory(
                os.path.splitext(file_info['original_filename'])[0]
            )
            temp_files.append(output_dir)
        
        # Run inference
        with TimingContext("inference") as inference_timer:
            result = inference_engine.predict(file_path, output_dir)
        
        logging.info(f"Inference completed in {inference_timer.duration_ms:.1f}ms")
        
        # Format result for API response
        formatted_result = format_inference_result(result)
        
        # Add processing metadata
        formatted_result.update({
            'upload_time_ms': upload_timer.duration_ms,
            'inference_time_ms': inference_timer.duration_ms,
            'total_time_ms': upload_timer.duration + inference_timer.duration,
            'file_info': {
                'original_filename': file_info['original_filename'],
                'size_bytes': file_info['size_bytes'],
                'content_type': file_info['content_type']
            }
        })
        
        # Convert to schema
        inference_result = InferenceResult(
            image_name=formatted_result['image_name'],
            image_shape=formatted_result['image_shape'],
            road_coverage=formatted_result['road_coverage'],
            total_defect_pixels=formatted_result['total_defect_pixels'],
            defect_counts=DefectCounts(**formatted_result['defect_counts']),
            mean_confidence=formatted_result['mean_confidence'],
            processing_status=formatted_result['processing_status'],
            error_message=formatted_result.get('error_message'),
            overlay_path=formatted_result.get('overlay_path'),
            defect_mask_path=formatted_result.get('defect_mask_path'),
            road_mask_path=formatted_result.get('road_mask_path')
        )
        
        # Schedule cleanup
        if not save_outputs:
            background_tasks.add_task(cleanup_temp_files, temp_files)
        
        return create_success_response(
            message="Inference completed successfully",
            data=inference_result.model_dump()
        )
        
    except Exception as e:
        logging.error(f"Inference error: {e}")
        # Clean up on error
        background_tasks.add_task(cleanup_temp_files, temp_files)
        return handle_inference_error(e)


@router.post("/predict/batch")
async def predict_batch_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Image files to process"),
    save_outputs: bool = Form(default=True, description="Save output images to disk"),
    overlay_alpha: float = Form(default=0.6, description="Overlay transparency (0.0-1.0)"),
    confidence_threshold: float = Form(default=0.6, description="Confidence threshold (0.0-1.0)")
):
    """
    Run defect segmentation on multiple images
    
    - **files**: List of image files
    - **save_outputs**: Whether to save result images to disk
    - **overlay_alpha**: Overlay transparency 
    - **confidence_threshold**: Minimum confidence for defect detection
    """
    temp_files = []
    
    try:
        # Validate parameters
        overlay_alpha = max(0.0, min(1.0, overlay_alpha))
        confidence_threshold = validate_confidence_threshold(confidence_threshold)
        
        # Limit batch size
        max_batch_size = 10
        if len(files) > max_batch_size:
            raise HTTPException(
                status_code=400, 
                detail=f"Batch size too large. Maximum {max_batch_size} files allowed."
            )
        
        # Process uploads
        file_paths = []
        file_infos = []
        
        with TimingContext("batch_upload") as upload_timer:
            for file in files:
                file_path, file_info = await process_upload(file)
                file_paths.append(file_path)
                file_infos.append(file_info)
                temp_files.append(file_path)
        
        logging.info(f"Batch upload completed in {upload_timer.duration_ms:.1f}ms: {len(files)} files")
        
        # Update inference settings
        inference_engine._inference.confidence_threshold = confidence_threshold
        
        # Create output directory if saving outputs
        output_dir = None
        if save_outputs:
            output_dir = create_output_directory("batch_inference")
            temp_files.append(output_dir)
        
        # Run batch inference
        with TimingContext("batch_inference") as inference_timer:
            batch_result = inference_engine.predict_batch(file_paths, output_dir)
        
        logging.info(f"Batch inference completed in {inference_timer.duration_ms:.1f}ms")
        
        # Format results
        formatted_results = []
        for i, result in enumerate(batch_result['results']):
            formatted_result = format_inference_result(result)
            # Add file info
            if i < len(file_infos):
                formatted_result['file_info'] = {
                    'original_filename': file_infos[i]['original_filename'],
                    'size_bytes': file_infos[i]['size_bytes'],
                    'content_type': file_infos[i]['content_type']
                }
            formatted_results.append(formatted_result)
        
        # Convert to schema
        inference_results = []
        for result in formatted_results:
            inference_result = InferenceResult(
                image_name=result['image_name'],
                image_shape=result['image_shape'],
                road_coverage=result['road_coverage'],
                total_defect_pixels=result['total_defect_pixels'],
                defect_counts=DefectCounts(**result['defect_counts']),
                mean_confidence=result['mean_confidence'],
                processing_status=result['processing_status'],
                error_message=result.get('error_message'),
                overlay_path=result.get('overlay_path'),
                defect_mask_path=result.get('defect_mask_path'),
                road_mask_path=result.get('road_mask_path')
            )
            inference_results.append(inference_result)
        
        batch_inference_result = BatchInferenceResult(
            total_images=batch_result['total_images'],
            successful=batch_result['successful'],
            failed=batch_result['failed'],
            results=inference_results
        )
        
        # Add processing metadata
        response_data = batch_inference_result.model_dump()
        response_data.update({
            'upload_time_ms': upload_timer.duration_ms,
            'inference_time_ms': inference_timer.duration_ms,
            'total_time_ms': upload_timer.duration + inference_timer.duration,
            'average_time_per_image_ms': inference_timer.duration_ms / len(files) if files else 0
        })
        
        # Schedule cleanup
        if not save_outputs:
            background_tasks.add_task(cleanup_temp_files, temp_files)
        
        return create_success_response(
            message=f"Batch inference completed: {batch_result['successful']}/{batch_result['total_images']} successful",
            data=response_data
        )
        
    except Exception as e:
        logging.error(f"Batch inference error: {e}")
        # Clean up on error
        background_tasks.add_task(cleanup_temp_files, temp_files)
        return handle_inference_error(e)


@router.get("/results/{filename}")
async def get_result_file(filename: str):
    """
    Download a result file (overlay, mask, etc.)
    
    - **filename**: Name of the result file to download
    """
    try:
        # Look for file in output directory
        file_path = os.path.join(settings.output_dir, filename)
        
        # Security check: ensure file is within output directory
        if not os.path.abspath(file_path).startswith(os.path.abspath(settings.output_dir)):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not os.path.exists(file_path):
            # Try to find file in subdirectories
            for root, dirs, files in os.walk(settings.output_dir):
                if filename in files:
                    file_path = os.path.join(root, filename)
                    break
            else:
                raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type
        media_type = "image/png"
        if filename.lower().endswith(('.jpg', '.jpeg')):
            media_type = "image/jpeg"
        elif filename.lower().endswith('.bmp'):
            media_type = "image/bmp"
        elif filename.lower().endswith('.tiff'):
            media_type = "image/tiff"
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error serving file {filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve file")


@router.get("/results/{result_id}/overlay")
async def get_overlay_image(result_id: str):
    """
    Get overlay image as a response stream
    
    - **result_id**: Result identifier (typically image name without extension)
    """
    try:
        # Look for overlay file
        overlay_filename = f"{result_id}_overlay.png"
        file_path = None
        
        # Search in output directory
        for root, dirs, files in os.walk(settings.output_dir):
            if overlay_filename in files:
                file_path = os.path.join(root, overlay_filename)
                break
        
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Overlay image not found")
        
        return FileResponse(
            path=file_path,
            media_type="image/png",
            filename=overlay_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error serving overlay for {result_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve overlay image")


@router.get("/browse")
async def browse_results():
    """
    Browse all available result files and directories
    
    Returns a structured list of all generated results for easy access
    """
    try:
        results = []
        
        if not os.path.exists(settings.output_dir):
            return create_success_response(
                message="No results directory found", 
                data={"results": [], "output_directory": settings.output_dir}
            )
        
        # Walk through output directory
        for root, dirs, files in os.walk(settings.output_dir):
            if files:  # Only include directories with files
                relative_path = os.path.relpath(root, settings.output_dir)
                
                # Group files by type
                overlays = [f for f in files if f.endswith('_overlay.png')]
                defects = [f for f in files if f.endswith('_defects.png')]
                roads = [f for f in files if f.endswith('_road.png')]
                
                if overlays or defects or roads:  # Only include if there are result images
                    result_entry = {
                        "directory": relative_path,
                        "full_path": os.path.normpath(root),
                        "files": {
                            "overlays": overlays,
                            "defect_masks": defects,
                            "road_masks": roads,
                            "other_files": [f for f in files if not any(f.endswith(suffix) for suffix in ['_overlay.png', '_defects.png', '_road.png'])]
                        },
                        "file_count": len(files),
                        "download_urls": {
                            "overlays": [f"/api/v1/results/{f}" for f in overlays],
                            "defect_masks": [f"/api/v1/results/{f}" for f in defects],
                            "road_masks": [f"/api/v1/results/{f}" for f in roads]
                        }
                    }
                    results.append(result_entry)
        
        return create_success_response(
            message=f"Found {len(results)} result directories",
            data={
                "results": results,
                "output_directory": os.path.normpath(settings.output_dir),
                "total_directories": len(results)
            }
        )
        
    except Exception as e:
        logging.error(f"Error browsing results: {e}")
        return create_error_response(
            message="Failed to browse results",
            error_type="browse_error",
            status_code=500
        )


def cleanup_temp_files(file_paths: List[str]):
    """Background task to clean up temporary files"""
    from ..utils.file_handler import safe_remove_file, safe_remove_directory
    
    for file_path in file_paths:
        if os.path.isfile(file_path):
            safe_remove_file(file_path)
        elif os.path.isdir(file_path):
            safe_remove_directory(file_path)
    
    logging.info(f"Cleaned up {len(file_paths)} temporary files/directories")