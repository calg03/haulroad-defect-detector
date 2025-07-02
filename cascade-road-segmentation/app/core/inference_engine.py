"""
Inference engine wrapper for the FastAPI application
"""

import sys
import os
from pathlib import Path
import logging
from typing import Dict, Any, Optional

# Import from local segmentation module
try:
    from ..segmentation.cascade_inference import CascadeInference
except ImportError:
    # Fallback for direct script execution
    try:
        from segmentation.cascade_inference import CascadeInference
    except ImportError as e:
        logging.error(f"Failed to import cascade_inference: {e}")
        raise

from .config import settings


class InferenceEngine:
    """Singleton inference engine for the FastAPI application"""
    
    _instance = None
    _inference = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._inference is None:
            self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the cascade inference engine"""
        try:
            logging.info("Initializing inference engine...")
            
            # Check if model files exist with multiple fallback paths
            road_fallback_paths = [
                settings.road_model_path,
                "C:/TESIS/cascade-road-segmentation/src/models/best_epoch26_besto.pth",
                "../src/models/best_epoch26_besto.pth",
                "/home/cloli/experimentation/cascade-road-segmentation/src/utils/segformer/best_epoch26_besto.pth"
            ]
            
            defect_fallback_paths = [
                settings.defect_model_path,
                "C:/TESIS/cascade-road-segmentation/src/models/unetplusplus_scse_road_defect_20250626_233608_best.pt",
                "../src/models/unetplusplus_scse_road_defect_20250626_233608_best.pt",
                "/home/cloli/experimentation/cascade-road-segmentation/src/models/unetplusplus_scse_road_defect_20250626_233608_best.pt"
            ]
            
            # Find road model
            road_model_found = False
            for road_path in road_fallback_paths:
                if os.path.exists(road_path):
                    settings.road_model_path = road_path
                    logging.info(f"✅ Found road model: {road_path}")
                    road_model_found = True
                    break
            
            if not road_model_found:
                logging.error(f"❌ Road model not found in any of these locations:")
                for path in road_fallback_paths:
                    logging.error(f"   - {path}")
                raise FileNotFoundError(f"Road model not found in any fallback location")
            
            # Find defect model
            defect_model_found = False
            for defect_path in defect_fallback_paths:
                if os.path.exists(defect_path):
                    settings.defect_model_path = defect_path
                    logging.info(f"✅ Found defect model: {defect_path}")
                    defect_model_found = True
                    break
            
            if not defect_model_found:
                logging.error(f"❌ Defect model not found in any of these locations:")
                for path in defect_fallback_paths:
                    logging.error(f"   - {path}")
                raise FileNotFoundError(f"Defect model not found in any fallback location")
            
            # Initialize cascade inference
            self._inference = CascadeInference(
                road_model_path=settings.road_model_path,
                defect_model_path=settings.defect_model_path,
                architecture=settings.architecture,
                device=settings.device
            )
            
            # Update confidence threshold
            self._inference.confidence_threshold = settings.confidence_threshold
            
            logging.info("✅ Inference engine initialized successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize inference engine: {e}")
            raise
    
    def predict(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            output_dir: Optional output directory
            
        Returns:
            Inference results dictionary
        """
        if self._inference is None:
            raise RuntimeError("Inference engine not initialized")
        
        return self._inference.predict(image_path, output_dir)
    
    def predict_batch(self, image_paths: list, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run inference on multiple images
        
        Args:
            image_paths: List of image paths
            output_dir: Optional output directory
            
        Returns:
            Batch inference results dictionary
        """
        if self._inference is None:
            raise RuntimeError("Inference engine not initialized")
        
        return self._inference.predict_batch(image_paths, output_dir)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        if self._inference is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "architecture": self._inference.architecture,
            "device": str(self._inference.device),
            "classes": self._inference.classes,
            "num_classes": self._inference.num_classes,
            "confidence_threshold": self._inference.confidence_threshold,
            "road_model_path": settings.road_model_path,
            "defect_model_path": settings.defect_model_path
        }


# Global inference engine instance
inference_engine = InferenceEngine()