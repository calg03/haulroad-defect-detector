"""
FastAPI application configuration
"""

import os
from pathlib import Path
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    app_name: str = "Road Defect Segmentation API"
    app_version: str = "1.0.0"
    app_description: str = "Cascade road and defect segmentation using SegFormer + UNet++"
    debug: bool = False
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # Model Configuration - Windows development paths
    road_model_path: str = "C:/TESIS/cascade-road-segmentation/src/models/best_epoch26_besto.pth"
    defect_model_path: str = "C:/TESIS/cascade-road-segmentation/src/models/unetplusplus_scse_road_defect_20250626_233608_best.pt"
    architecture: str = "unetplusplus_scse"
    device: Optional[str] = None  # Auto-detect if None
    
    # File Upload Settings
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    upload_dir: str = "C:/TESIS/cascade-road-segmentation/app/uploads"
    output_dir: str = "C:/TESIS/cascade-road-segmentation/app/outputs"
    
    # Processing Settings
    confidence_threshold: float = 0.6
    overlay_alpha: float = 0.6
    cleanup_after_hours: int = 24  # Clean up temp files after 24 hours
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS Settings
    allowed_origins: list = ["*"]  # Configure appropriately for production
    allowed_methods: list = ["GET", "POST"]
    allowed_headers: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Create necessary directories
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.output_dir).mkdir(parents=True, exist_ok=True)