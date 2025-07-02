#!/usr/bin/env python3
"""
Development server runner for Road Defect Segmentation API
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Configure logging for development
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'torch',
        'transformers',
        'segmentation_models_pytorch',
        'cv2',
        'PIL',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        logger.error("Please install them using: pip install -r requirements.txt")
        return False
    
    return True


def check_model_files():
    """Check if model files exist"""
    # Development model paths
    dev_models = [
        "C:/TESIS/cascade-road-segmentation/src/models/best_epoch26_besto.pth",
        "C:/TESIS/cascade-road-segmentation/src/models/unetplusplus_scse_road_defect_20250626_233608_best.pt"]
    
    missing_models = []
    for model_path in dev_models:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        logger.warning(f"Some model files not found: {missing_models}")
        logger.warning("The API will try to initialize but may fail if models are not accessible")
    else:
        logger.info("✅ All development model files found")
    
    return True


def setup_environment():
    """Setup development environment"""
    # Create necessary directories
    os.makedirs("/tmp/uploads", exist_ok=True)
    os.makedirs("/tmp/outputs", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Set development environment variables
    os.environ.setdefault("DEBUG", "true")
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("HOST", "127.0.0.1")
    os.environ.setdefault("PORT", "8000")
    
    logger.info("✅ Development environment setup complete")


def main():
    """Main entry point for development server"""
    logger.info("🚀 Starting Road Defect Segmentation API - Development Mode")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model files
    check_model_files()
    
    # Setup environment
    setup_environment()
    
    # Import and run the application
    try:
        import uvicorn
        
        # Add app directory to Python path to fix relative imports
        app_dir = Path(__file__).parent
        logger.info(f"🔧 Adding to Python path: {app_dir}")
        if str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
        
        logger.info("📦 Python path:")
        for i, path in enumerate(sys.path[:5]):  # Show first 5 paths
            logger.info(f"   [{i}] {path}")
        
        logger.info("🔄 Attempting to import main application...")
        
        try:
            from main import app
            logger.info("✅ Successfully imported main application")
        except ImportError as import_err:
            logger.error(f"❌ Failed to import main: {import_err}")
            logger.info("🔍 Trying to diagnose the issue...")
            
            # Try to import each module individually
            modules_to_test = [
                ("core.config", "settings"),
                ("segmentation.config", "CLASSES"),
                ("segmentation.cascade_inference", "CascadeInference"),
                ("core.inference_engine", "inference_engine")
            ]
            
            for module_name, attr_name in modules_to_test:
                try:
                    module = __import__(module_name, fromlist=[attr_name])
                    getattr(module, attr_name)
                    logger.info(f"✅ {module_name}.{attr_name} - OK")
                except Exception as e:
                    logger.error(f"❌ {module_name}.{attr_name} - FAILED: {e}")
            
            raise import_err
        
        logger.info("🌐 Starting development server...")
        logger.info("📊 API Documentation: http://127.0.0.1:8000/docs")
        logger.info("🔍 Health Check: http://127.0.0.1:8000/health")
        logger.info("🤖 Model Info: http://127.0.0.1:8000/api/v1/model/info")
        
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8000,
            reload=False,  # Disable reload to avoid the warning
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        logger.error(f"❌ Import Error Details: {e}")
        logger.error(f"❌ Error Type: {type(e).__name__}")
        logger.error("💡 This usually means:")
        logger.error("   1. Missing dependencies (run: pip install -r requirements.txt)")
        logger.error("   2. Python path issues (try running from parent directory)")
        logger.error("   3. Module structure problems")
        logger.error("")
        logger.error("🔧 Try these solutions:")
        logger.error("   Solution 1: cd .. && python -m app.run_dev")
        logger.error("   Solution 2: uvicorn main:app --host 127.0.0.1 --port 8000")
        logger.error("   Solution 3: Check if all files exist in segmentation/ directory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.error(f"❌ Error Type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()