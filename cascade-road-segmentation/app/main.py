"""
FastAPI Application for Road Defect Segmentation
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

try:
    from .core.config import settings
    from .core.inference_engine import inference_engine
    from .api.endpoints import router as api_router
    from .utils.file_handler import cleanup_old_files
except ImportError:
    # Fallback for direct execution
    from core.config import settings
    from core.inference_engine import inference_engine
    from api.endpoints import router as api_router
    from utils.file_handler import cleanup_old_files


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log") if not settings.debug else logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Road Defect Segmentation API")
    logger.info(f"📊 Settings: {settings.app_name} v{settings.app_version}")
    logger.info(f"🔧 Debug mode: {settings.debug}")
    logger.info(f"📱 Device: {settings.device or 'auto-detect'}")
    
    try:
        # Initialize inference engine
        logger.info("🔄 Initializing inference engine...")
        model_info = inference_engine.get_model_info()
        logger.info(f"✅ Model initialized: {model_info}")
        
        # Clean up old files
        logger.info("🧹 Cleaning up old temporary files...")
        cleanup_old_files(settings.upload_dir, settings.cleanup_after_hours)
        cleanup_old_files(settings.output_dir, settings.cleanup_after_hours)
        
        logger.info("✅ Application startup completed")
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Road Defect Segmentation API")
    logger.info("✅ Shutdown completed")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Add trusted host middleware for security
if not settings.debug:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", settings.host]
    )

# Include API routes
app.include_router(
    api_router,
    prefix="/api/v1",
    tags=["inference"]
)

# Mount static files if output directory exists
if os.path.exists(settings.output_dir):
    app.mount(
        "/static", 
        StaticFiles(directory=settings.output_dir), 
        name="static"
    )


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Road Defect Segmentation API",
        "version": settings.app_version,
        "docs": "/docs" if settings.debug else "Documentation disabled in production",
        "health": "/api/v1/health",
        "model_info": "/api/v1/model/info"
    }


@app.get("/health")
async def health():
    """Simple health check"""
    return {"status": "healthy", "service": settings.app_name}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": "internal_server_error",
            "details": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Log request
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


def create_app() -> FastAPI:
    """Factory function to create FastAPI app"""
    return app


if __name__ == "__main__":
    # Development server
    import time
    
    logger.info(f"🚀 Starting development server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1,  # Single worker for development
        log_level=settings.log_level.lower(),
        access_log=True
    )