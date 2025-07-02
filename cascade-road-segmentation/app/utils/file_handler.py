"""
File handling utilities for the FastAPI application
"""

import os
import uuid
import shutil
import aiofiles
from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import UploadFile, HTTPException
from PIL import Image
import logging

try:
    from ..core.config import settings
except ImportError:
    from core.config import settings


def generate_unique_filename(original_filename: str) -> str:
    """Generate a unique filename to avoid conflicts"""
    file_extension = Path(original_filename).suffix.lower()
    unique_id = str(uuid.uuid4())
    return f"{unique_id}{file_extension}"


def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    file_extension = Path(filename).suffix.lower()
    return file_extension in settings.allowed_extensions


def validate_image_file(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that the file is a valid image
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with Image.open(file_path) as img:
            # Try to load the image to verify it's valid
            img.verify()
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


async def save_upload_file(upload_file: UploadFile, destination: str) -> str:
    """
    Save uploaded file to destination
    
    Args:
        upload_file: FastAPI UploadFile object
        destination: Destination file path
        
    Returns:
        Saved file path
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Save file
        async with aiofiles.open(destination, 'wb') as f:
            content = await upload_file.read()
            await f.write(content)
        
        return destination
    
    except Exception as e:
        logging.error(f"Error saving file {destination}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


async def process_upload(upload_file: UploadFile) -> Tuple[str, dict]:
    """
    Process file upload with validation
    
    Args:
        upload_file: FastAPI UploadFile object
        
    Returns:
        Tuple of (file_path, file_info)
    """
    # Validate file size
    if upload_file.size and upload_file.size > settings.max_file_size:
        size_mb = upload_file.size / (1024 * 1024)
        max_mb = settings.max_file_size / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f}MB. Maximum allowed: {max_mb:.1f}MB"
        )
    
    # Validate file extension
    if not is_allowed_file(upload_file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {settings.allowed_extensions}"
        )
    
    # Generate unique filename
    unique_filename = generate_unique_filename(upload_file.filename)
    file_path = os.path.join(settings.upload_dir, unique_filename)
    
    # Save file
    saved_path = await save_upload_file(upload_file, file_path)
    
    # Validate image
    is_valid, error_msg = validate_image_file(saved_path)
    if not is_valid:
        # Clean up invalid file
        try:
            os.remove(saved_path)
        except:
            pass
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Get file info
    file_info = {
        'original_filename': upload_file.filename,
        'saved_filename': unique_filename,
        'saved_path': saved_path,
        'size_bytes': os.path.getsize(saved_path),
        'content_type': upload_file.content_type
    }
    
    return saved_path, file_info


def cleanup_old_files(directory: str, max_age_hours: int = 24):
    """
    Clean up old files from directory
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours before deletion
    """
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Check file age
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        os.remove(file_path)
                        logging.info(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to clean up file {file_path}: {e}")
                    
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")


def get_file_info(file_path: str) -> dict:
    """Get information about a file"""
    try:
        stat = os.stat(file_path)
        with Image.open(file_path) as img:
            width, height = img.size
        
        return {
            'name': os.path.basename(file_path),
            'size_bytes': stat.st_size,
            'width': width,
            'height': height,
            'modified_time': stat.st_mtime
        }
    except Exception as e:
        logging.error(f"Error getting file info for {file_path}: {e}")
        return {}


def create_output_directory(base_name: str) -> str:
    """Create a unique output directory for processing results"""
    unique_id = str(uuid.uuid4())
    output_dir = os.path.join(settings.output_dir, f"{base_name}_{unique_id}")
    os.makedirs(output_dir, exist_ok=True)
    # Normalize path for Windows
    return os.path.normpath(output_dir)


def safe_remove_file(file_path: str) -> bool:
    """Safely remove a file, return True if successful"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return True  # File doesn't exist, consider it successful
    except Exception as e:
        logging.warning(f"Failed to remove file {file_path}: {e}")
        return False


def safe_remove_directory(dir_path: str) -> bool:
    """Safely remove a directory and its contents"""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            return True
        return True  # Directory doesn't exist, consider it successful
    except Exception as e:
        logging.warning(f"Failed to remove directory {dir_path}: {e}")
        return False