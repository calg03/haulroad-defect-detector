#!/bin/bash
# Startup script for ECS container with S3 model download

set -e

echo "🚀 Starting Road Defect Segmentation API..."

# Download models from S3 if they don't exist locally
if [ ! -f "/app/models/road_segmentation.pth" ] || [ ! -f "/app/models/defect_segmentation.pt" ]; then
    echo "📥 Models not found locally, downloading from S3..."
    /app/download_models.sh
else
    echo "✅ Models already exist locally, skipping download"
fi

# Update config paths to use downloaded models
export ROAD_MODEL_PATH="/app/models/road_segmentation.pth"
export DEFECT_MODEL_PATH="/app/models/defect_segmentation.pt"

echo "🌐 Starting FastAPI server..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --access-log --log-level info --proxy-headers