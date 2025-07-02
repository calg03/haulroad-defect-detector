#!/bin/bash
# Download model weights from S3 on container startup

set -e

# Configuration from environment variables
S3_BUCKET=${S3_MODELS_BUCKET:-"road-defects-bkt"}
S3_PREFIX=${S3_MODELS_PREFIX:-"road-defects-bkt/"}
MODELS_DIR="/app/models"

echo "🔄 Downloading models from S3..."
echo "  Bucket: s3://$S3_BUCKET/"
echo "  Local directory: $MODELS_DIR"

# Create models directory
mkdir -p $MODELS_DIR

# Download road segmentation model
echo "📥 Downloading road segmentation model..."
aws s3 cp s3://$S3_BUCKET/road_segmentation.pth $MODELS_DIR/road_segmentation.pth

# Download defect segmentation model  
echo "📥 Downloading defect segmentation model..."
aws s3 cp s3://$S3_BUCKET/defect_segmentation.pt $MODELS_DIR/defect_segmentation.pt

# Verify downloads
if [ -f "$MODELS_DIR/road_segmentation.pth" ] && [ -f "$MODELS_DIR/defect_segmentation.pt" ]; then
    echo "✅ Models downloaded successfully!"
    echo "  Road model: $(ls -lh $MODELS_DIR/road_segmentation.pth | awk '{print $5}')"
    echo "  Defect model: $(ls -lh $MODELS_DIR/defect_segmentation.pt | awk '{print $5}')"
else
    echo "❌ Model download failed!"
    exit 1
fi