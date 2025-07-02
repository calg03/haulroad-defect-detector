# Road Defect Segmentation API

A FastAPI-based web service for detecting road defects using cascade segmentation with SegFormer (road detection) and UNet++ (defect detection).

## Features

- **Cascade Segmentation**: Two-stage approach for accurate defect detection
- **Multiple Architectures**: Supports various neural network architectures
- **Real-time Inference**: Fast prediction on uploaded images
- **Batch Processing**: Process multiple images at once
- **Overlay Generation**: Creates visual overlays showing detected defects
- **RESTful API**: Clean, documented REST endpoints
- **Docker Support**: Containerized deployment ready
- **Production Ready**: Includes monitoring, logging, and error handling

## Architecture

The system uses a two-stage cascade approach:

1. **Road Segmentation** (SegFormer): Identifies road regions in the image
2. **Defect Detection** (UNet++): Detects defects only within road regions

### Detected Defect Classes

- Potholes (blue)
- Cracks (green)  
- Water puddles (light blue)
- Distressed patches (purple)
- Mud (brown)

## Quick Start

### Development Setup

1. **Clone and navigate to the app directory**:
```bash
cd cascade-road-segmentation/app
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment** (optional):
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Run development server**:
```bash
python run_dev.py
```

The API will be available at:
- Main API: http://127.0.0.1:8000
- Documentation: http://127.0.0.1:8000/docs
- Health Check: http://127.0.0.1:8000/health

### Docker Deployment

1. **Build and run with Docker Compose**:
```bash
# Make sure model files are in ./models/ directory
docker-compose up --build
```

2. **Or build Docker image manually**:
```bash
docker build -t road-defect-api .
docker run -p 8000:8000 \
  -v ./models:/app/models:ro \
  -e DEVICE=cpu \
  road-defect-api
```

## API Endpoints

### Health Check
- `GET /health` - Basic health status
- `GET /api/v1/health` - Detailed health information

### Model Information
- `GET /api/v1/model/info` - Get loaded model details

### Inference
- `POST /api/v1/predict/single` - Process single image
- `POST /api/v1/predict/batch` - Process multiple images

### Results
- `GET /api/v1/results/{filename}` - Download result files
- `GET /api/v1/results/{result_id}/overlay` - Get overlay image

## Usage Examples

### Single Image Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/single" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@road_image.jpg" \
  -F "save_outputs=true" \
  -F "overlay_alpha=0.6" \
  -F "confidence_threshold=0.6"
```

### Batch Processing

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "save_outputs=true"
```

### Python Client Example

```python
import requests

# Single image prediction
with open('road_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/predict/single',
        files={'file': f},
        data={
            'save_outputs': True,
            'overlay_alpha': 0.6,
            'confidence_threshold': 0.6
        }
    )

result = response.json()
print(f"Detected {result['data']['total_defect_pixels']} defect pixels")

# Download overlay image
if result['data']['overlay_path']:
    overlay_response = requests.get(
        f"http://localhost:8000/api/v1/results/{result['data']['image_name']}/overlay"
    )
    with open('overlay.png', 'wb') as f:
        f.write(overlay_response.content)
```

## Configuration

### Environment Variables

Key configuration options (see `.env.example` for full list):

- `DEVICE`: Computing device (`cuda`, `cpu`, or auto-detect)
- `MAX_FILE_SIZE`: Maximum upload file size in bytes
- `CONFIDENCE_THRESHOLD`: Default confidence threshold (0.0-1.0)
- `DEBUG`: Enable debug mode and API documentation
- `LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

### Model Files

Place your trained model files in the appropriate directories:

- Road model: Configure `ROAD_MODEL_PATH`
- Defect model: Configure `DEFECT_MODEL_PATH`

Default development paths:
- Road: `/home/cloli/experimentation/cascade-road-segmentation/src/utils/segformer/best_epoch26_besto.pth`
- Defect: `/home/cloli/experimentation/cascade-road-segmentation/src/models/unetplusplus_scse_road_defect_20250626_233608_best.pt`

## Response Format

### Successful Response
```json
{
  "success": true,
  "message": "Inference completed successfully",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "data": {
    "image_name": "road_image",
    "image_shape": [1024, 1536, 3],
    "road_coverage": 0.75,
    "total_defect_pixels": 1250,
    "defect_counts": {
      "pothole": 500,
      "crack": 750,
      "puddle": 0,
      "distressed_patch": 0,
      "mud": 0
    },
    "mean_confidence": 0.85,
    "processing_status": "success",
    "overlay_path": "/tmp/outputs/road_image_overlay.png"
  }
}
```

### Error Response
```json
{
  "success": false,
  "message": "Inference failed: Invalid image format",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "data": {
    "error": "invalid_input",
    "message": "Inference failed: Invalid image format",
    "details": null
  }
}
```

## Monitoring and Logging

- Application logs are written to `app.log` in production
- Health checks available at `/health` and `/api/v1/health`
- Request timing and error logging included
- Optional Prometheus metrics support

## Security Considerations

- File upload validation and size limits
- Trusted host middleware in production
- Non-root user in Docker container
- Input sanitization and path validation
- CORS configuration for cross-origin requests

## Performance Optimization

- GPU acceleration when available
- Efficient tiled inference for large images
- Confidence-based filtering
- Background cleanup of temporary files
- Configurable batch size limits

## Troubleshooting

### Common Issues

1. **Model not found**: Verify model file paths and permissions
2. **CUDA out of memory**: Reduce batch size or use CPU mode
3. **File upload errors**: Check file size limits and formats
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug mode for detailed error messages and API documentation:
```bash
export DEBUG=true
python run_dev.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[Add your license information here]