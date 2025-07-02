# Road Defect Segmentation API Dockerfile - ECS Optimized
FROM python:3.11.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and ML libraries
RUN apt-get update && apt-get install -y \
    # Core dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    # Network and utilities for health checks
    curl \
    wget \
    unzip \
    # AWS CLI for S3 model downloads
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws \
    # Clean up to reduce image size
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application directories with proper permissions
RUN mkdir -p /app/models /app/uploads /app/outputs /app/logs /app/tmp /app/.cache/huggingface

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations for ECS
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --compile -r requirements.txt \
    && pip cache purge

# Copy application code
COPY . .

# Copy model download script and startup script, make executable
COPY download_models.sh start.sh /app/
RUN chmod +x /app/download_models.sh /app/start.sh

# Create non-root user for security (ECS best practice)
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app \
    && chmod -R 755 /app/.cache

# Switch to non-root user
USER appuser

# Set environment variables for production
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# Expose port
EXPOSE 8000

# Health check optimized for ECS
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ECS-optimized startup command with S3 model download
CMD ["/app/start.sh"]