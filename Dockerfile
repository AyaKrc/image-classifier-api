# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Hugging Face
ENV HF_HOME=/tmp/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONUNBUFFERED=1

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/huggingface /tmp/huggingface_cache && \
    chmod -R 777 /tmp/huggingface /tmp/huggingface_cache

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY artifacts/ artifacts/

# Create a non-root user (optional but recommended)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app /tmp/huggingface /tmp/huggingface_cache

# Switch to non-root user
USER user

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]