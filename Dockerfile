# ---------- Stage 1: Builder ----------
FROM python:3.10-slim-bullseye AS builder

WORKDIR /app

# Install system deps (needed for Pillow etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libjpeg-dev zlib1g-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install deps into /install (CPU-only wheels, no cache)
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.10-slim-bullseye

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only necessary project files
COPY app/ app/
COPY artifacts/labels.json artifacts/labels.json
COPY requirements.txt .

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Run FastAPI app on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
