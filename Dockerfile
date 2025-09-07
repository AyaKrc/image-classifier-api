# ---------- Stage 1: Build ----------
FROM python:3.10-slim AS builder

WORKDIR /app

# Install system deps (needed for Pillow etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libjpeg-dev zlib1g-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies into a temporary folder
RUN pip install --upgrade pip \
 && pip install --prefix=/install -r requirements.txt

# ---------- Stage 2: Runtime ----------
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project code
COPY app/ app/
COPY artifacts/ artifacts/
COPY requirements.txt .

EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
