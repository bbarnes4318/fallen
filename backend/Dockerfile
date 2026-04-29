# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV, dlib, and general compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
# We assume a requirements.txt will be created in the next steps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download DeepFace models to ~/.deepface/weights so they are baked into the image
# This prevents Cloud Run from timing out during container startup
RUN TF_USE_LEGACY_KERAS=1 python -c "from deepface import DeepFace; DeepFace.build_model('ArcFace'); DeepFace.build_model('Facenet512')"

# Copy the rest of the application code
COPY . .

# Expose the port Cloud Run expects
EXPOSE 8080

# Command to run the FastAPI application using Uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
