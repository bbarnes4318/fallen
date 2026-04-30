# Use the official Python 3.10 slim image as the base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    DEEPFACE_HOME=/app

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

# Pre-download DeepFace models to /app/.deepface/weights so they are baked into the image
# This prevents Cloud Run from timing out during container startup
RUN mkdir -p /app/.deepface/weights && \
    curl -L -o /app/.deepface/weights/arcface_weights.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5 && \
    curl -L -o /app/.deepface/weights/facenet512_weights.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5 && \
    curl -L -o /app/.deepface/weights/retinaface.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5 && \
    curl -L -o /app/.deepface/weights/age_model_weights.h5 https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5

# Copy the rest of the application code
COPY . .

# Download missing MiniFASNetV2.onnx for anti-spoofing
RUN mkdir -p models && \
    curl -L -o models/MiniFASNetV2.onnx https://github.com/facenox/face-antispoof-onnx/raw/main/models/MiniFASNetV2.onnx

# Expose the port Cloud Run expects
EXPOSE 8080

# Command to run the FastAPI application using Uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
