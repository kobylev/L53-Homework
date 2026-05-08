# Multi-stage Dockerfile for RL Trading System
# Optimized for both CPU and GPU deployment with multiple service targets

# ============================================================================
# Stage 1: Base Python Environment (CPU)
# ============================================================================
FROM python:3.10-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 2: Dependencies Installation (CPU)
# ============================================================================
FROM base as dependencies-cpu

COPY requirements.txt .

# Install CPU-only PyTorch and other dependencies
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# ============================================================================
# Stage 3: Training Service (CPU)
# ============================================================================
FROM dependencies-cpu as training

# Copy application code
COPY src/ ./src/
COPY generate_evaluation_graph.py .
COPY generate_mock_evaluation.py .

# Create necessary directories
RUN mkdir -p assets/logs

# Default command: run full training pipeline
CMD ["python", "-m", "src.main", "--mode", "full", "--ticker", "MSFT"]

# ============================================================================
# Stage 4: Dashboard Service (CPU)
# ============================================================================
FROM dependencies-cpu as dashboard

# Copy application code
COPY src/ ./src/
COPY assets/ ./assets/

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit dashboard
CMD ["streamlit", "run", "src/dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8501"]

# ============================================================================
# Stage 5: Evaluation Service (CPU)
# ============================================================================
FROM dependencies-cpu as evaluation

# Copy application code
COPY src/ ./src/
COPY generate_evaluation_graph.py .
COPY assets/ ./assets/

RUN mkdir -p assets/logs

# Run evaluation and graph generation
CMD ["python", "generate_evaluation_graph.py"]

# ============================================================================
# GPU-ENABLED VARIANT
# ============================================================================
# Stage 6: GPU Base (CUDA 12.4)
# ============================================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 as gpu-base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_VISIBLE_DEVICES=0

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# ============================================================================
# Stage 7: GPU Dependencies
# ============================================================================
FROM gpu-base as dependencies-gpu

COPY requirements.txt .

# Install CUDA-enabled PyTorch
RUN pip3 install --upgrade pip && \
    pip3 install torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install -r requirements.txt

# ============================================================================
# Stage 8: GPU Training Service
# ============================================================================
FROM dependencies-gpu as training-gpu

# Copy application code
COPY src/ ./src/
COPY generate_evaluation_graph.py .
COPY generate_mock_evaluation.py .

RUN mkdir -p assets/logs

# Verify CUDA availability
RUN python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

CMD ["python", "-m", "src.main", "--mode", "full", "--ticker", "MSFT"]

# ============================================================================
# Build Examples:
# ============================================================================
# CPU Training:    docker build --target training -t trading-rl:cpu-training .
# CPU Dashboard:   docker build --target dashboard -t trading-rl:cpu-dashboard .
# CPU Evaluation:  docker build --target evaluation -t trading-rl:cpu-eval .
# GPU Training:    docker build --target training-gpu -t trading-rl:gpu-training .
#
# Run Examples:
# ============================================================================
# Training:   docker run -v $(pwd)/assets:/app/assets trading-rl:cpu-training
# Dashboard:  docker run -p 8501:8501 -v $(pwd)/assets:/app/assets trading-rl:cpu-dashboard
# GPU Train:  docker run --gpus all -v $(pwd)/assets:/app/assets trading-rl:gpu-training
# ============================================================================
