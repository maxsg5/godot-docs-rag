# Multi-stage Docker build for Godot RAG System

# Base Python image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY ui/streamlit_app.py ui/streamlit_app.py
COPY scripts/ scripts/
COPY legacy/ingest/ legacy/ingest/

# Make scripts executable
RUN chmod +x scripts/*.sh legacy/ingest/*.sh || true

# Create necessary directories
RUN mkdir -p data/raw data/parsed data/chunks data/processed data/metrics logs config

# Set Python path
ENV PYTHONPATH=/app

# Development stage
FROM base as development

# Install additional development dependencies
RUN pip install jupyter notebook ipywidgets pytest pytest-asyncio black flake8 mypy

# Expose development ports
EXPOSE 8000 8501 8888

# Development command
CMD ["python", "-c", "print('Development container ready. Use docker exec to run commands.')"]

# Production API stage
FROM base as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run FastAPI application
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Streamlit UI stage
FROM base as ui

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Run Streamlit application
CMD ["streamlit", "run", "ui/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Data processor stage
FROM base as processor

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Default command runs the main pipeline
CMD ["python", "-m", "src.main"]
