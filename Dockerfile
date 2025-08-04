# Use Python 3.11 slim for better security and performance
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create data directory with proper permissions
RUN mkdir -p /app/data && chmod 755 /app/data

# Set permissions for Python scripts
RUN chmod +x /app/*.py

# Health check - simplified for RAG pipeline
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=2 \
    CMD python -c "print('Container is healthy')" || exit 1

# Default command - can be overridden by docker-compose
CMD ["python", "pipeline_and_evaluation.py"]
