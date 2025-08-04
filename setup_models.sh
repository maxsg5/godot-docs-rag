#!/bin/bash

# Godot RAG Pipeline - Model Setup Script
echo "🚀 Setting up Godot Documentation RAG Pipeline..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Determine if we should use GPU
USE_GPU=${1:-"auto"}

if [ "$USE_GPU" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
        echo "🎮 NVIDIA GPU detected, using GPU acceleration"
        USE_GPU="yes"
    else
        echo "💻 No GPU detected, using CPU only"
        USE_GPU="no"
    fi
fi

# Start services based on GPU availability
if [ "$USE_GPU" = "yes" ]; then
    echo "🚀 Starting services with GPU support..."
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d ollama
else
    echo "🚀 Starting services with CPU only..."
    docker compose up -d ollama
fi

# Wait for Ollama to be ready
echo "⏳ Waiting for Ollama to be ready..."
until docker exec ollama ollama list > /dev/null 2>&1; do
    echo "   Still waiting..."
    sleep 5
done

echo "✅ Ollama is ready!"

# Check if models are already downloaded
echo "📦 Checking for required models..."

if docker exec ollama ollama list | grep -q "nomic-embed-text"; then
    echo "✅ nomic-embed-text model already available"
else
    echo "📥 Downloading nomic-embed-text model (optimized for documentation)..."
    docker exec ollama ollama pull nomic-embed-text
fi

if docker exec ollama ollama list | grep -q "llama3"; then
    echo "✅ llama3 model already available"
else
    echo "📥 Downloading llama3 model..."
    docker exec ollama ollama pull llama3
fi

echo "🎉 Setup complete! Available models:"
docker exec ollama ollama list

echo ""
echo "📚 Next steps:"
echo "1. Run the RAG pipeline: python godot_rag_pipeline.py"
echo "2. Or start the full app service: docker compose up app"
echo ""
echo "🔧 Configuration has been optimized for:"
echo "   - nomic-embed-text for better code/documentation embeddings"
echo "   - Smaller chunk sizes for precise retrieval"
echo "   - Chroma vector store for persistent storage"
echo "   - Document filtering for Godot-specific content"
echo "   - Fixed health check using ollama CLI instead of curl"
