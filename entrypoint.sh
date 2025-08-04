#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for server to be ready by polling `ollama list`
echo "Waiting for Ollama to be ready..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

# Preload models optimized for RAG
echo "Preloading embedding model for documentation RAG..."
ollama pull nomic-embed-text

echo "Preloading language model..."
ollama pull llama3

echo "All models loaded successfully!"

# Keep Ollama in foreground
wait
