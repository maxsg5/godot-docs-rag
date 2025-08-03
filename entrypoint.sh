#!/bin/bash

# Start Ollama in the background
ollama serve &

# Wait for server to be ready by polling `ollama list`
echo "Waiting for Ollama to be ready..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done

# Preload models
echo "Preloading model"
ollama run llama3


# Keep Ollama in foreground
wait
