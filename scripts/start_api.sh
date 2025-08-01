#!/bin/bash

# Start FastAPI server
echo "🚀 Starting Godot RAG FastAPI server..."
source venv/bin/activate
export PYTHONPATH=/home/max/llmcapstone/godot-docs-rag:$PYTHONPATH
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
