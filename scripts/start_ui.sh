#!/bin/bash

# Start Streamlit UI
echo "🌐 Starting Godot RAG Streamlit interface..."
source venv/bin/activate
export PYTHONPATH=/home/max/llmcapstone/godot-docs-rag:$PYTHONPATH
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
