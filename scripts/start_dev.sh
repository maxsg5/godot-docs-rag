#!/bin/bash

echo "🎮 Starting Godot RAG System in development mode..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Start Docker services in background
echo "🐳 Starting Docker services..."
if command -v docker-compose &> /dev/null; then
    docker-compose -f deployment/docker-compose.yml up -d ollama qdrant redis
    echo "✅ Docker services starting in background"
else
    echo "⚠️ Docker Compose not found. Services may need to be started manually."
fi

# Wait for services to be ready
echo "⏱️ Waiting for services to initialize..."
sleep 10

# Start API in background
echo "🚀 Starting FastAPI server..."
./scripts/start_api.sh &
API_PID=$!

# Start Streamlit
echo "🌐 Starting Streamlit UI..."
./scripts/start_ui.sh &
UI_PID=$!

echo ""
echo "🎉 Godot RAG System is starting up!"
echo ""
echo "📍 Access points:"
echo "   • Streamlit UI: http://localhost:8501"
echo "   • FastAPI API: http://localhost:8000"
echo "   • API Docs: http://localhost:8000/docs"
echo "   • Ollama: http://localhost:11434"
echo "   • Qdrant: http://localhost:6333"
echo ""
echo "💡 Tips:"
echo "   • Wait 30-60 seconds for all services to fully initialize"
echo "   • Check service health at the /health endpoints"
echo "   • Use Ctrl+C to stop all services gracefully"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down Godot RAG System..."
    kill $API_PID $UI_PID 2>/dev/null
    if command -v docker-compose &> /dev/null; then
        echo "🐳 Stopping Docker services..."
        docker-compose -f deployment/docker-compose.yml down
    fi
    echo "✅ Shutdown complete"
    exit 0
}

# Trap Ctrl+C and cleanup
trap cleanup SIGINT SIGTERM

echo "🔄 System running... Press Ctrl+C to stop all services"
wait
