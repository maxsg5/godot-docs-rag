#!/bin/bash

# One-click setup and execution script for Godot Docs RAG Pipeline
set -e

echo "🚀 Starting Godot Docs RAG Pipeline Setup..."
echo "============================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

if ! command_exists git; then
    echo "❌ Git is required but not installed."
    exit 1
fi

echo "✅ Prerequisites check passed!"

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv .venv
fi

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and add your OpenAI API key!"
    echo "   Required: OPENAI_API_KEY=your_key_here"
    read -p "Press Enter after you've added your API key to .env file..."
fi

# Make download script executable
chmod +x ingest/download_docs.sh

# Step 1: Download Godot documentation
echo "📥 Step 1: Downloading Godot documentation..."
bash ingest/download_docs.sh

# Step 2: Parse documentation with Sphinx
echo "🔧 Step 2: Parsing documentation with Sphinx..."
python ingest/parse_docs.py

# Step 3: Generate Q&A pairs with LLM
echo "🧠 Step 3: Generating Q&A pairs with LLM..."
python chunk/llm_chunking.py

echo ""
echo "🎉 Pipeline completed successfully!"
echo "============================================="
echo "📁 Check the following directories for outputs:"
echo "   • data/raw/godot-docs/     - Raw documentation"
echo "   • data/parsed/html/        - Parsed HTML files"  
echo "   • data/chunks/             - Generated Q&A pairs"
echo ""
echo "💡 Next steps:"
echo "   1. Review the generated Q&A pairs in data/chunks/"
echo "   2. Set up a vector database (FAISS, Chroma, etc.)"
echo "   3. Build a search interface (Streamlit, Gradio, etc.)"
echo ""
echo "✅ All done! Happy RAG building! 🔍"
