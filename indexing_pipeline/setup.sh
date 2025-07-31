#!/bin/bash

# Godot Documentation RAG - Indexing Pipeline Setup
# =================================================

echo "🚀 Setting up Godot Documentation Indexing Pipeline"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "indexer.py" ]; then
    echo "❌ Error: Please run this script from the indexing_pipeline directory"
    exit 1
fi

# Create Python virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Create output directory
echo "📁 Creating output directory..."
mkdir -p output

# Check if the source HTML data exists
HTML_DIR="../data/parsed/html"
if [ ! -d "$HTML_DIR" ]; then
    echo "⚠️  Warning: HTML source directory not found at $HTML_DIR"
    echo "   Please make sure you've run the HTML processing pipeline first"
    echo "   From the main project directory, run:"
    echo "   python ingest/parse_docs.py"
else
    echo "✅ HTML source directory found"
fi

echo ""
echo "🎉 Setup complete! To run the indexing pipeline:"
echo ""
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run the indexer:"
echo "   python indexer.py"
echo ""
echo "3. Generate Q&A dataset:"
echo "   python qa_generator.py"
echo ""
echo "Output will be saved to the 'output' directory:"
echo "- output/chroma_db/          (Vector database)"
echo "- output/indexing_metadata.json  (Indexing statistics)"
echo "- output/godot_qa_dataset.json   (Q&A pairs)"
echo "- output/qa_statistics.json      (Q&A statistics)"
