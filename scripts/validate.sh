#!/bin/bash

# Simple validation script to test the setup
echo "🧪 Running Godot Docs RAG Pipeline Validation..."
echo "================================================"

# Check if required files exist
echo "📋 Checking project structure..."

required_files=(
    "requirements.txt"
    "Dockerfile" 
    ".env.example"
    "ingest/download_docs.sh"
    "ingest/parse_docs.py"
    "chunk/llm_chunking.py"
    "scripts/setup.sh"
)

missing_files=()

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -eq 0 ]]; then
    echo "✅ All required files present"
else
    echo "❌ Missing files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

# Check if directories exist
echo "📁 Checking directories..."
required_dirs=(
    "data/raw"
    "data/parsed"
    "data/chunks"
)

for dir in "${required_dirs[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "❌ Missing directory: $dir"
        exit 1
    fi
done

echo "✅ All required directories present"

# Check if Python files are valid syntax
echo "🐍 Validating Python syntax..."
python_files=(
    "ingest/parse_docs.py"
    "chunk/llm_chunking.py"
)

for file in "${python_files[@]}"; do
    if ! python3 -m py_compile "$file" 2>/dev/null; then
        echo "❌ Syntax error in $file"
        exit 1
    fi
done

echo "✅ Python syntax validation passed"

# Check if .env file exists
if [[ ! -f ".env" ]]; then
    echo "⚠️  .env file not found (this is expected for first run)"
    echo "   Run: cp .env.example .env"
    echo "   Then add your OpenAI API key"
else
    echo "✅ .env file found"
    
    # Check if OpenAI API key is set
    if grep -q "OPENAI_API_KEY=your_openai_api_key_here" .env; then
        echo "⚠️  Please update your OpenAI API key in .env file"
    else
        echo "✅ OpenAI API key appears to be configured"
    fi
fi

echo ""
echo "🎉 Validation complete!"
echo "🚀 Ready to run: bash scripts/setup.sh"
