# Optimized Setup Guide

## Quick Start with Optimized Configuration

The configuration has been optimized specifically for Godot documentation with the following improvements:

### 🚀 Automated Model Setup

Run the setup script to automatically configure everything:

```bash
# Auto-detect GPU and setup models
./setup_models.sh

# Force GPU usage
./setup_models.sh yes

# Force CPU only
./setup_models.sh no
```

### 📦 What's Been Optimized

#### 1. **Better Embedding Model**

- **Changed to `nomic-embed-text`**: Specialized for code and technical documentation
- **Improved semantic understanding** of API methods, properties, and code examples
- **Better performance** on technical queries

#### 2. **Smarter Text Chunking**

- **Smaller chunks (1500 → 800 chars)**: More precise retrieval
- **Increased overlap (200 → 300 chars)**: Better context preservation
- **Code-aware separators**: Handles API documentation structure better
- **Enhanced element preservation**: Keeps code blocks, tables, and examples intact

#### 3. **Godot-Specific Document Filtering**

- **Prioritizes important content**: Classes, tutorials, getting started guides
- **Filters out noise**: Navigation, search pages, static assets
- **Content quality rules**: Minimum/maximum length constraints

#### 4. **Advanced Retrieval**

- **Hybrid search**: Combines semantic similarity with keyword matching
- **Reranking**: Improves result quality
- **Context enhancement**: Includes surrounding chunks for better context

#### 5. **Performance Improvements**

- **Persistent Chroma vector store**: Faster subsequent queries
- **GPU optimization**: Better memory management and parallel processing
- **Health checks**: Ensures services are ready before processing

### 🔧 Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Start Ollama with GPU support
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d ollama

# Pull the optimized embedding model
docker exec ollama ollama pull nomic-embed-text

# Pull the language model
docker exec ollama ollama pull llama3

# Run the pipeline
python godot_rag_pipeline.py
```

### 📊 Expected Improvements

With these optimizations, you should see:

1. **Better retrieval accuracy** for API-specific queries
2. **More relevant code examples** in responses
3. **Improved understanding** of Godot-specific terminology
4. **Faster embedding generation** with the specialized model
5. **Better context preservation** in chunked content

### 🐛 Troubleshooting

If you encounter issues:

```bash
# Check if models are loaded
docker exec ollama ollama list

# Check Ollama logs
docker logs ollama

# Restart services if needed
docker compose down && docker compose up -d
```

### ⚙️ Configuration Details

The key optimizations in `config.yaml`:

- **Embedding model**: `nomic-embed-text` (specialized for documentation)
- **Vector store**: `chroma` (persistent storage)
- **Chunk size**: 800 characters (better granularity)
- **Document loading**: `bs4` (better HTML parsing)
- **Content filtering**: Godot-specific patterns and rules
