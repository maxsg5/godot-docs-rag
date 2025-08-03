# 🔍 Godot Docs RAG Assistant

A fully Dockerized (Everything is in docker-compose) Retrieval-Augmented Generation (RAG) system for the Godot game engine documentation. Features an offline indexing pipeline and supports both OpenAI API and local LLM inference via Ollama with **GPU acceleration support**.

---

## ❓ Problem

Godot's documentation is comprehensive but can be difficult to search semantically or query conversationally. This project transforms the official docs into a RAG system with vector embeddings and intelligent Q&A capabilities.

---

## 🏗️ Architecture

This project follows the standard RAG pattern with Automated ingestion pipeline:

### 🎮 GPU Acceleration Support

The system supports GPU acceleration for faster embedding generation:

- **NVIDIA GPU**: Automatically detected with CUDA support
- **CPU Fallback**: Works on systems without GPU
- **Configurable**: Control GPU usage via config files
- **Docker Integration**: GPU support through Docker Compose

---

## 📊 Data Source

<https://docs.godotengine.org/en/stable/>

## 📚 Document Processing

### Text Splitting Strategy

Uses LangChain's `HTMLSemanticPreservingSplitter`:

- **Structure-Aware**: Preserves tables, lists, code blocks
- **Header-Based**: Splits on H1-H4 with metadata preservation  
- **Semantic Coherence**: Maintains context within chunks
- **Configurable**: 1000 chars default, 200 char overlap

### Q&A Generation

Automatically generates training data:

- **Categories**: scripting, rendering, physics, ui_input, etc.
- **Difficulty Levels**: basic, intermediate, advanced
- **Question Types**: definitional, procedural, troubleshooting
- **Context Preservation**: Links answers to source chunks

---

# Godot Documentation RAG Pipeline

## Overview

This project provides a production-ready pipeline for processing Godot documentation for Retrieval-Augmented Generation (RAG) applications. The pipeline includes:

1. **Download and Extraction**: Automatically downloads and extracts the latest Godot documentation.
2. **Document Loading**: Converts HTML files into LangChain Document objects.
3. **Text Splitting**: Splits documents into smaller chunks using configurable methods.
4. **Embedding**: Generates vector embeddings using Ollama LLM.
5. **Vector Store**: Stores and retrieves documents based on semantic similarity.

## Features

- **Modular Design**: Each stage is cleanly separated for easy maintenance.
- **Configurable**: Centralized configuration via `config.yaml`.
- **Dockerized**: Fully containerized for deployment with `docker-compose`.
- **Production-Ready**: Optimized for scalability and real-world use cases.

## Setup

### Prerequisites

- Docker and Docker Compose installed.
- Python 3.9+.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/maxsg5/godot-docs-rag.git
   cd godot-docs-rag
   ```

2. Build and start the Docker containers:

   ```bash
   # Basic CPU-only setup
   docker compose up --build
   ```

### 🎮 GPU Acceleration (Optional)

For faster embedding generation with NVIDIA GPU support:

1. **Prerequisites**:
   - NVIDIA GPU with CUDA support
   - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
   - Docker with GPU support enabled

2. **GPU Configuration**:

   Configure GPU settings in `config.yaml`:

   ```yaml
   embedding:
     gpu_enabled: true      # Enable GPU acceleration
     gpu_layers: -1         # -1 for auto, 0 for CPU only
     batch_size: 100        # Larger batches for GPU
     show_progress: true    # Detailed progress feedback
   ```

3. **Run with GPU support**:

   ```bash
   # Using the GPU compose file
   docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
   
   # Or set environment variable
   OLLAMA_GPU_LAYERS=-1 docker compose up --build
   ```

4. **Verify GPU usage**:

   ```bash
   # Check if GPU is being used
   docker compose exec ollama nvidia-smi
   ```

## Configuration

All settings are managed via `config.yaml`. Key options include:

- **Document Loading**:
  - `unstructured`: Uses the Unstructured library.
  - `bs4`: Uses BeautifulSoup4.

- **Text Splitting**:
  - `html_header`: Splits based on HTML headers.
  - `html_section`: Splits based on sections.
  - `html_semantic`: Preserves semantic elements.

- **Embedding**:
  - Uses Ollama LLM for vector embeddings.

- **Vector Store**:
  - InMemoryVectorStore for semantic search.

## Usage

### Run the Pipeline

Execute the pipeline with:

```bash
python main.py
```

### Search Documents

Search for similar documents:

```python
from main import search_documents
query = "How to use Godot Engine?"
results = search_documents(query)
for doc in results:
    print(doc.page_content)
```

## Architecture

### Pipeline Stages

1. **Download**: Fetches and extracts documentation.
2. **Load**: Converts HTML files to LangChain Documents.
3. **Split**: Splits documents into chunks.
4. **Embed**: Generates vector embeddings.
5. **Store**: Saves embeddings in a vector store.

### Docker Setup

- **App Container**: Runs the pipeline.
- **Ollama Container**: Hosts the LLM for embeddings.

## Next Steps

- Integrate advanced vector stores like Pinecone or FAISS.
- Add metadata filtering for enhanced search capabilities.
- Implement hybrid search combining semantic and keyword-based methods.

## Troubleshooting

- **Missing Dependencies**: Ensure `requirements.txt` is installed.
- **Docker Issues**: Check Docker logs for errors.
- **Memory Usage**: Adjust chunk sizes in `config.yaml`.

## License

This project is licensed under the MIT License.
