# 🔍 Godot Docs RAG Assistant

A fully Dockerized Retrieval-Augmented Generation (RAG) system for the Godot game engine documentation. Features an offline indexing pipeline and supports both OpenAI API and local LLM inference via Ollama.

---

## ❓ Problem

Godot's documentation is comprehensive but can be difficult to search semantically or query conversationally. This project transforms the official docs into a RAG system with vector embeddings and intelligent Q&A capabilities.

---

## 🏗️ Architecture

This project follows the standard RAG pattern with separate indexing and retrieval pipelines:

### Indexing (Offline)

```text
HTML Docs → Document Loading → Text Splitting → Embedding → Vector Store
```

### Retrieval & Generation (Runtime)

```text
User Query → Similarity Search → Context Retrieval → LLM Generation → Answer
```

---

## 🎯 Features

- ✅ **Dual LLM Support**: OpenAI API or local Ollama models
- ✅ **Fully Dockerized**: One-command setup with Docker Compose
- ✅ **LangChain Integration**: Advanced document processing and splitting
- ✅ **Semantic Chunking**: Preserves HTML structure (tables, lists, code)
- ✅ **Vector Database**: ChromaDB for efficient similarity search
- ✅ **Q&A Generation**: Automatic question-answer pair creation
- ✅ **Pre-built HTML**: Downloads official Godot HTML docs (updated weekly)
- ✅ **Production Ready**: Scalable container architecture

---

## 📊 Data Source

**Godot Official Documentation** (Updated Weekly)

- **Source**: Pre-built HTML from Godot's nightly builds
- **URL**: [Godot HTML Docs](https://nightly.link/godotengine/godot-docs/workflows/build_offline_docs/master/godot-docs-html-stable.zip)
- **Updates**: Every Monday automatically
- **Advantages**: Clean HTML, always current

---

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

## 🧰 Tech Stack

| Component      | Purpose                             |
|----------------|-------------------------------------|
| `Docker Compose` | Service orchestration              |
| `Ollama`       | Local LLM inference (optional)     |
| `OpenAI API`   | Cloud LLM inference (optional)     |
| `BeautifulSoup4` | HTML parsing and cleaning         |
| `Python 3.11`  | Main application runtime          |
| `HTML2Text`    | Convert HTML to clean text         |

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Git
- Python 3.8+ (for indexing pipeline)

### Setup Process

```bash
# 1. Clone the repository
git clone https://github.com/maxsg5/godot-docs-rag.git
cd godot-docs-rag

# 2. Download and parse HTML documentation
chmod +x legacy/ingest/download_docs.sh
./legacy/ingest/download_docs.sh
python legacy/ingest/parse_docs.py

# 3. Run indexing pipeline (offline processing)
cd legacy/indexing_pipeline
./setup.sh
python indexer.py --input ../../data/parsed/html --output ./output
python qa_generator.py --input ./output/vector_store --output ./output/qa_pairs

# 4. Setup main RAG system
cd ../..
cp .env.example .env
# Edit .env - configure your LLM provider (OpenAI or Ollama)

# 5. Start RAG services
docker-compose up -d
```

### Alternative: Legacy Chunking Mode

The original chunking pipeline is still available:

```bash
# Setup environment
cp .env.example .env
# Edit .env - choose OpenAI or Ollama and configure

# Start services and run legacy pipeline
docker-compose up -d
docker-compose run --rm godot-docs-rag
```

---

## 📁 Project Structure

```
godot-docs-rag/
├── src/                      # Core application code
│   ├── app.py               # FastAPI REST API
│   ├── config.py            # Configuration management
│   ├── rag_system.py        # Main RAG implementation
│   ├── monitoring.py        # Metrics and monitoring
│   ├── data_processor.py    # Document processing pipeline
│   └── main.py              # CLI interface
├── ui/                      # User interfaces
│   └── streamlit_app.py     # Web interface
├── deployment/              # Docker and deployment configs
│   ├── docker-compose.yml   # Multi-service orchestration
│   └── Dockerfile          # Multi-stage build
├── scripts/                 # Setup and utility scripts
│   ├── setup.sh            # Environment setup
│   ├── start_dev.sh         # Development startup
│   ├── start_ui.sh          # UI startup
│   └── validate.sh          # System validation
├── docs/                    # Documentation
│   ├── FEATURES.md          # Feature documentation
│   └── SCORING.md           # Scoring criteria
├── legacy/                  # Legacy components
│   ├── chunk/              # Old chunking logic
│   ├── ingest/             # Data ingestion scripts
│   └── indexing_pipeline/  # Old pipeline code
├── data/                    # Data storage
│   ├── raw/                # Raw documents
│   ├── parsed/             # Processed documents
│   └── chunks/             # Generated chunks
└── requirements.txt         # Dependencies
```

---

## 📦 Indexing Pipeline Outputs

The LangChain-based indexing pipeline generates several key outputs:

### Vector Store (`indexing_pipeline/output/vector_store/`)

```text
├── chroma.sqlite3               # ChromaDB database
├── embeddings.pkl               # Cached embeddings
└── metadata.json                # Document metadata
```

### Q&A Pairs (`indexing_pipeline/output/qa_pairs/`)

```json
[
  {
    "question": "How do I create a RigidBody2D in Godot?",
    "answer": "To create a RigidBody2D, add it as a node in your scene...",
    "category": "physics",
    "difficulty": "basic",
    "code_example": "extends RigidBody2D\n\nfunc _ready():\n    ...",
    "source_chunk": "physics_introduction_chunk_1",
    "confidence": 0.85
  }
]
```

### Document Statistics (`indexing_pipeline/output/stats.json`)

```json
{
  "total_documents": 1359,
  "total_chunks": 4200,
  "total_qa_pairs": 2100,
  "categories": {
    "scripting": 680,
    "rendering": 420,
    "physics": 280,
    "ui_input": 340
  },
  "processing_time": "27 minutes"
}
```

### Legacy Outputs (`data/chunks/`)

The original chunking system still outputs to:

```bash
data/chunks/
├── physics_introduction_qa.json
├── animation_player_qa.json
├── scripting_gdscript_qa.json
└── ...
```

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# LLM Provider Selection
LLM_PROVIDER=ollama                    # Options: "openai" or "ollama"

# OpenAI Configuration (if using OpenAI)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Ollama Configuration (if using Ollama)
OLLAMA_BASE_URL=http://ollama:11434    # Docker service URL
OLLAMA_MODEL=llama3.2:3b               # Model to use

# Pipeline Settings
GODOT_VERSION_BRANCH=4.4
TEMPERATURE=0.3
MAX_TOKENS=2000
```

### LLM Provider Options

#### 🤖 OpenAI API

- **Advantages**: Fast, high-quality output, no local resources
- **Requirements**: API key from OpenAI (costs money)
- **Best for**: Production environments, quick results

#### 🦙 Ollama (Local LLM)

- **Advantages**: Free, private, no external dependencies
- **Requirements**: More disk space (~4GB for models), longer processing time
- **Best for**: Development, privacy-conscious users, cost optimization

---

## 📦 Outputs

All generated Q&A chunks will be saved in:

```bash
data/chunks/
├── physics_introduction_qa.json
├── animation_player_qa.json
├── scripting_gdscript_qa.json
└── ...
```

Each JSON file contains structured Q&A pairs:

```json
[
  {
    "question": "How do I create a RigidBody2D in Godot?",
    "answer": "To create a RigidBody2D, add it as a node in your scene...",
    "category": "physics",
    "difficulty": "beginner",
    "code_example": "extends RigidBody2D\n\nfunc _ready():\n    ...",
    "source_file": "physics_introduction.html"
  }
]
```

---

## 🔧 Development & Management

### Docker Commands

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Run pipeline manually
docker-compose run --rm godot-docs-rag

# Open shell in container
docker-compose run --rm godot-docs-rag bash

# Stop all services
docker-compose down

# Clean up everything (including volumes)
docker-compose down -v
```

### Customizing the Pipeline

#### Indexing Pipeline

1. **Chunk size**: Edit `chunk_size` and `chunk_overlap` in `indexing_pipeline/indexer.py`
2. **Embedding model**: Change `embedding_model` in the indexer configuration
3. **Q&A categories**: Modify `CATEGORY_KEYWORDS` in `indexing_pipeline/qa_generator.py`
4. **Vector store**: Switch between ChromaDB, FAISS, or other LangChain vector stores

#### Legacy Pipeline

1. **Change Godot version**: Edit `GODOT_VERSION_BRANCH` in `.env`
2. **Switch LLM providers**: Change `LLM_PROVIDER` in `.env`
3. **Adjust model parameters**: Modify `TEMPERATURE`, `MAX_TOKENS` in `.env`
4. **Custom prompts**: Edit prompt templates in `chunk/llm_chunking.py`

---

## 🧠 Future Roadmap

- [x] **Vector Database Integration**: ✅ ChromaDB with LangChain
- [x] **Semantic Chunking**: ✅ HTML structure-preserving splitting
- [x] **Q&A Generation**: ✅ Automated question-answer pair creation
- [ ] **Web Interface**: Streamlit or Gradio search UI
- [ ] **Multi-version Support**: Handle Godot 3.x, 4.x simultaneously
- [ ] **Additional LLM Providers**: Claude, Gemini, local Transformers
- [ ] **Batch Processing**: Parallel processing for faster chunking
- [ ] **Quality Metrics**: Automated evaluation of Q&A pair quality
- [ ] **REST API**: API endpoint for querying generated Q&A pairs
- [ ] **Multi-engine Support**: Unity, Unreal Engine documentation

---

## 🐛 Troubleshooting

### Common Issues

#### Indexing Pipeline Issues

1. **ChromaDB creation fails**:
   - Ensure sufficient disk space for embeddings
   - Check Python environment: `cd indexing_pipeline && python --version`
   - Verify dependencies: `pip install -r requirements.txt`

2. **Embedding model download fails**:
   - Check internet connection for HuggingFace model downloads
   - Try alternative embedding model in `indexer.py`
   - Clear model cache: `rm -rf ~/.cache/huggingface/`

3. **Q&A generation produces poor results**:
   - Adjust `CATEGORY_KEYWORDS` in `qa_generator.py`
   - Increase chunk overlap in indexer configuration
   - Review and refine question generation prompts

4. **HTML parsing errors**:
   - Ensure parsed HTML files exist in `data/parsed/html/`
   - Run HTML parser: `python ingest/parse_docs.py`
   - Check for malformed HTML files in the source

#### Docker & Legacy Pipeline Issues

1. **Docker services won't start**:
   - Ensure Docker is running: `docker --version`
   - Check port conflicts: `docker-compose down` then retry
   - Check disk space for Ollama models (~4GB needed)

2. **OpenAI API errors**:
   - Verify your API key is correctly set in `.env`
   - Check your OpenAI account has sufficient credits
   - Ensure `LLM_PROVIDER=openai` is set

3. **Ollama connection issues**:
   - Wait for Ollama service to fully start (check `docker-compose logs ollama`)
   - Verify model is pulled: `docker-compose logs ollama-init`
   - Check Ollama health: `curl http://localhost:11434/api/tags`

4. **No documentation downloaded**:
   - Run download script: `./ingest/download_docs.sh`
   - Check internet connection for GitHub access
   - Verify `data/raw/` directory exists and contains HTML files

5. **Pipeline fails during parsing**:
   - Check `data/parsed/html/` has processed files
   - View detailed logs: `docker-compose logs godot-docs-rag`
   - Ensure sufficient memory for large HTML processing

### Debug Commands

#### Indexing Pipeline Debugging

```bash
# Test the indexing pipeline
cd indexing_pipeline
python test_pipeline.py

# Run indexer with verbose output
python indexer.py --input ../data/parsed/html --output ./output --verbose

# Check embedding dimensions
python -c "from indexer import GodotDocumentationIndexer; print('Embeddings ready')"

# Validate Q&A generation
python qa_generator.py --input ./output/vector_store --output ./output/qa_pairs --limit 10
```

```bash
# Check service status
docker-compose ps

# View real-time logs
docker-compose logs -f [service-name]

# Test Ollama manually
curl http://localhost:11434/api/tags

# Interactive container shell
docker-compose run --rm godot-docs-rag bash
```

---

<!-- ## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--- -->

## 🧑‍💻 Author

Max Schafer - Building developer tools, game tech, and open-source RAG projects.

---

## 🙏 Acknowledgments

- [Godot Engine](https://godotengine.org/) for their excellent documentation
- [Sphinx](https://www.sphinx-doc.org/) for reStructuredText parsing
- [Ollama](https://ollama.com/) for making local LLMs accessible
- [OpenAI](https://openai.com/) for powerful API-based models
- The open-source community for inspiration and tools
