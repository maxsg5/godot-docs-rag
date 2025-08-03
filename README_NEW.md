# Godot Documentation RAG Pipeline

A production-ready, containerized pipeline for processing Godot Engine documentation for Retrieval-Augmented Generation (RAG) applications.

## 🏗️ Architecture

### Pipeline Overview

```
📥 Download → 📄 Load → ✂️ Split → 🧠 Embed → 🗄️ Store → 🔍 Search
```

1. **Download & Extract**: Fetches latest Godot documentation (HTML)
2. **Document Loading**: Converts HTML to LangChain Documents
3. **Text Splitting**: Chunks documents using configurable methods
4. **Embedding**: Generates vectors using Ollama LLM
5. **Vector Storage**: Stores embeddings for semantic search
6. **Search Interface**: Retrieves relevant documents by similarity

### Components

- **Pipeline Engine** (`pipeline.py`): Core orchestration logic
- **Configuration** (`config.yaml`): Centralized settings management
- **Docker Setup**: Containerized deployment with Ollama LLM
- **Legacy Scripts**: Individual components for granular control

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- 4GB+ RAM (for Ollama LLM)
- Internet connection (for initial download)

### Installation & Deployment

1. **Clone Repository**:

   ```bash
   git clone https://github.com/maxsg5/godot-docs-rag.git
   cd godot-docs-rag
   ```

2. **Start Services**:

   ```bash
   docker-compose up --build
   ```

   This will:
   - Build the Python application container
   - Start Ollama LLM service
   - Download and setup the `llama3` model
   - Run the complete RAG pipeline

3. **Monitor Progress**:

   ```bash
   docker-compose logs -f app
   ```

## ⚙️ Configuration

All settings are managed through `config.yaml`:

### Document Loading

```yaml
document_loading:
  method: "unstructured"  # Options: "unstructured", "bs4"
```

### Embedding Model

```yaml
embedding:
  model: "llama3"
  base_url: "http://ollama:11434"
```

### Text Splitting Methods

```yaml
text_splitting:
  method: "html_semantic"  # Options: "html_header", "html_section", "html_semantic"
```

**Available Methods**:

- `html_header`: Splits on HTML headers (h1-h6) - Fast, structure-based
- `html_section`: Splits on HTML sections - Semantic organization  
- `html_semantic`: Preserves tables, lists, code blocks - Best for technical docs

### Secondary Splitting

```yaml
secondary_splitting:
  enabled: true
  chunk_size: 1000
  chunk_overlap: 100
```

## 📊 Performance Metrics

| Metric | Value |
|--------|--------|
| Documents Processed | ~1,490 HTML files |
| Total Size | 361MB |
| Processing Time | ~8-10 minutes |
| Final Chunks | ~15,000 chunks |
| Memory Usage | ~2GB peak |
| Storage | ~500MB processed |

## 🔧 Usage

### Programmatic Access

```python
from pipeline import GodotRAGPipeline

# Initialize pipeline
pipeline = GodotRAGPipeline()

# Run complete pipeline
pipeline.run_pipeline()

# Search documents
results = pipeline.search_documents("How to create a scene?", k=5)
for doc in results:
    print(f"Content: {doc.page_content[:200]}...")
    print(f"Source: {doc.metadata.get('source')}")
```

### Command Line

```bash
# Run complete pipeline
python main.py

# Or use the full pipeline class
python pipeline.py
```

### Docker Environment

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs app

# Execute commands in container
docker-compose exec app python -c "
from pipeline import GodotRAGPipeline
pipeline = GodotRAGPipeline()
results = pipeline.search_documents('Godot scripting')
print(f'Found {len(results)} results')
"
```

## 🗂️ Project Structure

```
godot-docs-rag/
├── 🐳 Docker Configuration
│   ├── Dockerfile                 # Application container
│   ├── docker-compose.yml         # Multi-service orchestration
│   └── requirements.txt           # Python dependencies
│
├── 🧠 Core Pipeline
│   ├── pipeline.py                # Main pipeline orchestration
│   ├── main.py                    # Entry point
│   └── config.yaml                # Configuration management
│
├── 📚 Legacy Components
│   ├── LoadData.py                # Document downloader
│   ├── LoadDataForLangChain.py    # HTML to LangChain converter
│   ├── SplitTextForRAG.py         # Text splitting logic
│   └── manage_config.py           # Configuration utility
│
├── 📁 Data Structure
│   └── data/
│       ├── raw/                   # Downloaded HTML files
│       ├── processed/             # LangChain documents
│       └── chunked/               # Split text chunks
│
└── 📖 Documentation
    ├── README.md                  # This file
    └── PIPELINE_GUIDE.md          # Detailed guide
```

## 🔄 Development Workflow

### Local Development

1. **Setup Python Environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Start Ollama Locally**:

   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull model
   ollama pull llama3
   
   # Start server
   ollama serve
   ```

3. **Run Pipeline**:

   ```bash
   python pipeline.py
   ```

### Configuration Changes

```bash
# View current config
python manage_config.py --show

# Apply preset
python manage_config.py --preset semantic

# Custom changes
python manage_config.py --split-method html_header --chunk-size 2000
```

## 🚀 Production Deployment

### Scaling Considerations

1. **Vector Store**: Replace InMemoryVectorStore with persistent solutions:
   - **Chroma**: Local persistent storage
   - **Pinecone**: Cloud-managed vector database
   - **Weaviate**: Self-hosted with advanced features

2. **Embedding Model**: Consider alternatives:
   - **OpenAI Embeddings**: Higher quality, API-based
   - **Sentence Transformers**: Local models, various sizes
   - **Cohere**: Specialized for retrieval

3. **Compute Resources**:
   - **CPU**: 4+ cores recommended
   - **RAM**: 8GB+ for large document sets
   - **Storage**: SSD for better I/O performance

### Environment Variables

```bash
# Ollama configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODELS=/path/to/models

# Pipeline configuration
CONFIG_PATH=/app/config.yaml
PIPELINE_OUTPUT_DIR=/app/data/processed
```

## 🔍 Advanced Features

### Metadata Filtering

```python
# Search with metadata filters
results = pipeline.vector_store.similarity_search(
    "Godot scripting",
    k=10,
    filter={"source": "tutorials"}
)
```

### Hybrid Search

```python
# Combine semantic and keyword search
from langchain_community.retrievers import BM25Retriever

# Create BM25 retriever for keyword search
bm25 = BM25Retriever.from_documents(documents)

# Combine with vector search
hybrid_results = bm25.get_relevant_documents("Godot") + \
                vector_store.similarity_search("Godot")
```

### Custom Processing

```python
# Custom document processor
class CustomGodotProcessor(GodotRAGPipeline):
    def preprocess_document(self, doc):
        # Custom preprocessing logic
        return doc
    
    def postprocess_splits(self, splits):
        # Custom postprocessing logic
        return splits
```

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Ollama connection failed | Wait 30s for service startup |
| Out of memory | Reduce chunk_size in config |
| Slow processing | Use smaller embedding model |
| Missing model | Check model name in config |

### Debug Commands

```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Check container logs
docker-compose logs ollama
docker-compose logs app

# Test configuration
python manage_config.py --show

# Validate pipeline
python -c "from pipeline import GodotRAGPipeline; p = GodotRAGPipeline(); print('Config loaded:', bool(p.config))"
```

## 📈 Monitoring & Metrics

### Key Metrics to Track

- **Processing Speed**: Documents/second
- **Memory Usage**: Peak RAM consumption
- **Search Latency**: Query response time
- **Accuracy**: Retrieval relevance scores

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
pipeline = GodotRAGPipeline()
pipeline.run_pipeline()
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Godot Engine**: For excellent documentation
- **LangChain**: For RAG framework
- **Ollama**: For local LLM hosting
- **Community**: For feedback and contributions

---

**Ready for production RAG applications!** 🚀
