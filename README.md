# 🔍 Godot Docs RAG Assistant 🎯 Features

- ✅ **Dual LLM Support**: OpenAI API or local Ollama models
- ✅ **Fully Dockerized**: One-command setup with Docker Compose
- ✅ **Sphinx Integration**: Parses `.rst` files to HTML automatically
- ✅ **Smart Q&A Generation**: Creates practical developer-focused Q&A pairs
- ✅ **Production Ready**: Scalable container architecture
- ✅ **No Manual Setup**: Everything automated in Docker containers Assistant

A fully Dockerized Retrieval-Augmented Generation (RAG) pipeline that transforms the Godot game engine documentation into structured Q&A chunks. Supports both OpenAI API and local LLM inference via Ollama.

---

## ❓ Problem

Godot's documentation is comprehensive but can be difficult to search semantically or query conversationally. This project aims to transform the official docs into a format suitable for intelligent Q&A retrieval using modern language models.

---

## 🎯 Features

- ✅ Parses `.rst` files using Sphinx
- ✅ Converts docs to HTML and extracts Q&A pairs using an LLM
- ✅ Prepares data for vector search / RAG pipeline
- ✅ Fully automatable with Docker and shell script
- ✅ Easily extendable to support other game engine docs

---

## 🧰 Tech Stack

| Component      | Purpose                             |
|----------------|-------------------------------------|
| `Docker Compose` | Service orchestration              |
| `Ollama`       | Local LLM inference (optional)     |
| `OpenAI API`   | Cloud LLM inference (optional)     |
| `Sphinx`       | Convert `.rst` to HTML              |
| `BeautifulSoup4` | HTML parsing and cleaning         |
| `Python 3.11`  | Main application runtime          |

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Git

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/maxsg5/godot-docs-rag.git
cd godot-docs-rag

# Run the automated setup
chmod +x scripts/docker-setup.sh
./scripts/docker-setup.sh
```

The setup script will:

1. **Configure your LLM provider** (OpenAI or Ollama)
2. **Start all Docker services** (including Ollama if selected)
3. **Download required models** automatically
4. **Run the complete pipeline** (download → parse → generate Q&A)

### Alternative: Step-by-Step Docker Setup

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env - choose OpenAI or Ollama and configure

# 2. Start services
docker-compose up -d

# 3. Run pipeline
docker-compose run --rm godot-docs-rag
```

---

## 📁 Project Structure

```text
godot-docs-rag/
├── src/
│   └── main.py                  # Main pipeline orchestrator
├── ingest/
│   ├── download_docs.py         # Git clone of godot-docs repo
│   └── parse_docs.py            # Sphinx reStructuredText to HTML parser
├── chunk/
│   ├── llm_provider.py          # LLM provider abstraction (OpenAI/Ollama)
│   └── chunker.py               # Document chunker and Q&A generator
├── data/
│   ├── raw/                     # Raw .rst files (auto-generated)
│   ├── parsed/                  # Converted HTML files (auto-generated)
│   └── chunks/                  # Generated Q&A pairs (auto-generated)
├── scripts/
│   └── docker-setup.sh          # One-command Docker setup
├── docker-compose.yml           # Multi-service Docker configuration
├── Dockerfile                   # Main application container
└── .env.example                 # Environment configuration template
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

1. **Change Godot version**: Edit `GODOT_VERSION_BRANCH` in `.env`
2. **Switch LLM providers**: Change `LLM_PROVIDER` in `.env`
3. **Adjust model parameters**: Modify `TEMPERATURE`, `MAX_TOKENS` in `.env`
4. **Custom prompts**: Edit prompt templates in `chunk/chunker.py`

---

## 🧠 Future Roadmap

- [ ] **Vector Database Integration**: FAISS, Chroma, or Weaviate support
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
   - Check internet connection for GitHub access
   - Verify git is installed in the container
   - Check `data/raw/godot-docs/` directory exists

5. **Pipeline fails during parsing**:
   - Ensure Sphinx can find `conf.py` in godot-docs
   - Check `data/parsed/html/` has generated files
   - View detailed logs: `docker-compose logs godot-docs-rag`

### Debug Commands

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
