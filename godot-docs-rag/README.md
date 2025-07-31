# 🔍 Godot Docs RAG Assistant

A Retrieval-Augmented Generation (RAG) pipeline that transforms the Godot game engine documentation into structured Q&A chunks using Sphinx parsing and LLM-based chunking.

---

## ❓ Problem

Godot's documentation is comprehensive but can be difficult to search semantically or query conversationally. This project aims to transform the official docs into a format suitable for intelligent Q&A retrieval using modern language models.

---

## 🎯 Features

- ✅ Parses `.rst` files using Sphinx
- ✅ Converts docs to HTML and extracts Q&A pairs using GPT
- ✅ Prepares data for vector search / RAG pipeline
- ✅ Fully automatable with Docker and shell script
- ✅ Easily extendable to support other game engine docs

---

## 🧰 Tech Stack

| Package        | Purpose                             |
|----------------|-------------------------------------|
| `sphinx`       | Convert `.rst` to HTML              |
| `openai`       | Q&A generation via GPT models       |
| `html2text`    | Convert HTML to Markdown            |
| `beautifulsoup4` | HTML parsing and cleaning         |
| `python-dotenv` | Environment variable management    |
| `Docker`       | Run project in a portable container |
| `bash`         | Shell script for automation         |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- OpenAI API key

### 1. With Docker (Recommended)

```bash
# Clone this repository
git clone <your-repo-url>
cd godot-docs-rag

# Copy environment template and add your OpenAI API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your_key_here

# Build and run with Docker
docker build -t godot-docs-rag .
docker run --rm -it -v $(pwd)/.env:/app/.env godot-docs-rag
```

### 2. Without Docker

```bash
# Clone this repository
git clone <your-repo-url>
cd godot-docs-rag

# Run the setup script (handles everything)
bash scripts/setup.sh
```

### 3. Manual Step-by-Step

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your OpenAI API key

# Download Godot docs
bash ingest/download_docs.sh

# Parse with Sphinx
python ingest/parse_docs.py

# Generate Q&A pairs
python chunk/llm_chunking.py
```

---

## 📁 Project Structure

```
godot-docs-rag/
├── ingest/
│   ├── download_docs.sh         # Git clone of godot-docs repo
│   └── parse_docs.py            # Use Sphinx to parse reStructuredText to HTML
├── chunk/
│   └── llm_chunking.py          # Use GPT to generate Q&A from parsed docs
├── data/
│   ├── raw/                     # Raw .rst files (gitignored)
│   ├── parsed/                  # Cleaned/converted files (gitignored)
│   └── chunks/                  # Q&A pairs (gitignored)
├── scripts/
│   └── setup.sh                 # One-click setup script
├── Dockerfile                   # Container configuration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## ⚙️ Configuration

Key environment variables in `.env`:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
OPENAI_MODEL=gpt-4
GODOT_VERSION_BRANCH=4.4
MAX_CHUNK_SIZE=4000
TEMPERATURE=0.3
MAX_TOKENS=2000
OUTPUT_FORMAT=json
```

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

## 🔧 Development

### Running Individual Components

```bash
# Only download docs
bash ingest/download_docs.sh

# Only parse docs (requires docs to be downloaded)
python ingest/parse_docs.py

# Only generate Q&A (requires parsed HTML)
python chunk/llm_chunking.py
```

### Customizing the Pipeline

1. **Change Godot version**: Edit `GODOT_VERSION_BRANCH` in `.env`
2. **Adjust LLM model**: Change `OPENAI_MODEL` in `.env`
3. **Modify Q&A generation**: Edit the system prompt in `chunk/llm_chunking.py`
4. **Add new output formats**: Extend the `LLMChunker` class

---

## 🧠 Future Roadmap

- [ ] **Vector Database Integration**: FAISS, Chroma, or Weaviate support
- [ ] **Search Interface**: Streamlit or Gradio web UI
- [ ] **Multi-version Support**: Handle Godot 3.x, 4.x simultaneously  
- [ ] **Offline LLM**: Support for local models (Ollama, LM Studio)
- [ ] **Batch Processing**: Parallel processing for faster chunking
- [ ] **Quality Metrics**: Automated evaluation of Q&A pair quality
- [ ] **API Endpoint**: REST API for querying generated Q&A pairs

---

## 🐛 Troubleshooting

### Common Issues

1. **Sphinx build fails**:
   - Ensure Godot docs are downloaded: `bash ingest/download_docs.sh`
   - Check that `data/raw/godot-docs/conf.py` exists

2. **OpenAI API errors**:
   - Verify your API key in `.env`
   - Check API rate limits and billing

3. **No Q&A pairs generated**:
   - Ensure parsed HTML exists in `data/parsed/html/`
   - Check OpenAI API key and model availability

4. **Permission denied errors**:
   - Make scripts executable: `chmod +x scripts/setup.sh ingest/download_docs.sh`

### Debug Mode

Enable verbose logging:

```bash
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from chunk.llm_chunking import main
main()
"
```

---

## 🤝 Contributing

Contributions are welcome! Areas where help is needed:

- Support for other game engines (Unity, Unreal, etc.)
- Alternative LLM providers (Anthropic, Cohere, local models)
- Vector database implementations
- Web interface development
- Documentation improvements

### Development Setup

```bash
# Fork and clone the repo
git clone https://github.com/your-username/godot-docs-rag.git
cd godot-docs-rag

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt  # If you create this

# Make your changes and submit a PR!
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🧑‍💻 Author

**Max Schafer**  
Building developer tools, game tech, and open-source RAG projects.

- GitHub: [@your-github-username](https://github.com/your-github-username)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/your-profile)

---

## 🙏 Acknowledgments

- [Godot Engine](https://godotengine.org/) for their excellent documentation
- [Sphinx](https://www.sphinx-doc.org/) for reStructuredText parsing
- [OpenAI](https://openai.com/) for GPT models
- The open-source community for inspiration and tools

---

**Want to contribute a dataset or add your engine docs? PRs welcome! 🚀**
