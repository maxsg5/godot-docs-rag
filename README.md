# 🔍 Godot Docs RAG Assistant

A fully Dockerized (Everything is in docker-compose) Retrieval-Augmented Generation (RAG) system for the Godot game engine documentation. Features an offline indexing pipeline and supports both OpenAI API and local LLM inference via Ollama.

---

## ❓ Problem

Godot's documentation is comprehensive but can be difficult to search semantically or query conversationally. This project transforms the official docs into a RAG system with vector embeddings and intelligent Q&A capabilities.

---

## 🏗️ Architecture

This project follows the standard RAG pattern with Automated ingestion pipeline:

---

## 🎯 Features

- ✅ **Dual LLM Support**: OpenAI API or local Ollama models
- ✅ **Fully Dockerized**: One-command setup with Docker Compose
- ✅ **LangChain Integration**: Advanced document processing and splitting using the langchain document loader https://python.langchain.com/docs/how_to/document_loader_web/
- ✅ **Semantic Chunking**: Preserves HTML structure (tables, lists, code)
- ✅ **Vector Database**: ChromaDB for efficient similarity search
- ✅ **Q&A Generation**: Automatic question-answer pair creation

- ✅ **Production Ready**: Scalable container architecture

---

## 📊 Data Source
https://docs.godotengine.org/en/stable/

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
GODOT_VERSION_BRANCH=release
TEMPERATURE=0.3
MAX_TOKENS=2000
```

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
