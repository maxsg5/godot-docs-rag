# Godot Documentation RAG - Indexing Pipeline

This directory contains the offline indexing pipeline that processes the Godot HTML documentation and creates a vector database for the RAG system.

## Architecture

The indexing pipeline follows the standard RAG pattern:

```
Raw HTML Docs → Document Loading → Text Splitting → Embedding → Vector Store
```

## Components

### 1. Document Loading (`indexer.py`)

- **Input**: Processed HTML files from `../data/parsed/html/`
- **Process**: Uses LangChain's `BSHTMLLoader` to parse HTML documents
- **Output**: Structured documents with metadata

### 2. Text Splitting (`indexer.py`)

- **Splitter**: `HTMLSemanticPreservingSplitter`
- **Features**:
  - Preserves HTML structure (tables, lists, code blocks)
  - Splits on headers (h1, h2, h3, h4) with metadata preservation
  - Maintains semantic coherence
  - Configurable chunk size (default: 1000 chars, 200 overlap)

### 3. Embedding & Vector Store (`indexer.py`)

- **Embeddings**: SentenceTransformers `all-MiniLM-L6-v2`
- **Vector DB**: ChromaDB (persistent storage)
- **Collection**: `godot_docs`

### 4. Q&A Generation (`qa_generator.py`)

- **Purpose**: Create question-answer pairs for evaluation
- **Categories**: Automatically categorizes content (scripting, rendering, physics, etc.)
- **Difficulty Levels**: Basic, Intermediate, Advanced
- **Question Types**:
  - Definitional ("What is X?")
  - Procedural ("How to do X?")
  - Troubleshooting ("How to fix X?")
  - Code examples ("How to implement X?")

## Setup

### Prerequisites

- Python 3.8+
- Processed HTML documentation in `../data/parsed/html/`

### Quick Start

```bash
# From the indexing_pipeline directory
./setup.sh
```

This will:

1. Create a Python virtual environment
2. Install required packages
3. Create output directories

### Manual Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p output
```

## Usage

### 1. Run the Indexing Pipeline

```bash
source venv/bin/activate
python indexer.py
```

**Output:**

- `output/chroma_db/` - Persistent ChromaDB vector store
- `output/indexing_metadata.json` - Processing statistics and metadata

### 2. Generate Q&A Dataset

```bash
python qa_generator.py
```

**Output:**

- `output/godot_qa_dataset.json` - Question-answer pairs with metadata
- `output/qa_statistics.json` - Dataset statistics

## Configuration

### Indexer Settings (`indexer.py`)

```python
indexer = GodotDocumentationIndexer(
    html_dir="../data/parsed/html",     # Source HTML directory
    output_dir="output",                # Output directory
    chunk_size=1000,                    # Max chunk size in characters
    chunk_overlap=200                   # Overlap between chunks
)
```

### Text Splitter Configuration

```python
HTMLSemanticPreservingSplitter(
    headers_to_split_on=[               # Headers to preserve as metadata
        ("h1", "Header 1"),
        ("h2", "Header 2"), 
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ],
    max_chunk_size=chunk_size,
    separators=["\n\n", "\n", ". ", "! ", "? "],
    elements_to_preserve=["table", "ul", "ol", "code", "pre"],
    preserve_images=True,
    preserve_videos=False,
    denylist_tags=["script", "style", "nav", "header", "footer"],
)
```

## Output Format

### Vector Store Structure

- **Collection**: `godot_docs`
- **Embeddings**: 384-dimensional vectors (all-MiniLM-L6-v2)
- **Metadata per chunk**:
  - `source`: Relative path to source HTML file
  - `chunk_id`: Unique identifier
  - `chunk_index`: Position in source document
  - `Header 1-4`: Preserved header hierarchy
  - `doc_type`: "godot_documentation"

### Q&A Dataset Format

```json
{
  "metadata": {
    "total_pairs": 1500,
    "categories": ["scripting", "rendering", "physics", ...],
    "difficulties": ["basic", "intermediate", "advanced"]
  },
  "qa_pairs": [
    {
      "id": "qa_00001",
      "question": "How do you create a node in Godot?",
      "answer": "To create a node in Godot, you can...",
      "source_chunk_id": "getting_started/step_by_step_0",
      "source_file": "getting_started/step_by_step.html",
      "headers": {"Header 1": "Creating Nodes"},
      "difficulty": "basic",
      "category": "getting_started",
      "context": "Original chunk content for reference..."
    }
  ]
}
```

## Quality Assurance

### Chunk Quality

- Average chunk size tracking
- Source file distribution analysis
- Header preservation verification

### Q&A Quality

- Category distribution balance
- Difficulty level distribution
- Answer relevance to questions
- Context preservation

## Integration with Main RAG System

The indexing pipeline outputs are designed to be consumed by the main RAG application:

1. **Vector Store**: `output/chroma_db/` can be loaded directly by the retrieval system
2. **Q&A Dataset**: Used for evaluation and testing of the complete RAG pipeline
3. **Metadata**: Provides insights for system optimization

## Troubleshooting

### Common Issues

1. **No HTML files found**
   - Ensure you've run the HTML processing pipeline first
   - Check path: `../data/parsed/html/`

2. **Memory issues with large datasets**
   - Reduce `chunk_size` parameter
   - Process in batches by limiting `max_chunks` in Q&A generation

3. **Import errors**
   - Activate virtual environment: `source venv/bin/activate`
   - Reinstall requirements: `pip install -r requirements.txt`

### Performance Optimization

- **Embedding Generation**: Uses CPU by default, can be optimized with GPU
- **Vector Store**: ChromaDB is optimized for local development
- **Batch Processing**: Q&A generation can be parallelized for large datasets

## Dependencies

See `requirements.txt` for the complete list. Key dependencies:

- **langchain**: Document processing framework
- **langchain-community**: Community integrations
- **langchain-text-splitters**: Advanced text splitting
- **chromadb**: Vector database
- **sentence-transformers**: Embedding generation
- **beautifulsoup4**: HTML parsing
- **tqdm**: Progress tracking

## Next Steps

After running the indexing pipeline:

1. Test the vector store with sample queries
2. Evaluate Q&A quality manually
3. Use outputs in the main RAG application
4. Iterate on chunk size and splitting strategy based on retrieval performance
