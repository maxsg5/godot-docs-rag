"""
Godot Documentation RAG - Indexing Pipeline
===========================================

This pipeline processes the Godot HTML documentation and creates embeddings
for use in a RAG (Retrieval-Augmented Generation) system.

Architecture:
1. Load HTML documents from the processed documentation
2. Split documents into semantically meaningful chunks
3. Generate embeddings for each chunk
4. Store in vector database for retrieval

Output: Ready-to-use vector database for the RAG application
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# LangChain imports
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class GodotDocumentationIndexer:
    """Main indexing pipeline for Godot documentation"""
    
    def __init__(self, 
                 html_dir: str = "../data/parsed/html",
                 output_dir: str = "output",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the indexer
        
        Args:
            html_dir: Directory containing processed HTML files
            output_dir: Directory to store the vector database
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.html_dir = Path(html_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings model
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Configure HTML splitter for Godot documentation
        self.text_splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[
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
    
    def load_html_documents(self) -> List[Document]:
        """Load all HTML documents from the processed directory"""
        logger.info(f"🔍 Loading HTML documents from {self.html_dir}")
        
        html_files = list(self.html_dir.rglob("*.html"))
        documents = []
        
        logger.info(f"Found {len(html_files)} HTML files to process")
        
        for html_file in tqdm(html_files, desc="Loading HTML files"):
            try:
                # Use BeautifulSoup loader for better HTML parsing
                loader = BSHTMLLoader(str(html_file))
                docs = loader.load()
                
                # Add source metadata
                for doc in docs:
                    # Create relative path for cleaner source reference
                    rel_path = html_file.relative_to(self.html_dir)
                    doc.metadata.update({
                        "source": str(rel_path),
                        "file_path": str(html_file),
                        "doc_type": "godot_documentation"
                    })
                
                documents.extend(docs)
                
            except Exception as e:
                logger.warning(f"Failed to load {html_file}: {e}")
                continue
        
        logger.info(f"✅ Successfully loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using semantic-preserving splitter"""
        logger.info("📝 Splitting documents into chunks...")
        
        all_chunks = []
        
        for doc in tqdm(documents, desc="Splitting documents"):
            try:
                # Split the document while preserving HTML structure
                chunks = self.text_splitter.split_documents([doc])
                
                # Add chunk metadata
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_id": f"{doc.metadata.get('source', 'unknown')}_{i}",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk.page_content)
                    })
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.warning(f"Failed to split document {doc.metadata.get('source')}: {e}")
                continue
        
        logger.info(f"✅ Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def create_vector_store(self, chunks: List[Document]) -> Chroma:
        """Create vector store from document chunks"""
        logger.info("🔮 Creating vector embeddings and storing in ChromaDB...")
        
        # Create persistent Chroma vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.output_dir / "chroma_db"),
            collection_name="godot_docs"
        )
        
        logger.info(f"✅ Vector store created with {len(chunks)} chunks")
        return vector_store
    
    def save_metadata(self, chunks: List[Document]):
        """Save chunk metadata for analysis and debugging"""
        logger.info("💾 Saving metadata...")
        
        metadata = {
            "total_chunks": len(chunks),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "source_files": list(set(chunk.metadata.get("source") for chunk in chunks)),
            "average_chunk_size": sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0,
            "chunks": [
                {
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "source": chunk.metadata.get("source"),
                    "headers": {k: v for k, v in chunk.metadata.items() if k.startswith("Header")},
                    "content_preview": chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content,
                    "size": len(chunk.page_content)
                }
                for chunk in chunks[:100]  # Save first 100 for inspection
            ]
        }
        
        with open(self.output_dir / "indexing_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Metadata saved to {self.output_dir / 'indexing_metadata.json'}")
    
    def run_pipeline(self):
        """Run the complete indexing pipeline"""
        logger.info("🚀 Starting Godot Documentation Indexing Pipeline")
        logger.info("=" * 50)
        
        # Step 1: Load HTML documents
        documents = self.load_html_documents()
        if not documents:
            logger.error("❌ No documents loaded. Check HTML directory path.")
            return
        
        # Step 2: Split into chunks
        chunks = self.split_documents(documents)
        if not chunks:
            logger.error("❌ No chunks created. Check document splitting configuration.")
            return
        
        # Step 3: Create vector store
        vector_store = self.create_vector_store(chunks)
        
        # Step 4: Save metadata
        self.save_metadata(chunks)
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 Indexing pipeline completed successfully!")
        logger.info(f"📁 Vector database: {self.output_dir / 'chroma_db'}")
        logger.info(f"📊 Metadata: {self.output_dir / 'indexing_metadata.json'}")
        logger.info(f"📈 Total chunks indexed: {len(chunks)}")
        
        return vector_store


def main():
    """Main entry point"""
    # Initialize indexer with configuration
    indexer = GodotDocumentationIndexer(
        html_dir="../data/parsed/html",
        output_dir="output", 
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Run the pipeline
    vector_store = indexer.run_pipeline()
    
    if vector_store:
        # Test the vector store with a simple query
        logger.info("\n🔍 Testing vector store with sample query...")
        results = vector_store.similarity_search("How to create a node in Godot?", k=3)
        
        print("\nSample search results:")
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Headers: {[v for k, v in doc.metadata.items() if k.startswith('Header')]}")
            print(f"   Content: {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
