"""
Test script for the Godot Documentation Indexing Pipeline
========================================================

This script validates that the indexing pipeline works correctly
and provides sample queries to test the vector store.
"""

import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from indexer import GodotDocumentationIndexer
from qa_generator import QAGenerator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_indexing_pipeline():
    """Test the complete indexing pipeline"""
    logger.info("🧪 Testing Godot Documentation Indexing Pipeline")
    logger.info("=" * 50)
    
    # Test 1: Initialize indexer
    logger.info("Test 1: Initializing indexer...")
    try:
        indexer = GodotDocumentationIndexer(
            html_dir="../data/parsed/html",
            output_dir="test_output",
            chunk_size=500,  # Smaller chunks for testing
            chunk_overlap=50
        )
        logger.info("✅ Indexer initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize indexer: {e}")
        return False
    
    # Test 2: Load documents
    logger.info("Test 2: Loading HTML documents...")
    try:
        documents = indexer.load_html_documents()
        if not documents:
            logger.error("❌ No documents loaded")
            return False
        logger.info(f"✅ Loaded {len(documents)} documents")
        
        # Show sample document
        sample_doc = documents[0]
        logger.info(f"Sample document source: {sample_doc.metadata.get('source')}")
        logger.info(f"Sample content preview: {sample_doc.page_content[:200]}...")
        
    except Exception as e:
        logger.error(f"❌ Failed to load documents: {e}")
        return False
    
    # Test 3: Split documents
    logger.info("Test 3: Splitting documents into chunks...")
    try:
        # Only use first 10 documents for testing
        test_docs = documents[:10]
        chunks = indexer.split_documents(test_docs)
        
        if not chunks:
            logger.error("❌ No chunks created")
            return False
            
        logger.info(f"✅ Created {len(chunks)} chunks from {len(test_docs)} documents")
        
        # Show sample chunk
        sample_chunk = chunks[0]
        logger.info(f"Sample chunk metadata: {sample_chunk.metadata}")
        logger.info(f"Sample chunk preview: {sample_chunk.page_content[:200]}...")
        
    except Exception as e:
        logger.error(f"❌ Failed to split documents: {e}")
        return False
    
    # Test 4: Create vector store (with limited chunks)
    logger.info("Test 4: Creating vector store...")
    try:
        # Use only first 20 chunks for testing
        test_chunks = chunks[:20]
        vector_store = indexer.create_vector_store(test_chunks)
        logger.info(f"✅ Vector store created with {len(test_chunks)} chunks")
        
    except Exception as e:
        logger.error(f"❌ Failed to create vector store: {e}")
        return False
    
    # Test 5: Test similarity search
    logger.info("Test 5: Testing similarity search...")
    try:
        test_queries = [
            "How to create a node in Godot?",
            "What is GDScript?", 
            "How do you handle input in Godot?",
            "Godot rendering system"
        ]
        
        for query in test_queries:
            results = vector_store.similarity_search(query, k=2)
            logger.info(f"Query: '{query}'")
            logger.info(f"Found {len(results)} results")
            
            if results:
                top_result = results[0]
                logger.info(f"  Top result source: {top_result.metadata.get('source')}")
                logger.info(f"  Top result preview: {top_result.page_content[:150]}...")
            logger.info("")
        
        logger.info("✅ Similarity search working correctly")
        
    except Exception as e:
        logger.error(f"❌ Failed similarity search test: {e}")
        return False
    
    # Test 6: Save metadata
    logger.info("Test 6: Saving metadata...")
    try:
        indexer.save_metadata(test_chunks)
        logger.info("✅ Metadata saved successfully")
    except Exception as e:
        logger.error(f"❌ Failed to save metadata: {e}")
        return False
    
    logger.info("\n" + "=" * 50)
    logger.info("🎉 All indexing tests passed!")
    
    return True


def test_qa_generation():
    """Test the Q&A generation pipeline"""
    logger.info("\n🧪 Testing Q&A Generation Pipeline")
    logger.info("=" * 50)
    
    # Check if vector store exists from previous test
    vector_store_path = Path("test_output/chroma_db")
    if not vector_store_path.exists():
        logger.error("❌ Vector store not found. Run indexing test first.")
        return False
    
    try:
        # Test 1: Initialize Q&A generator
        logger.info("Test 1: Initializing Q&A generator...")
        qa_generator = QAGenerator(str(vector_store_path))
        logger.info("✅ Q&A generator initialized successfully")
        
        # Test 2: Generate Q&A pairs (limited)
        logger.info("Test 2: Generating Q&A pairs...")
        qa_pairs = qa_generator.generate_qa_dataset(max_chunks=5)  # Only 5 chunks for testing
        
        if not qa_pairs:
            logger.error("❌ No Q&A pairs generated")
            return False
            
        logger.info(f"✅ Generated {len(qa_pairs)} Q&A pairs")
        
        # Show sample Q&A pairs
        for i, qa in enumerate(qa_pairs[:3], 1):
            logger.info(f"\nSample Q&A {i}:")
            logger.info(f"  Category: {qa.category}")
            logger.info(f"  Difficulty: {qa.difficulty}")
            logger.info(f"  Question: {qa.question}")
            logger.info(f"  Answer: {qa.answer[:200]}...")
            logger.info(f"  Source: {qa.source_file}")
        
        # Test 3: Save Q&A dataset
        logger.info("\nTest 3: Saving Q&A dataset...")
        qa_generator.save_qa_dataset(qa_pairs, "test_output/test_qa_dataset.json")
        logger.info("✅ Q&A dataset saved successfully")
        
        logger.info("\n" + "=" * 50)
        logger.info("🎉 All Q&A generation tests passed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Q&A generation test failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test output files"""
    import shutil
    
    test_output = Path("test_output")
    if test_output.exists():
        shutil.rmtree(test_output)
        logger.info("🧹 Test output files cleaned up")


def main():
    """Run all tests"""
    logger.info("🚀 Starting Godot Documentation Pipeline Tests")
    
    # Check prerequisites
    html_dir = Path("../data/parsed/html")
    if not html_dir.exists():
        logger.error(f"❌ HTML source directory not found: {html_dir}")
        logger.error("Please run the HTML processing pipeline first:")
        logger.error("  cd .. && python ingest/parse_docs.py")
        return
    
    # Run tests
    try:
        # Test indexing pipeline
        if not test_indexing_pipeline():
            logger.error("❌ Indexing pipeline tests failed")
            return
        
        # Test Q&A generation
        if not test_qa_generation():
            logger.error("❌ Q&A generation tests failed")
            return
        
        logger.info("\n" + "🎉" * 20)
        logger.info("🏆 ALL TESTS PASSED! 🏆")
        logger.info("The indexing pipeline is working correctly!")
        logger.info("🎉" * 20)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Tests interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error during testing: {e}")
    finally:
        # Clean up test files
        cleanup_test_files()


if __name__ == "__main__":
    main()
