"""
Main entry point for the Godot RAG System
Orchestrates system initialization and startup with comprehensive features
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv()

from src.config import RAGConfig
from src.rag_system import ComprehensiveRAGSystem
from src.monitoring import MetricsCollector
from src.data_processor import DocumentProcessor

# Legacy imports for backward compatibility
from legacy.ingest.download_docs import download_godot_html_docs
from legacy.ingest.parse_docs import GodotHTMLProcessor
from legacy.chunk.llm_provider import LLMProvider
from legacy.chunk.chunker import DocumentChunker


async def main():
    """Main entry point for the comprehensive RAG system"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("godot_rag.main")
    logger.info("🚀 Starting Godot RAG System...")
    
    try:
        # Initialize configuration
        config = RAGConfig()
        
        # Initialize components
        logger.info("📊 Initializing metrics collector...")
        metrics_collector = MetricsCollector(config.data_dir)
        metrics_collector.start_background_collection()
        
        logger.info("🔄 Initializing data processor...")
        data_processor = DocumentProcessor(config.data_dir)
        
        logger.info("🤖 Initializing RAG system...")
        rag_system = ComprehensiveRAGSystem(config)
        
        # Initialize RAG system
        await rag_system.initialize()
        
        if rag_system.is_ready():
            logger.info("✅ Godot RAG System is ready!")
            
            # Example query for testing
            test_query = "How do I create a 2D player character in Godot?"
            logger.info(f"🔍 Testing with query: {test_query}")
            
            result = await rag_system.answer_question(test_query)
            
            logger.info("📝 Test result:")
            logger.info(f"Answer: {result['answer'][:200]}...")
            logger.info(f"Processing time: {result['processing_time']:.2f}s")
            logger.info(f"Retrieved documents: {result['num_documents']}")
            
            # Record test metrics
            metrics_collector.record_query(result, success=True)
            
        else:
            logger.error("❌ RAG system failed to initialize properly")
            return 1
            
    except Exception as e:
        logger.error(f"❌ System startup failed: {e}")
        return 1
    
    logger.info("✅ System startup completed successfully")
    return 0


def legacy_main():
    """Legacy main function for backward compatibility"""
    logger = logging.getLogger("godot_rag.legacy")
    logger.info("🔄 Running legacy pipeline...")
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Create data directories
        data_dir = Path('data')
        raw_dir = data_dir / 'raw'
        parsed_dir = data_dir / 'parsed' / 'html'
        chunks_dir = data_dir / 'chunks'
        
        for dir_path in [raw_dir, parsed_dir, chunks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Download and parse documents
        logger.info("📥 Downloading Godot documentation...")
        success = download_godot_html_docs(str(raw_dir))
        
        if success:
            logger.info("✅ Documentation downloaded successfully")
            
            # Parse HTML documents
            logger.info("🔄 Parsing HTML documents...")
            processor = GodotHTMLProcessor(str(raw_dir), str(parsed_dir))
            processed_count = processor.process_all()
            
            logger.info(f"✅ Processed {processed_count} HTML files")
            
            # Initialize LLM provider
            logger.info("🤖 Initializing LLM provider...")
            llm_provider = LLMProvider()
            
            # Chunk documents
            logger.info("📄 Chunking documents...")
            chunker = DocumentChunker(llm_provider)
            result = chunker.process_directory(str(parsed_dir), str(chunks_dir))
            
            if result:
                logger.info("✅ Legacy pipeline completed successfully")
                return 0
            else:
                logger.error("❌ Chunking failed")
                return 1
        else:
            logger.error("❌ Failed to download documentation")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Legacy pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    # Check if we should run legacy mode
    if "--legacy" in sys.argv:
        exit_code = legacy_main()
    else:
        exit_code = asyncio.run(main())
    
    sys.exit(exit_code)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("🔍 Checking prerequisites...")
    
    # Check LLM provider configuration
    llm_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            logger.error("❌ OpenAI API key not configured. Set OPENAI_API_KEY in .env file")
            return False
        logger.info("✅ OpenAI API key configured")
        
    elif llm_provider == "ollama":
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        logger.info(f"✅ Using Ollama at {ollama_url}")
        
    else:
        logger.error(f"❌ Unknown LLM provider: {llm_provider}")
        return False
    
    return True


def run_pipeline():
    """Run the complete RAG pipeline"""
    try:
        logger.info("🚀 Starting Godot Docs RAG Pipeline...")
        
        # Check prerequisites
        if not check_prerequisites():
            sys.exit(1)
        
        # Step 1: Download documentation
        logger.info("📥 Step 1: Downloading Godot HTML documentation...")
        download_success = download_godot_html_docs()
        if not download_success:
            logger.error("❌ Failed to download documentation")
            sys.exit(1)
        
        # Step 2: Process HTML documentation
        logger.info("🔧 Step 2: Processing HTML documentation...")
        processor = GodotHTMLProcessor()
        if not processor.validate_source():
            sys.exit(1)
        
        processor.setup_directories()
        if not processor.process_files():
            logger.error("❌ Failed to process documentation")
            sys.exit(1)
        
        # Step 3: Initialize LLM provider
        logger.info("🧠 Step 3: Initializing LLM provider...")
        llm_provider = LLMProvider()
        if not llm_provider.test_connection():
            logger.error("❌ Failed to connect to LLM provider")
            sys.exit(1)
        
        # Step 4: Generate Q&A pairs
        logger.info("🔍 Step 4: Generating Q&A pairs...")
        chunker = DocumentChunker(llm_provider)
        chunker.setup_output_dir()
        
        success_count, total_qa_pairs = chunker.process_all_files()
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"✅ Successfully processed: {success_count} files")
        logger.info(f"📝 Total Q&A pairs generated: {total_qa_pairs}")
        logger.info(f"📁 Output directory: data/chunks/")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return False


if __name__ == "__main__":
    run_pipeline()
