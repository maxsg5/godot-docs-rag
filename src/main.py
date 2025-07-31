"""
Main entry point for Godot Docs RAG Pipeline
Supports both OpenAI API and local LLM (Ollama) providers
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ingest.download_docs import download_godot_html_docs
from ingest.parse_docs import GodotHTMLProcessor
from chunk.llm_provider import LLMProvider
from chunk.chunker import DocumentChunker

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
