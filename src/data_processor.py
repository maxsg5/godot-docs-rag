"""
Data Processing Pipeline for Godot Documentation
Handles document ingestion, preprocessing, and chunk generation
"""

import logging
import asyncio
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass
import hashlib

try:
    from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader, TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_community.document_transformers import Html2TextTransformer
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class ProcessingStats:
    """Statistics from document processing"""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_chunks: int
    avg_chunk_size: float
    processing_time: float
    timestamp: str


class DocumentProcessor:
    """Processes various document types for RAG ingestion"""
    
    def __init__(self, data_dir: Path, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.chunks_dir = self.data_dir / "chunks"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.chunks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.logger = logging.getLogger("godot_rag.processor")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
            length_function=len,
        )
        
        # HTML to text transformer
        if LANGCHAIN_AVAILABLE:
            self.html_transformer = Html2TextTransformer()
        else:
            self.html_transformer = None
        
        # Godot documentation URLs
        self.godot_doc_urls = [
            # Getting Started
            "https://docs.godotengine.org/en/stable/getting_started/introduction/index.html",
            "https://docs.godotengine.org/en/stable/getting_started/step_by_step/index.html",
            "https://docs.godotengine.org/en/stable/getting_started/first_2d_game/index.html",
            "https://docs.godotengine.org/en/stable/getting_started/first_3d_game/index.html",
            
            # Scripting
            "https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/index.html",
            "https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_basics.html",
            "https://docs.godotengine.org/en/stable/tutorials/scripting/nodes_and_scene_instances.html",
            
            # 2D and 3D Tutorials
            "https://docs.godotengine.org/en/stable/tutorials/2d/physics_introduction.html",
            "https://docs.godotengine.org/en/stable/tutorials/2d/using_tilemaps.html",
            "https://docs.godotengine.org/en/stable/tutorials/3d/introduction_to_3d.html",
            "https://docs.godotengine.org/en/stable/tutorials/3d/using_3d_transforms.html",
            
            # Animation and Audio
            "https://docs.godotengine.org/en/stable/tutorials/animation/introduction.html",
            "https://docs.godotengine.org/en/stable/tutorials/audio/audio_streams.html",
            
            # UI and Input
            "https://docs.godotengine.org/en/stable/tutorials/ui/index.html",
            "https://docs.godotengine.org/en/stable/tutorials/inputs/index.html",
            
            # Networking and Files
            "https://docs.godotengine.org/en/stable/tutorials/networking/index.html",
            "https://docs.godotengine.org/en/stable/tutorials/io/data_paths.html",
            
            # Platform specific
            "https://docs.godotengine.org/en/stable/tutorials/export/index.html",
            "https://docs.godotengine.org/en/stable/tutorials/platform/index.html",
        ]
    
    async def process_all_sources(self) -> ProcessingStats:
        """Process all available document sources"""
        start_time = datetime.now()
        
        stats = ProcessingStats(
            total_documents=0,
            successful_documents=0,
            failed_documents=0,
            total_chunks=0,
            avg_chunk_size=0.0,
            processing_time=0.0,
            timestamp=start_time.isoformat()
        )
        
        try:
            # Process web sources
            web_stats = await self.process_web_sources()
            
            # Process local files
            local_stats = await self.process_local_files()
            
            # Combine statistics
            stats.total_documents = web_stats.total_documents + local_stats.total_documents
            stats.successful_documents = web_stats.successful_documents + local_stats.successful_documents
            stats.failed_documents = web_stats.failed_documents + local_stats.failed_documents
            stats.total_chunks = web_stats.total_chunks + local_stats.total_chunks
            
            if stats.total_chunks > 0:
                stats.avg_chunk_size = (
                    (web_stats.avg_chunk_size * web_stats.total_chunks + 
                     local_stats.avg_chunk_size * local_stats.total_chunks) / stats.total_chunks
                )
            
            stats.processing_time = (datetime.now() - start_time).total_seconds()
            
            # Save processing report
            await self._save_processing_report(stats)
            
            self.logger.info(f"✅ Processing completed: {stats.successful_documents}/{stats.total_documents} documents, {stats.total_chunks} chunks")
            
        except Exception as e:
            self.logger.error(f"❌ Processing failed: {e}")
            raise
        
        return stats
    
    async def process_web_sources(self) -> ProcessingStats:
        """Process web-based documentation"""
        if not LANGCHAIN_AVAILABLE or not REQUESTS_AVAILABLE:
            self.logger.warning("⚠️ Web processing requires langchain-community and requests")
            return ProcessingStats(0, 0, 0, 0, 0.0, 0.0, datetime.now().isoformat())
        
        start_time = datetime.now()
        
        successful_docs = []
        failed_urls = []
        all_chunks = []
        
        self.logger.info(f"🌐 Processing {len(self.godot_doc_urls)} web sources...")
        
        # Process URLs in batches to avoid overwhelming the server
        batch_size = 5
        for i in range(0, len(self.godot_doc_urls), batch_size):
            batch_urls = self.godot_doc_urls[i:i + batch_size]
            
            try:
                # Load documents
                loader = WebBaseLoader(web_paths=batch_urls)
                loader.requests_kwargs = {
                    'timeout': 30,
                    'headers': {
                        'User-Agent': 'Mozilla/5.0 (compatible; GodotRAGBot/1.0)'
                    }
                }
                
                documents = []
                for doc in loader.lazy_load():
                    if self._is_valid_document(doc):
                        # Clean and preprocess document
                        cleaned_doc = await self._clean_web_document(doc)
                        if cleaned_doc:
                            documents.append(cleaned_doc)
                            successful_docs.append(doc.metadata.get('source', 'unknown'))
                    else:
                        failed_urls.append(doc.metadata.get('source', 'unknown'))
                
                # Process documents into chunks
                for doc in documents:
                    chunks = await self._create_chunks(doc)
                    all_chunks.extend(chunks)
                
                # Small delay between batches
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"❌ Batch processing failed: {e}")
                failed_urls.extend(batch_urls)
        
        # Save web documents
        await self._save_web_documents(successful_docs, all_chunks)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        avg_chunk_size = sum(len(chunk.page_content) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        
        stats = ProcessingStats(
            total_documents=len(self.godot_doc_urls),
            successful_documents=len(successful_docs),
            failed_documents=len(failed_urls),
            total_chunks=len(all_chunks),
            avg_chunk_size=avg_chunk_size,
            processing_time=processing_time,
            timestamp=start_time.isoformat()
        )
        
        self.logger.info(f"🌐 Web processing completed: {len(successful_docs)}/{len(self.godot_doc_urls)} URLs processed")
        
        return stats
    
    async def process_local_files(self) -> ProcessingStats:
        """Process local files"""
        if not LANGCHAIN_AVAILABLE:
            self.logger.warning("⚠️ Local file processing requires langchain-community")
            return ProcessingStats(0, 0, 0, 0, 0.0, 0.0, datetime.now().isoformat())
        
        start_time = datetime.now()
        
        # Check for local files
        local_files = []
        for pattern in ["*.txt", "*.md", "*.html", "*.rst"]:
            local_files.extend(self.raw_dir.glob(f"**/{pattern}"))
        
        if not local_files:
            self.logger.info("📁 No local files found for processing")
            return ProcessingStats(0, 0, 0, 0, 0.0, 0.0, datetime.now().isoformat())
        
        successful_docs = []
        failed_files = []
        all_chunks = []
        
        self.logger.info(f"📁 Processing {len(local_files)} local files...")
        
        for file_path in local_files:
            try:
                # Load document based on file type
                if file_path.suffix.lower() in ['.txt', '.md']:
                    loader = TextLoader(str(file_path), encoding='utf-8')
                else:
                    # For other file types, use DirectoryLoader
                    loader = DirectoryLoader(
                        str(file_path.parent),
                        glob=file_path.name,
                        loader_cls=TextLoader
                    )
                
                documents = loader.load()
                
                for doc in documents:
                    if self._is_valid_document(doc):
                        # Clean and preprocess
                        cleaned_doc = await self._clean_local_document(doc)
                        if cleaned_doc:
                            chunks = await self._create_chunks(cleaned_doc)
                            all_chunks.extend(chunks)
                            successful_docs.append(str(file_path))
                    else:
                        failed_files.append(str(file_path))
                        
            except Exception as e:
                self.logger.error(f"❌ Failed to process {file_path}: {e}")
                failed_files.append(str(file_path))
        
        # Save local file processing results
        await self._save_local_documents(successful_docs, all_chunks)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        avg_chunk_size = sum(len(chunk.page_content) for chunk in all_chunks) / len(all_chunks) if all_chunks else 0
        
        stats = ProcessingStats(
            total_documents=len(local_files),
            successful_documents=len(successful_docs),
            failed_documents=len(failed_files),
            total_chunks=len(all_chunks),
            avg_chunk_size=avg_chunk_size,
            processing_time=processing_time,
            timestamp=start_time.isoformat()
        )
        
        self.logger.info(f"📁 Local processing completed: {len(successful_docs)}/{len(local_files)} files processed")
        
        return stats
    
    def _is_valid_document(self, doc: Document) -> bool:
        """Check if document is valid for processing"""
        if not doc.page_content or len(doc.page_content.strip()) < 100:
            return False
        
        # Check for common error pages or invalid content
        content_lower = doc.page_content.lower()
        if any(phrase in content_lower for phrase in [
            "404 not found", "page not found", "access denied",
            "error occurred", "temporarily unavailable"
        ]):
            return False
        
        return True
    
    async def _clean_web_document(self, doc: Document) -> Optional[Document]:
        """Clean and preprocess web document"""
        try:
            content = doc.page_content
            
            # Extract title from content if available
            title = ""
            if BS4_AVAILABLE:
                soup = BeautifulSoup(content, 'html.parser')
                title_tag = soup.find('title') or soup.find('h1')
                if title_tag:
                    title = title_tag.get_text().strip()
                
                # Remove navigation and footer elements
                for tag in soup.find_all(['nav', 'footer', 'aside', 'script', 'style']):
                    tag.decompose()
                
                # Extract main content
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                if main_content:
                    content = main_content.get_text()
                else:
                    content = soup.get_text()
            
            # Clean text
            content = self._clean_text(content)
            
            if len(content.strip()) < 100:
                return None
            
            # Update metadata
            metadata = doc.metadata.copy()
            metadata.update({
                'title': title,
                'content_type': 'web',
                'processed_at': datetime.now().isoformat(),
                'content_hash': hashlib.md5(content.encode()).hexdigest(),
                'char_count': len(content),
                'word_count': len(content.split())
            })
            
            return Document(page_content=content, metadata=metadata)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clean web document: {e}")
            return None
    
    async def _clean_local_document(self, doc: Document) -> Optional[Document]:
        """Clean and preprocess local document"""
        try:
            content = self._clean_text(doc.page_content)
            
            if len(content.strip()) < 100:
                return None
            
            # Extract title from filename or first line
            title = Path(doc.metadata.get('source', '')).stem
            if content.startswith('#'):
                first_line = content.split('\n')[0]
                title = first_line.strip('# ').strip()
            
            # Update metadata
            metadata = doc.metadata.copy()
            metadata.update({
                'title': title,
                'content_type': 'local',
                'processed_at': datetime.now().isoformat(),
                'content_hash': hashlib.md5(content.encode()).hexdigest(),
                'char_count': len(content),
                'word_count': len(content.split())
            })
            
            return Document(page_content=content, metadata=metadata)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to clean local document: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\!\?\;\:\,\-\(\)\[\]\{\}\"\'\/\\\@\#\$\%\^\&\*\+\=\|\~\`]', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([\.!\?;:,])', r'\1', text)
        text = re.sub(r'([\.!\?;:,])\s+', r'\1 ', text)
        
        # Remove excessive spaces
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()
    
    async def _create_chunks(self, doc: Document) -> List[Document]:
        """Create chunks from document"""
        try:
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_id': f"{chunk.metadata.get('content_hash', 'unknown')}_{i}",
                    'chunk_size': len(chunk.page_content),
                    'chunk_words': len(chunk.page_content.split())
                })
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create chunks: {e}")
            return []
    
    async def _save_web_documents(self, successful_urls: List[str], chunks: List[Document]):
        """Save processed web documents"""
        try:
            # Save metadata
            web_metadata = {
                'processed_at': datetime.now().isoformat(),
                'successful_urls': successful_urls,
                'total_chunks': len(chunks),
                'processing_method': 'web_loader'
            }
            
            with open(self.processed_dir / 'web_metadata.json', 'w') as f:
                json.dump(web_metadata, f, indent=2)
            
            # Save chunks
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    'content': chunk.page_content,
                    'metadata': chunk.metadata
                })
            
            with open(self.chunks_dir / 'web_chunks.json', 'w') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Saved {len(chunks)} web chunks")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save web documents: {e}")
    
    async def _save_local_documents(self, successful_files: List[str], chunks: List[Document]):
        """Save processed local documents"""
        try:
            # Save metadata
            local_metadata = {
                'processed_at': datetime.now().isoformat(),
                'successful_files': successful_files,
                'total_chunks': len(chunks),
                'processing_method': 'local_loader'
            }
            
            with open(self.processed_dir / 'local_metadata.json', 'w') as f:
                json.dump(local_metadata, f, indent=2)
            
            # Save chunks
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    'content': chunk.page_content,
                    'metadata': chunk.metadata
                })
            
            with open(self.chunks_dir / 'local_chunks.json', 'w') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Saved {len(chunks)} local chunks")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save local documents: {e}")
    
    async def _save_processing_report(self, stats: ProcessingStats):
        """Save processing report"""
        try:
            report = {
                'processing_stats': {
                    'total_documents': stats.total_documents,
                    'successful_documents': stats.successful_documents,
                    'failed_documents': stats.failed_documents,
                    'total_chunks': stats.total_chunks,
                    'avg_chunk_size': stats.avg_chunk_size,
                    'processing_time': stats.processing_time,
                    'timestamp': stats.timestamp
                },
                'configuration': {
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap,
                    'splitter_type': 'RecursiveCharacterTextSplitter'
                },
                'sources': {
                    'web_urls_count': len(self.godot_doc_urls),
                    'raw_files_checked': True
                }
            }
            
            with open(self.processed_dir / 'processing_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info("📊 Processing report saved")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save processing report: {e}")
    
    async def load_processed_chunks(self) -> List[Dict[str, Any]]:
        """Load all processed chunks"""
        all_chunks = []
        
        # Load web chunks
        web_chunks_file = self.chunks_dir / 'web_chunks.json'
        if web_chunks_file.exists():
            with open(web_chunks_file, 'r') as f:
                web_chunks = json.load(f)
                all_chunks.extend(web_chunks)
        
        # Load local chunks
        local_chunks_file = self.chunks_dir / 'local_chunks.json'
        if local_chunks_file.exists():
            with open(local_chunks_file, 'r') as f:
                local_chunks = json.load(f)
                all_chunks.extend(local_chunks)
        
        self.logger.info(f"📥 Loaded {len(all_chunks)} processed chunks")
        return all_chunks
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        try:
            report_file = self.processed_dir / 'processing_report.json'
            if report_file.exists():
                with open(report_file, 'r') as f:
                    return json.load(f)
            else:
                return {"status": "No processing completed yet"}
        except Exception as e:
            self.logger.error(f"❌ Failed to get processing status: {e}")
            return {"status": "Error loading status", "error": str(e)}
