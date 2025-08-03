from http import client
import os
import sys
import time
import glob
import pickle
import warnings
from pathlib import Path
from typing import List, Dict, Any
import yaml
import requests
from tqdm import tqdm
from fastembed import TextEmbedding
import json

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter, 
    HTMLSemanticPreservingSplitter,
    RecursiveCharacterTextSplitter
)

from qdrant_client import QdrantClient, models

overall_start_time = time.time()

class GodotRAGPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the RAG pipeline with configuration"""
        self.config = self.load_config(config_path)
        self._print_config()
        self.embeddings = None
        self.vector_store = None
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"❌ Configuration file {config_path} not found")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            sys.exit(1)
    
    def _print_config(self):
        """Print configuration in a nicely formatted way"""
        print("\n" + "=" * 60)
        print("📋 GODOT RAG PIPELINE CONFIGURATION")
        print("=" * 60)
        
        # Document Loading
        print("\n📄 Document Loading:")
        doc_config = self.config.get('document_loading', {})
        print(f"   Method: {doc_config.get('method', 'unstructured')}")
        
        # Embedding Configuration
        print("\n🤖 Embedding Configuration:")
        embed_config = self.config.get('embedding', {})
        print(f"   Model: {embed_config.get('model', 'llama3')}")
        print(f"   Base URL: {embed_config.get('base_url', 'http://localhost:11434')}")
        print(f"   GPU Enabled: {embed_config.get('gpu_enabled', True)}")
        print(f"   GPU Layers: {embed_config.get('gpu_layers', -1)}")
        print(f"   Batch Size: {embed_config.get('batch_size', 50)}")
        print(f"   Show Progress: {embed_config.get('show_progress', True)}")
        print(f"   Timeout: {embed_config.get('timeout', 300)}s")
        
        # Vector Store
        print("\n🗃️  Vector Store:")
        vector_config = self.config.get('vector_store', {})
        print(f"   Type: {vector_config.get('type', 'in_memory')}")
        
        # Text Splitting
        print("\n✂️  Text Splitting:")
        split_config = self.config.get('text_splitting', {})
        print(f"   Method: {split_config.get('method', 'html_semantic')}")
        
        # Show method-specific settings
        method = split_config.get('method', 'html_semantic')
        if method == 'html_semantic':
            semantic_config = split_config.get('html_semantic', {})
            print(f"   Max Chunk Size: {semantic_config.get('max_chunk_size', 2000)}")
            print(f"   Chunk Overlap: {semantic_config.get('chunk_overlap', 200)}")
            print(f"   Preserve Images: {semantic_config.get('preserve_images', True)}")
            print(f"   Preserve Videos: {semantic_config.get('preserve_videos', True)}")
        
        # Headers to split on
        headers = split_config.get('headers_to_split_on', [])
        if headers:
            print(f"   Headers to Split: {', '.join([h[0] for h in headers])}")
        
        # Secondary Splitting
        print("\n🔄 Secondary Splitting:")
        secondary_config = self.config.get('secondary_splitting', {})
        enabled = secondary_config.get('enabled', True)
        print(f"   Enabled: {enabled}")
        if enabled:
            print(f"   Chunk Size: {secondary_config.get('chunk_size', 1000)}")
            print(f"   Chunk Overlap: {secondary_config.get('chunk_overlap', 100)}")
        
        # Output Settings
        print("\n💾 Output Settings:")
        output_config = self.config.get('output', {})
        print(f"   Save Splits: {output_config.get('save_splits', False)}")
        print(f"   Output Directory: {output_config.get('output_directory', 'data/processed')}")
        print(f"   Save Metadata: {output_config.get('save_metadata', True)}")
        
        print("=" * 60 + "\n")
    
    def wait_for_ollama(self, base_url: str, max_retries: int = 30) -> bool:
        """Wait for Ollama service to be ready"""
        print("🔄 Waiting for Ollama service to start...")
        for i in range(max_retries):
            try:
                response = requests.get(f"{base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✅ Ollama service is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print(f"⏳ Attempt {i+1}/{max_retries} - Waiting for Ollama...")
            time.sleep(2)
        
        print("❌ Ollama service failed to start")
        return False
    
    def wait_for_qdrant(self, url: str, max_retries: int = 30):
        """Wait for Qdrant to be ready"""
        print("🔄 Waiting for Qdrant to be ready...")
        
        client = QdrantClient(url)
        # Check if Qdrant is already running
        try:
            response = client.get_collections()
            if response:
                print("✅ Qdrant is already running!")
                return True
        except Exception as e:
            print(f"⚠️ Error checking Qdrant status: {e}")
       
        try:
            # List existing collections
            collections = client.get_collections()
            print(f"📁 Existing collections: {collections}")
            
            print(f"\n🌐 Qdrant Web UI available at: {qdrant_base_url}/dashboard")
            
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            print("💡 Make sure to run: docker run -d -p 6333:6333 -p 6334:6334 -v \"$(pwd)/qdrant_storage:/qdrant/storage:z\" --name qdrant qdrant/qdrant")

        print("❌ Qdrant failed to start")
        return False
    
    def setup_ollama_model(self, base_url: str, model_name: str) -> bool:
        """Pull and setup the Ollama model with GPU support"""
        print(f"📥 Setting up model: {model_name}")
        
        # Configure GPU settings if enabled
        gpu_enabled = self.config.get('embedding', {}).get('gpu_enabled', True)
        gpu_layers = self.config.get('embedding', {}).get('gpu_layers', -1)
        
        if gpu_enabled:
            print(f"🎮 GPU acceleration enabled with {gpu_layers} layers")
            # Set GPU environment variables
            os.environ['CUDA_VISIBLE_DEVICES'] = self.config.get('embedding', {}).get('cuda_devices', '0')
        else:
            print("🖥️  Using CPU-only mode")
            
        try:
            # Check if model exists
            response = requests.get(f"{base_url}/api/tags")
            models = response.json().get('models', [])
            
            if not any(model['name'].startswith(model_name) for model in models):
                print(f"📥 Pulling model {model_name}...")
                pull_response = requests.post(f"{base_url}/api/pull", 
                                            json={"name": model_name})
                
                if pull_response.status_code != 200:
                    print(f"❌ Failed to pull model: {pull_response.text}")
                    return False
            
            print(f"✅ Model {model_name} is ready!")
            return True
            
        except Exception as e:
            print(f"❌ Error setting up model: {e}")
            return False
    
    def initialize_components(self):
        """Initialize embeddings and vector store"""
        import traceback
        
        qdrant_base_url = self.config.get('vector_store', {}).get('base_url',"")

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", 
                                   self.config.get('embedding', {}).get('base_url', "http://localhost:11434"))
        model_name = self.config.get('embedding', {}).get('model', 'llama3')
        
        # Wait for Ollama and setup model
        if not self.wait_for_ollama(ollama_base_url,5):
            return False
        
        if not self.setup_ollama_model(ollama_base_url, model_name):
            return False
        
        if not self.wait_for_qdrant(qdrant_base_url,5):
            return False
        
        print("✅ Components initialized successfully!")
        return True
        # Initialize embeddings and vector store
        # try:
        #     self.embeddings = OllamaEmbeddings(
        #         model=model_name,
        #         base_url=ollama_base_url
        #     )
        #     self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
            
        #     # Test the embedding connection
        #     print("🧪 Testing embedding connection...")
        #     test_text = "This is a test sentence for embedding."
        #     test_embedding = self.embeddings.embed_query(test_text)
        #     print(f"✅ Embedding test successful! Vector size: {len(test_embedding)}")
            
        #     print("✅ Components initialized successfully!")
        #     return True
        # except Exception as e:
        #     print(f"❌ Error initializing components: {e}")
        #     print(f"📋 Full error details: {traceback.format_exc()}")
        #     return False
    
    def download_and_extract(self) -> bool:
        """Download and extract Godot documentation"""
        print("📥 Downloading Godot documentation...")
        try:
            from LoadData import download_and_extract_godot_docs
            return download_and_extract_godot_docs()
        except Exception as e:
            print(f"❌ Error downloading documentation: {e}")
            return False
    
    def find_html_files(self, directory: str) -> List[str]:
        """Find all HTML files recursively"""
        pattern = os.path.join(directory, "**", "*.html")
        return glob.glob(pattern, recursive=True)
    
    def load_documents(self) -> List[Document]:
        """Load HTML documents into LangChain Document objects"""
        print("📄 Loading documents...")
        docs_dir = "data/raw"
        
        if not os.path.exists(docs_dir):
            print(f"❌ Documentation directory {docs_dir} not found")
            return []
        
        html_files = self.find_html_files(docs_dir)
        print(f"📊 Found {len(html_files)} HTML files")
        
        if not html_files:
            print("❌ No HTML files found")
            return []
        
        documents = []
        method = self.config['document_loading']['method']
        
        print(f"🔄 Loading documents using {method} method...")
        
        for file_path in tqdm(html_files, desc="Loading documents"):
            try:
                if method.lower() == "unstructured":
                    loader = UnstructuredHTMLLoader(file_path)
                elif method.lower() == "bs4":
                    loader = BSHTMLLoader(file_path)
                elif method.lower() == "fastembed":
                    
                    
                    
                    
                    EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-small-en"
                    EMBEDDING_DIMENSIONS = 512
                    COLLECTION_NAME = "godot-docs-optimized"
                    
                    
                    
                    
                    print(f"\\n🎯 Selected model: {EMBEDDING_MODEL}")
                    print(f"📐 Dimensions: {EMBEDDING_DIMENSIONS}")
                    
                    try:
                        client = QdrantClient(self.config.get('vector_store', {}).get('base_url', "http://localhost:6333"))
                        # Delete existing collection if exists
                        try:
                            client.delete_collection(COLLECTION_NAME)
                            print(f"🗑️ Deleted existing collection: {COLLECTION_NAME}")
                        except:
                            pass  # Collection doesn't exist
    
                    
                        # Create new collection
                        client.create_collection(
                            collection_name=COLLECTION_NAME,
                            vectors_config=models.VectorParams(
                                size=EMBEDDING_DIMENSIONS,
                                distance=models.Distance.COSINE  # Best for semantic similarity
                            )
                        )
                    
                        print(f"✅ Created collection: {COLLECTION_NAME}")
                        # Create payload indexes for efficient filtering
                        indexes_to_create = [
                            ("section", "keyword"),
                            ("chunk_type", "keyword"), 
                            ("difficulty", "keyword"),
                            ("keywords", "keyword")
                        ]

                        for field_name, field_type in indexes_to_create:
                            try:
                                client.create_payload_index(
                                    collection_name=COLLECTION_NAME,
                                    field_name=field_name,
                                    field_schema=field_type
                                )
                                print(f"📊 Created index for: {field_name}")
                            except Exception as e:
                                print(f"⚠️ Index creation warning for {field_name}: {e}")
    
                        print("\\n🎉 Collection setup complete!")
                    #loader = FastEmbedLoader(file_path)
                    except Exception as e:
                        print(f"❌ Error creating collection: {e}")
                else:
                    print(f"❌ Unknown loading method: {method}")
                    continue
                
                docs = loader.load()
                documents.extend(docs)
                
            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        print("✂️ Splitting documents...")
        method = self.config['text_splitting']['method']
        
        print(f"🔄 Using {method} splitting method...")
        
        if method == "html_header":
            return self._split_html_header(documents)
        elif method == "html_section":
            return self._split_html_section(documents)
        elif method == "html_semantic":
            return self._split_html_semantic(documents)
        else:
            print(f"❌ Unknown splitting method: {method}")
            return documents
    
    def _split_html_header(self, documents: List[Document]) -> List[Document]:
        """Split using HTML headers"""
        headers_to_split = self.config['text_splitting']['headers_to_split_on']
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split)
        
        splits = []
        for doc in tqdm(documents, desc="HTML Header Splitting"):
            try:
                doc_splits = splitter.split_text(doc.page_content)
                for split in doc_splits:
                    split.metadata.update(doc.metadata)
                splits.extend(doc_splits)
            except Exception as e:
                print(f"⚠️ Error splitting document: {e}")
                continue
        
        return self._apply_secondary_splitting(splits)
    
    def _split_html_section(self, documents: List[Document]) -> List[Document]:
        """Split using HTML sections"""
        splitter = HTMLSectionSplitter()
        
        splits = []
        for doc in tqdm(documents, desc="HTML Section Splitting"):
            try:
                doc_splits = splitter.split_text(doc.page_content)
                for split in doc_splits:
                    split.metadata.update(doc.metadata)
                splits.extend(doc_splits)
            except Exception as e:
                print(f"⚠️ Error splitting document: {e}")
                continue
        
        return self._apply_secondary_splitting(splits)
    
    def _split_html_semantic(self, documents: List[Document]) -> List[Document]:
        """Split using HTML semantic preservation"""
        config = self.config['text_splitting']['html_semantic']
        headers_to_split = self.config['text_splitting']['headers_to_split_on']
        
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=headers_to_split,
            max_chunk_size=config.get('max_chunk_size', 2000),
            chunk_overlap=config.get('chunk_overlap', 200),
            separators=config.get('separators', ["\n\n", "\n", ". "]),
            preserve_images=config.get('preserve_images', True),
            preserve_videos=config.get('preserve_videos', True),
            elements_to_preserve=config.get('elements_to_preserve', ["table", "ul", "ol"]),
            denylist_tags=config.get('denylist_tags', ["script", "style"])
        )
        
        splits = []
        for doc in tqdm(documents, desc="HTML Semantic Splitting"):
            try:
                doc_splits = splitter.split_text(doc.page_content)
                for split in doc_splits:
                    split.metadata.update(doc.metadata)
                splits.extend(doc_splits)
            except Exception as e:
                print(f"⚠️ Error splitting document: {e}")
                continue
        
        return self._apply_secondary_splitting(splits)
    
    def _apply_secondary_splitting(self, splits: List[Document]) -> List[Document]:
        """Apply secondary splitting if enabled"""
        if not self.config['secondary_splitting']['enabled']:
            return splits
        
        chunk_size = self.config['secondary_splitting']['chunk_size']
        chunk_overlap = self.config['secondary_splitting']['chunk_overlap']
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        final_splits = []
        for split in tqdm(splits, desc="Secondary Splitting"):
            try:
                sub_splits = text_splitter.split_documents([split])
                final_splits.extend(sub_splits)
            except Exception as e:
                print(f"⚠️ Error in secondary splitting: {e}")
                final_splits.append(split)
        
        print(f"📊 Final splits: {len(final_splits)} chunks")
        return final_splits
    
    def embed_and_store_documents(self, documents: List[Document]) -> bool:
        """Embed documents and store in vector store with detailed progress"""
        if not self.vector_store:
            print("❌ Vector store not initialized")
            return False
        
        print("📊 Embedding and storing documents...")
        print(f"📝 Total documents to process: {len(documents)}")
        
        # Get configuration
        batch_size = self.config.get('embedding', {}).get('batch_size', 50)
        show_progress = self.config.get('embedding', {}).get('show_progress', True)
        
        print(f"📦 Using batch size: {batch_size}")
        
        try:
            # Add documents in batches to avoid memory issues
            total_batches = (len(documents) + batch_size - 1) // batch_size
            total_processed = 0
            
            print(f"🔄 Processing {total_batches} batches...")
            print("⏳ Starting first batch (this may take a moment to initialize)...")
            
            # Use enumerate for better progress tracking
            for batch_idx, i in enumerate(range(0, len(documents), batch_size)):
                batch = documents[i:i + batch_size]
                batch_num = batch_idx + 1
                
                if show_progress:
                    print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
                
                # Time the embedding process
                import time
                start_time = time.time()
                
                try:
                    # Test connection with a small sample first for the first batch
                    if batch_idx == 0:
                        print("🧪 Testing embedding connection with first document...")
                        test_doc = [batch[0]] if batch else []
                        if test_doc:
                            self.vector_store.add_documents(documents=test_doc)
                            print("✅ Embedding connection successful!")
                            # Process the rest of the first batch
                            if len(batch) > 1:
                                remaining_batch = batch[1:]
                                self.vector_store.add_documents(documents=remaining_batch)
                        total_processed += len(batch)
                    else:
                        # Process normally for subsequent batches
                        self.vector_store.add_documents(documents=batch)
                        total_processed += len(batch)
                    
                    embed_time = time.time() - start_time
                    overall_elapsed = time.time() - overall_start_time
                    
                    if show_progress:
                        print(f"✅ Batch {batch_num} completed in {embed_time:.2f}s")
                        print(f"📊 Progress: {total_processed}/{len(documents)} documents ({total_processed/len(documents)*100:.1f}%)")
                        
                        # Calculate and show rate
                        docs_per_sec = len(batch) / embed_time if embed_time > 0 else 0
                        overall_docs_per_sec = total_processed / overall_elapsed if overall_elapsed > 0 else 0
                        print(f"⚡ Batch rate: {docs_per_sec:.1f} docs/sec | Overall rate: {overall_docs_per_sec:.1f} docs/sec")
                        
                        # Estimate remaining time based on overall performance
                        if total_processed < len(documents):
                            remaining_docs = len(documents) - total_processed
                            eta_seconds = remaining_docs / overall_docs_per_sec if overall_docs_per_sec > 0 else 0
                            eta_minutes = eta_seconds / 60
                            print(f"⏱️  ETA: {eta_minutes:.1f} minutes ({eta_seconds:.0f}s)")
                        
                        print("-" * 50)
                
                except Exception as batch_error:
                    print(f"⚠️ Error processing batch {batch_num}: {batch_error}")
                    print("🔄 Attempting individual document processing...")
                    
                    # Try to process documents individually in this batch
                    for doc_idx, doc in enumerate(batch):
                        try:
                            self.vector_store.add_documents(documents=[doc])
                            total_processed += 1
                            if show_progress and doc_idx % 10 == 0:  # Show progress every 10 docs
                                print(f"✅ Individual document {doc_idx + 1}/{len(batch)} in batch {batch_num} processed")
                        except Exception as doc_error:
                            print(f"❌ Failed to process document {doc_idx + 1} in batch {batch_num}: {doc_error}")
                            continue
            
            print("\n" + "=" * 60)
            print(f"✅ Successfully stored {total_processed}/{len(documents)} documents in vector store")
            
            if total_processed < len(documents):
                failed_count = len(documents) - total_processed
                print(f"⚠️ {failed_count} documents failed to process")
            
            # Show final statistics
            print(f"📊 Vector store now contains documents for semantic search")
            print(f"🎯 Ready for queries!")
            print("=" * 60)
            
            return total_processed > 0
            
        except Exception as e:
            print(f"❌ Error storing documents: {e}")
            import traceback
            print(f"📋 Full error details: {traceback.format_exc()}")
            return False
    
    def search_documents(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        if not self.vector_store:
            print("❌ Vector store not initialized")
            return []
        
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"❌ Error searching documents: {e}")
            return []
    
    def save_pipeline_state(self, documents: List[Document], output_dir: str = "data/processed"):
        """Save pipeline state for later use"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save processed documents
        with open(f"{output_dir}/processed_documents.pkl", "wb") as f:
            pickle.dump(documents, f)
        
        print(f"💾 Pipeline state saved to {output_dir}")
    
    def run_pipeline(self) -> bool:
        """Run the complete RAG pipeline"""
        print("🚀 Starting Godot Documentation RAG Pipeline")
        print("=" * 50)
        
        # Step 1: Initialize components
        if not self.initialize_components():
            return False
        
        # # Step 2: Download documentation
        # if not self.download_and_extract():
        #     return False
        
        # Step 3: Load documents
        documents = self.load_documents()
        if not documents:
            return False
        
        # # Step 4: Split documents
        # splits = self.split_documents(documents)
        # if not splits:
        #     return False
        
        # # Step 5: Embed and store documents
        # if not self.embed_and_store_documents(splits):
        #     return False
        
        # # # Step 6: Save pipeline state
        # if self.config.get('output', {}).get('save_splits', False):
        #     output_dir = self.config.get('output', {}).get('output_directory', 'data/processed')
        #     self.save_pipeline_state(splits, output_dir)
        
        print("=" * 50)
        print("✅ Pipeline completed successfully!")
        # print(f"📊 Total documents processed: {len(splits)}")
        print("🔍 You can now search documents using search_documents(query)")
        
        return True
