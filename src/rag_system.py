"""
Comprehensive RAG System Implementation
Combines all retrieval methods and LLM integration
"""

import logging
import asyncio
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .config import RAGConfig


class ComprehensiveRAGSystem:
    """Complete RAG system with multiple retrieval methods"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger("godot_rag.system")
        
        # Component initialization flags
        self._llm_ready = False
        self._embeddings_ready = False
        self._vector_store_ready = False
        self._retrievers_ready = False
        
        # Core components
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.processed_chunks = []
        
        # Retrieval components
        self.bm25_retriever = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Prompt templates
        self.prompt_templates = {
            "default": ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant for Godot game engine documentation. Answer questions based on the provided context clearly and concisely."),
                ("human", "Context: {context}\n\nQuestion: {question}\n\nAnswer:")
            ]),
            "detailed": ChatPromptTemplate.from_messages([
                ("system", "You are an expert Godot game engine developer. Provide detailed, technical answers with code examples when relevant. Include step-by-step instructions when appropriate."),
                ("human", "Context: {context}\n\nQuestion: {question}\n\nProvide a comprehensive answer with examples if applicable:")
            ]),
            "beginner": ChatPromptTemplate.from_messages([
                ("system", "You are a friendly tutor for beginners learning Godot. Explain concepts clearly and simply, avoiding technical jargon. Use analogies and examples to make concepts easier to understand."),
                ("human", "Context: {context}\n\nQuestion: {question}\n\nExplain this in beginner-friendly terms:")
            ])
        }
    
    async def initialize(self):
        """Initialize all components of the RAG system"""
        self.logger.info("🔄 Initializing RAG system components...")
        
        # Create directories
        self.config.create_directories()
        
        # Initialize components
        await self._initialize_llm()
        await self._initialize_embeddings()
        await self._initialize_vector_store()
        await self._load_or_process_documents()
        await self._initialize_retrievers()
        
        self.logger.info("✅ RAG system initialization completed")
    
    async def _initialize_llm(self):
        """Initialize the LLM"""
        if not OLLAMA_AVAILABLE:
            self.logger.error("❌ Ollama not available. Install with: pip install langchain-ollama")
            return
        
        try:
            self.llm = ChatOllama(
                base_url=self.config.ollama_base_url,
                model=self.config.ollama_model,
                temperature=0.1
            )
            
            # Test the connection
            test_response = await self.llm.ainvoke("Hello")
            if test_response:
                self._llm_ready = True
                self.logger.info(f"✅ LLM initialized: {self.config.ollama_model}")
            else:
                self.logger.error("❌ LLM test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize LLM: {e}")
    
    async def _initialize_embeddings(self):
        """Initialize embeddings model"""
        if not OLLAMA_AVAILABLE:
            self.logger.error("❌ Ollama embeddings not available")
            return
        
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=self.config.ollama_base_url,
                model=self.config.embedding_model
            )
            
            # Test embeddings
            test_embedding = await self.embeddings.aembed_query("test")
            if test_embedding and len(test_embedding) > 0:
                self._embeddings_ready = True
                self.logger.info(f"✅ Embeddings initialized: {self.config.embedding_model}")
            else:
                self.logger.error("❌ Embeddings test failed")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize embeddings: {e}")
    
    async def _initialize_vector_store(self):
        """Initialize vector store"""
        if not QDRANT_AVAILABLE:
            self.logger.warning("⚠️ Qdrant not available. Using in-memory store")
            return
        
        try:
            # Try to connect to Qdrant
            client = QdrantClient(url=self.config.qdrant_url)
            
            # Check if collection exists
            collections = client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.config.qdrant_collection not in collection_names:
                # Create collection
                client.create_collection(
                    collection_name=self.config.qdrant_collection,
                    vectors_config=VectorParams(
                        size=self.config.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                self.logger.info(f"✅ Created Qdrant collection: {self.config.qdrant_collection}")
            
            self.vector_store_client = client
            self._vector_store_ready = True
            self.logger.info("✅ Vector store initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize vector store: {e}")
    
    async def _load_or_process_documents(self):
        """Load existing processed documents or process new ones"""
        processed_file = Path(self.config.processed_data_dir) / "processed_chunks.json"
        
        if processed_file.exists():
            self.logger.info("📥 Loading existing processed documents...")
            await self._load_processed_documents(processed_file)
        else:
            self.logger.info("🔄 Processing documents from scratch...")
            await self._process_documents_from_scratch()
    
    async def _load_processed_documents(self, filepath: Path):
        """Load pre-processed documents"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.processed_chunks = []
            for item in data:
                chunk = ProcessedChunk(
                    content=item['content'],
                    metadata=item['metadata'],
                    chunk_id=item['chunk_id'],
                    parent_doc_id=item['parent_doc_id'],
                    chunk_index=item['chunk_index'],
                    tokens=item['tokens']
                )
                self.processed_chunks.append(chunk)
            
            self.logger.info(f"✅ Loaded {len(self.processed_chunks)} processed chunks")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load processed documents: {e}")
            await self._process_documents_from_scratch()
    
    async def _process_documents_from_scratch(self):
        """Process documents from web sources"""
        # Sample Godot documentation URLs for demonstration
        sample_urls = [
            "https://docs.godotengine.org/en/stable/getting_started/introduction/index.html",
            "https://docs.godotengine.org/en/stable/getting_started/scripting/gdscript/index.html",
            "https://docs.godotengine.org/en/stable/tutorials/scripting/nodes_and_scene_instances.html",
            "https://docs.godotengine.org/en/stable/tutorials/2d/physics_introduction.html",
            "https://docs.godotengine.org/en/stable/tutorials/animation/introduction.html"
        ]
        
        # Load documents
        documents = await self._load_web_documents(sample_urls)
        
        # Process into chunks
        self.processed_chunks = await self._process_documents(documents)
        
        # Save processed chunks
        await self._save_processed_documents()
    
    async def _load_web_documents(self, urls: List[str]) -> List[Document]:
        """Load documents from web URLs"""
        try:
            loader = WebBaseLoader(web_paths=urls)
            documents = []
            
            for doc in loader.lazy_load():
                if len(doc.page_content.strip()) > 100:  # Filter short documents
                    documents.append(doc)
            
            self.logger.info(f"✅ Loaded {len(documents)} web documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load web documents: {e}")
            return []
    
    async def _process_documents(self, documents: List[Document]) -> List['ProcessedChunk']:
        """Process documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
        )
        
        processed_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            try:
                # Split document
                chunks = text_splitter.split_documents([doc])
                
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                    
                    processed_chunk = ProcessedChunk(
                        content=chunk.page_content,
                        metadata=chunk.metadata,
                        chunk_id=chunk_id,
                        parent_doc_id=f"doc_{doc_idx}",
                        chunk_index=chunk_idx,
                        tokens=len(chunk.page_content.split())
                    )
                    
                    processed_chunks.append(processed_chunk)
                    
            except Exception as e:
                self.logger.error(f"Error processing document {doc_idx}: {e}")
                continue
        
        self.logger.info(f"✅ Processed {len(processed_chunks)} chunks")
        return processed_chunks
    
    async def _save_processed_documents(self):
        """Save processed documents to disk"""
        try:
            processed_file = Path(self.config.processed_data_dir) / "processed_chunks.json"
            
            data = []
            for chunk in self.processed_chunks:
                data.append({
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'chunk_id': chunk.chunk_id,
                    'parent_doc_id': chunk.parent_doc_id,
                    'chunk_index': chunk.chunk_index,
                    'tokens': chunk.tokens
                })
            
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Saved {len(data)} processed chunks")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save processed documents: {e}")
    
    async def _initialize_retrievers(self):
        """Initialize all retrieval methods"""
        if not self.processed_chunks:
            self.logger.error("❌ No processed chunks available for retrieval initialization")
            return
        
        # Initialize BM25
        if BM25_AVAILABLE:
            try:
                tokenized_docs = [chunk.content.lower().split() for chunk in self.processed_chunks]
                self.bm25_retriever = BM25Okapi(tokenized_docs)
                self.logger.info("✅ BM25 retriever initialized")
            except Exception as e:
                self.logger.error(f"❌ BM25 initialization failed: {e}")
        
        # Initialize TF-IDF
        if SKLEARN_AVAILABLE:
            try:
                documents = [chunk.content for chunk in self.processed_chunks]
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=5000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
                self.logger.info("✅ TF-IDF retriever initialized")
            except Exception as e:
                self.logger.error(f"❌ TF-IDF initialization failed: {e}")
        
        self._retrievers_ready = True
        self.logger.info("✅ All retrievers initialized")
    
    def is_ready(self) -> bool:
        """Check if the system is ready"""
        return (self._llm_ready and 
                self._embeddings_ready and 
                self._retrievers_ready and 
                len(self.processed_chunks) > 0)
    
    def llm_ready(self) -> bool:
        """Check if LLM is ready"""
        return self._llm_ready
    
    def vector_store_ready(self) -> bool:
        """Check if vector store is ready"""
        return self._vector_store_ready
    
    async def answer_question(self, query: str, method: str = "hybrid", 
                             prompt_type: str = "default", use_reranking: bool = True,
                             use_query_rewriting: bool = True, max_documents: int = 5) -> Dict[str, Any]:
        """Answer a question using the RAG system"""
        start_time = time.time()
        
        if not self.is_ready():
            raise RuntimeError("RAG system is not ready")
        
        try:
            # Retrieve relevant documents
            retrieved_docs = await self._retrieve_documents(query, method, max_documents)
            
            # Re-rank if requested
            if use_reranking and retrieved_docs:
                retrieved_docs = self._rerank_documents(query, retrieved_docs)
            
            # Generate context
            context = "\n\n".join([doc["content"] for doc in retrieved_docs[:max_documents]])
            
            # Generate answer
            answer = await self._generate_answer(query, context, prompt_type)
            
            processing_time = time.time() - start_time
            
            return {
                "query": query,
                "answer": answer,
                "retrieved_documents": retrieved_docs[:max_documents],
                "method": method,
                "prompt_type": prompt_type,
                "processing_time": processing_time,
                "context_length": len(context),
                "num_documents": len(retrieved_docs)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Question answering failed: {e}")
            raise
    
    async def _retrieve_documents(self, query: str, method: str, max_documents: int) -> List[Dict[str, Any]]:
        """Retrieve documents using specified method"""
        if method == "vector":
            return await self._retrieve_vector(query, max_documents)
        elif method == "bm25":
            return self._retrieve_bm25(query, max_documents)
        elif method == "tfidf":
            return self._retrieve_tfidf(query, max_documents)
        elif method == "hybrid":
            return await self._retrieve_hybrid(query, max_documents)
        else:
            # Default to hybrid
            return await self._retrieve_hybrid(query, max_documents)
    
    async def _retrieve_vector(self, query: str, max_documents: int) -> List[Dict[str, Any]]:
        """Vector similarity retrieval"""
        if not self._embeddings_ready:
            self.logger.warning("Embeddings not ready, falling back to BM25")
            return self._retrieve_bm25(query, max_documents)
        
        try:
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Simple similarity search (in production, use vector store)
            similarities = []
            for i, chunk in enumerate(self.processed_chunks):
                # Generate chunk embedding if not cached
                chunk_embedding = await self.embeddings.aembed_query(chunk.content)
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for i, score in similarities[:max_documents]:
                chunk = self.processed_chunks[i]
                results.append({
                    "content": chunk.content,
                    "score": float(score),
                    "method": "vector",
                    "chunk_id": chunk.chunk_id
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Vector retrieval failed: {e}")
            return []
    
    def _retrieve_bm25(self, query: str, max_documents: int) -> List[Dict[str, Any]]:
        """BM25 keyword retrieval"""
        if not self.bm25_retriever:
            return []
        
        try:
            query_tokens = query.lower().split()
            scores = self.bm25_retriever.get_scores(query_tokens)
            
            # Get top results
            top_indices = np.argsort(scores)[-max_documents:][::-1]
            
            results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    chunk = self.processed_chunks[idx]
                    results.append({
                        "content": chunk.content,
                        "score": float(scores[idx]),
                        "method": "bm25",
                        "chunk_id": chunk.chunk_id
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"BM25 retrieval failed: {e}")
            return []
    
    def _retrieve_tfidf(self, query: str, max_documents: int) -> List[Dict[str, Any]]:
        """TF-IDF retrieval"""
        if not self.tfidf_vectorizer or self.tfidf_matrix is None:
            return []
        
        try:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            top_indices = np.argsort(similarities)[-max_documents:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0:
                    chunk = self.processed_chunks[idx]
                    results.append({
                        "content": chunk.content,
                        "score": float(similarities[idx]),
                        "method": "tfidf",
                        "chunk_id": chunk.chunk_id
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"TF-IDF retrieval failed: {e}")
            return []
    
    async def _retrieve_hybrid(self, query: str, max_documents: int) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining multiple methods"""
        # Get results from different methods
        vector_results = await self._retrieve_vector(query, max_documents * 2)
        bm25_results = self._retrieve_bm25(query, max_documents * 2)
        
        # Combine and normalize scores
        combined_results = {}
        alpha = self.config.hybrid_alpha
        
        # Add vector results
        if vector_results:
            max_vector_score = max(r["score"] for r in vector_results)
            for result in vector_results:
                content = result["content"]
                normalized_score = result["score"] / max_vector_score if max_vector_score > 0 else 0
                combined_results[content] = {
                    "content": content,
                    "vector_score": normalized_score,
                    "bm25_score": 0,
                    "method": "hybrid",
                    "chunk_id": result["chunk_id"]
                }
        
        # Add BM25 results
        if bm25_results:
            max_bm25_score = max(r["score"] for r in bm25_results)
            for result in bm25_results:
                content = result["content"]
                normalized_score = result["score"] / max_bm25_score if max_bm25_score > 0 else 0
                if content in combined_results:
                    combined_results[content]["bm25_score"] = normalized_score
                else:
                    combined_results[content] = {
                        "content": content,
                        "vector_score": 0,
                        "bm25_score": normalized_score,
                        "method": "hybrid",
                        "chunk_id": result["chunk_id"]
                    }
        
        # Calculate hybrid scores
        for result in combined_results.values():
            result["score"] = (alpha * result["vector_score"] + 
                             (1 - alpha) * result["bm25_score"])
        
        # Sort and return top results
        sorted_results = sorted(combined_results.values(), 
                              key=lambda x: x["score"], reverse=True)
        
        return sorted_results[:max_documents]
    
    def _rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank documents based on query-document similarity"""
        query_terms = set(query.lower().split())
        
        for doc in documents:
            content_terms = set(doc["content"].lower().split())
            overlap = len(query_terms.intersection(content_terms))
            doc["rerank_score"] = overlap / len(query_terms) if query_terms else 0
            doc["final_score"] = 0.7 * doc["score"] + 0.3 * doc["rerank_score"]
        
        return sorted(documents, key=lambda x: x["final_score"], reverse=True)
    
    async def _generate_answer(self, query: str, context: str, prompt_type: str = "default") -> str:
        """Generate answer using LLM"""
        if not self._llm_ready:
            return "LLM not available. Please check the system configuration."
        
        try:
            prompt = self.prompt_templates.get(prompt_type, self.prompt_templates["default"])
            chain = prompt | self.llm | StrOutputParser()
            
            response = await chain.ainvoke({
                "context": context,
                "question": query
            })
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {e}"
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_chunks": len(self.processed_chunks),
            "avg_chunk_tokens": sum(chunk.tokens for chunk in self.processed_chunks) / len(self.processed_chunks) if self.processed_chunks else 0,
            "llm_ready": self._llm_ready,
            "embeddings_ready": self._embeddings_ready,
            "vector_store_ready": self._vector_store_ready,
            "retrievers_ready": self._retrievers_ready,
            "system_ready": self.is_ready(),
            "available_methods": ["vector", "bm25", "tfidf", "hybrid"],
            "available_prompt_types": list(self.prompt_templates.keys())
        }


class ProcessedChunk:
    """Processed document chunk"""
    def __init__(self, content: str, metadata: dict, chunk_id: str, 
                 parent_doc_id: str, chunk_index: int, tokens: int):
        self.content = content
        self.metadata = metadata
        self.chunk_id = chunk_id
        self.parent_doc_id = parent_doc_id
        self.chunk_index = chunk_index
        self.tokens = tokens
