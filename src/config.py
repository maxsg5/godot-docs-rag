"""
Configuration management for Godot RAG System
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class RAGConfig:
    """Central configuration for the RAG system"""
    # LLM Configuration
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    
    # Embedding Configuration
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768
    
    # Vector Store Configuration
    vector_store_type: str = "qdrant"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "godot_docs"
    
    # Retrieval Configuration
    retrieval_methods: List[str] = None
    top_k_documents: int = 5
    similarity_threshold: float = 0.7
    hybrid_alpha: float = 0.7  # Weight for vector vs keyword search
    
    # Processing Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_documents: int = -1  # -1 for no limit
    
    # Evaluation Configuration
    eval_dataset_size: int = 100
    eval_metrics: List[str] = None
    
    # Data paths
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    embeddings_dir: str = "data/embeddings"
    evaluation_dir: str = "data/evaluation"
    
    def __post_init__(self):
        if self.retrieval_methods is None:
            self.retrieval_methods = ["vector", "keyword", "hybrid", "rerank"]
        if self.eval_metrics is None:
            self.eval_metrics = ["precision", "recall", "f1", "mrr", "ndcg"]
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Load configuration from environment variables"""
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "ollama"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "godot_docs"),
        )
    
    @classmethod
    def from_file(cls, filepath: str) -> 'RAGConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.embeddings_dir,
            self.evaluation_dir,
            "logs",
            "monitoring",
            "config",
            "models"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate configuration"""
        errors = []
        
        # Check required fields
        if self.llm_provider == "openai" and not self.openai_api_key:
            errors.append("OpenAI API key is required when using OpenAI provider")
        
        if self.chunk_size <= 0:
            errors.append("Chunk size must be positive")
        
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            errors.append("Chunk overlap must be between 0 and chunk_size")
        
        if self.top_k_documents <= 0:
            errors.append("top_k_documents must be positive")
        
        if not 0 <= self.hybrid_alpha <= 1:
            errors.append("hybrid_alpha must be between 0 and 1")
        
        if errors:
            print(f"❌ Configuration validation errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True


# Global configuration instance
_config = None

def get_config() -> RAGConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = RAGConfig.from_env()
    return _config

def set_config(config: RAGConfig):
    """Set the global configuration instance"""
    global _config
    _config = config
