"""
Main FastAPI application for Godot RAG System
Provides REST API endpoints for the RAG functionality
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag_system import ComprehensiveRAGSystem
from src.config import RAGConfig
from src.monitoring import MetricsCollector

# Initialize FastAPI app
app = FastAPI(
    title="Godot Documentation RAG API",
    description="Advanced RAG system for Godot game engine documentation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = RAGConfig.from_env()
rag_system = None
metrics_collector = MetricsCollector()
logger = logging.getLogger("godot_rag.api")

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    method: str = "hybrid"
    prompt_type: str = "default"
    use_reranking: bool = True
    use_query_rewriting: bool = True
    max_documents: int = 5

class QueryResponse(BaseModel):
    query: str
    answer: str
    method: str
    processing_time: float
    retrieved_documents: List[Dict[str, Any]]
    confidence_score: float
    metadata: Dict[str, Any]

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    rating: int  # 1-5 scale
    feedback_text: Optional[str] = None
    method_used: str
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]
    version: str

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    
    logger.info("🚀 Starting Godot RAG API server...")
    
    try:
        # Initialize RAG system
        rag_system = ComprehensiveRAGSystem(config)
        await rag_system.initialize()
        
        logger.info("✅ RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {
        "rag_system": "healthy" if rag_system and rag_system.is_ready() else "unhealthy",
        "vector_store": "healthy" if rag_system and rag_system.vector_store_ready() else "unhealthy",
        "llm": "healthy" if rag_system and rag_system.llm_ready() else "unhealthy"
    }
    
    overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        components=components,
        version="1.0.0"
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, background_tasks: BackgroundTasks):
    """Query the RAG system"""
    if not rag_system or not rag_system.is_ready():
        raise HTTPException(status_code=503, detail="RAG system not ready")
    
    try:
        # Process the query
        result = await rag_system.answer_question(
            query=request.query,
            method=request.method,
            prompt_type=request.prompt_type,
            use_reranking=request.use_reranking,
            use_query_rewriting=request.use_query_rewriting,
            max_documents=request.max_documents
        )
        
        # Calculate confidence score based on retrieval scores
        confidence_score = 0.0
        if result["retrieved_documents"]:
            avg_score = sum(doc.get("score", 0) for doc in result["retrieved_documents"]) / len(result["retrieved_documents"])
            confidence_score = min(avg_score * 100, 100)  # Convert to percentage, cap at 100
        
        # Record metrics in background
        background_tasks.add_task(
            metrics_collector.record_query,
            request.query,
            request.method,
            result["processing_time"],
            confidence_score
        )
        
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            method=result["method"],
            processing_time=result["processing_time"],
            retrieved_documents=result["retrieved_documents"],
            confidence_score=confidence_score,
            metadata={
                "context_length": result.get("context_length", 0),
                "num_documents": result.get("num_documents", 0),
                "prompt_type": result.get("prompt_type", "default")
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Submit user feedback"""
    try:
        # Record feedback in background
        background_tasks.add_task(
            metrics_collector.record_feedback,
            request.query,
            request.answer,
            request.rating,
            request.feedback_text,
            request.method_used,
            request.processing_time
        )
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        logger.error(f"❌ Feedback recording failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        metrics = await metrics_collector.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"❌ Metrics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@app.get("/methods")
async def get_available_methods():
    """Get available retrieval methods"""
    return {
        "methods": ["vector", "bm25", "tfidf", "hybrid"],
        "prompt_types": ["default", "detailed", "beginner"],
        "default_method": "hybrid",
        "default_prompt_type": "default"
    }

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    if not rag_system or not rag_system.is_ready():
        raise HTTPException(status_code=503, detail="RAG system not ready")
    
    try:
        stats = await rag_system.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"❌ Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve stats")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
