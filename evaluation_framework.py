"""
Comprehensive Evaluation Framework for Godot RAG Pipeline
Implements both retrieval evaluation and RAG flow evaluation metrics
"""

import json
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain_ollama import OllamaLLM
from godot_rag_pipeline import GodotRAGPipeline


@dataclass
class EvaluationQuery:
    """Structure for evaluation queries with ground truth"""
    question: str
    expected_docs: List[str]  # List of document sources that should be relevant
    category: str  # e.g., "scene_creation", "scripting", "rendering"
    difficulty: str  # "basic", "intermediate", "advanced"


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics"""
    hit_rate: float
    mrr: float  # Mean Reciprocal Rank
    precision_at_k: List[float]  # Precision at different k values
    total_queries: int


@dataclass
class RAGFlowMetrics:
    """RAG flow evaluation metrics"""
    relevant_count: int
    partly_relevant_count: int
    non_relevant_count: int
    total_queries: int
    
    @property
    def relevant_percentage(self) -> float:
        return (self.relevant_count / self.total_queries) * 100
    
    @property
    def partly_relevant_percentage(self) -> float:
        return (self.partly_relevant_count / self.total_queries) * 100
    
    @property
    def non_relevant_percentage(self) -> float:
        return (self.non_relevant_count / self.total_queries) * 100


class GodotEvaluationSuite:
    """Comprehensive evaluation suite for Godot RAG pipeline"""
    
    def __init__(self, pipeline: GodotRAGPipeline, judge_model: str = "llama3"):
        self.pipeline = pipeline
        self.judge_llm = OllamaLLM(
            model=judge_model,
            temperature=0.1,
            base_url=pipeline.config.get('embedding', {}).get('base_url', "http://localhost:11434")
        )
        self.evaluation_queries = []
        
    def load_evaluation_dataset(self) -> List[EvaluationQuery]:
        """Load or create evaluation dataset for Godot documentation"""
        
        # Create a comprehensive set of Godot-specific evaluation queries
        queries = [
            # Scene Management
            EvaluationQuery(
                question="How do I create a new scene in Godot?",
                expected_docs=["getting_started", "scene", "tutorial"],
                category="scene_creation",
                difficulty="basic"
            ),
            EvaluationQuery(
                question="How to instance a scene as a child of another scene?",
                expected_docs=["scene", "node", "instantiate"],
                category="scene_creation", 
                difficulty="intermediate"
            ),
            
            # Scripting
            EvaluationQuery(
                question="How do I attach a script to a node in Godot?",
                expected_docs=["scripting", "node", "gdscript"],
                category="scripting",
                difficulty="basic"
            ),
            EvaluationQuery(
                question="What is the difference between _ready() and _init() in GDScript?",
                expected_docs=["gdscript", "scripting", "lifecycle"],
                category="scripting",
                difficulty="intermediate"
            ),
            
            # Physics and Movement
            EvaluationQuery(
                question="How do I make a character move in 2D using physics?",
                expected_docs=["physics", "2d", "character", "movement"],
                category="physics",
                difficulty="intermediate"
            ),
            EvaluationQuery(
                question="How to detect collisions between two rigid bodies?",
                expected_docs=["physics", "collision", "rigid", "body"],
                category="physics",
                difficulty="intermediate"
            ),
            
            # UI/Controls
            EvaluationQuery(
                question="How do I create a button and connect its signal?",
                expected_docs=["ui", "button", "signal", "control"],
                category="ui",
                difficulty="basic"
            ),
            EvaluationQuery(
                question="How to create a responsive UI that adapts to different screen sizes?",
                expected_docs=["ui", "responsive", "screen", "viewport"],
                category="ui",
                difficulty="advanced"
            ),
            
            # Animation
            EvaluationQuery(
                question="How do I create sprite animations using AnimationPlayer?",
                expected_docs=["animation", "sprite", "player"],
                category="animation",
                difficulty="intermediate"
            ),
            EvaluationQuery(
                question="How to create smooth transitions between animation states?",
                expected_docs=["animation", "transition", "state"],
                category="animation",
                difficulty="advanced"
            ),
            
            # Rendering/Shaders
            EvaluationQuery(
                question="How do I create a custom shader material?",
                expected_docs=["shader", "material", "rendering"],
                category="rendering",
                difficulty="advanced"
            ),
            EvaluationQuery(
                question="How to optimize rendering performance in Godot?",
                expected_docs=["performance", "rendering", "optimization"],
                category="rendering",
                difficulty="advanced"
            ),
            
            # Audio
            EvaluationQuery(
                question="How do I play background music in my game?",
                expected_docs=["audio", "music", "sound"],
                category="audio",
                difficulty="basic"
            ),
            
            # File System/Resources
            EvaluationQuery(
                question="How do I save and load game data in Godot?",
                expected_docs=["save", "load", "file", "data"],
                category="data",
                difficulty="intermediate"
            ),
            EvaluationQuery(
                question="What are the different resource formats in Godot?",
                expected_docs=["resource", "format", "file"],
                category="data",
                difficulty="intermediate"
            ),
        ]
        
        self.evaluation_queries = queries
        return queries
    
    def evaluate_retrieval(self, k_values: List[int] = [1, 3, 5, 10]) -> RetrievalMetrics:
        """Evaluate retrieval performance using Hit Rate and MRR with semantic relevance"""
        
        if not self.evaluation_queries:
            self.load_evaluation_dataset()
        
        print("🔍 Starting Retrieval Evaluation...")
        print(f"📊 Evaluating {len(self.evaluation_queries)} queries")
        
        hits_at_k = {k: 0 for k in k_values}
        reciprocal_ranks = []
        precision_scores = {k: [] for k in k_values}
        
        # Create a relevance judge for retrieval evaluation
        relevance_prompt = """You are evaluating whether a retrieved document is relevant to answer a specific question about Godot Engine.

Question: {question}

Document Content: {content}

Document Source: {source}

Is this document relevant for answering the question? Consider:
- Does it contain information that directly helps answer the question?
- Would a developer find this useful for the specific topic?
- Is it related to the core concepts in the question?

Respond with only: RELEVANT or NOT_RELEVANT"""
        
        for i, query in enumerate(self.evaluation_queries):
            print(f"🔄 Processing query {i+1}/{len(self.evaluation_queries)}: {query.question[:50]}...")
            
            # Get retrieval results
            results = self.pipeline.search_documents(query.question, k=max(k_values))
            
            # Evaluate each document for relevance using LLM judge
            relevant_docs = []
            for rank, doc in enumerate(results, 1):
                try:
                    # Use LLM to judge relevance
                    judge_input = relevance_prompt.format(
                        question=query.question,
                        content=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        source=doc.metadata.get('source', 'Unknown')
                    )
                    
                    relevance_response = self.judge_llm.invoke(judge_input).strip().upper()
                    is_relevant = "RELEVANT" in relevance_response and "NOT_RELEVANT" not in relevance_response
                    
                    if is_relevant:
                        relevant_docs.append(rank)
                    
                except Exception as e:
                    print(f"    ⚠️ Error judging document {rank}: {e}")
                    # Fallback to keyword matching
                    is_relevant = any(keyword.lower() in doc.metadata.get('source', '').lower() or
                                    keyword.lower() in doc.page_content.lower()
                                    for keyword in query.expected_docs)
                    if is_relevant:
                        relevant_docs.append(rank)
            
            # Calculate hit rate and MRR based on LLM-judged relevance
            found_rank = relevant_docs[0] if relevant_docs else None
            
            # Update hit rates for different k values
            for k in k_values:
                if any(rank <= k for rank in relevant_docs):
                    hits_at_k[k] += 1
                
                # Calculate precision@k
                relevant_in_k = sum(1 for rank in relevant_docs if rank <= k)
                precision_at_k = relevant_in_k / k if k > 0 else 0
                precision_scores[k].append(precision_at_k)
            
            # Add reciprocal rank
            if found_rank:
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)
            
            print(f"    ✅ Found {len(relevant_docs)} relevant docs in top {max(k_values)}")
        
        # Calculate final metrics
        total_queries = len(self.evaluation_queries)
        hit_rates = {k: (hits_at_k[k] / total_queries) * 100 for k in k_values}
        mrr = np.mean(reciprocal_ranks) * 100
        avg_precision = [np.mean(precision_scores[k]) * 100 for k in k_values]
        
        print("\n📊 Retrieval Evaluation Results:")
        print("=" * 50)
        for k in k_values:
            print(f"Hit Rate @ {k}: {hit_rates[k]:.1f}%")
        print(f"MRR (Mean Reciprocal Rank): {mrr:.1f}%")
        for i, k in enumerate(k_values):
            print(f"Precision @ {k}: {avg_precision[i]:.1f}%")
        
        return RetrievalMetrics(
            hit_rate=hit_rates[5],  # Use hit rate @ 5 as primary metric
            mrr=mrr,
            precision_at_k=avg_precision,
            total_queries=total_queries
        )
    
    def evaluate_rag_flow(self, sample_size: int = None) -> RAGFlowMetrics:
        """Evaluate RAG flow using LLM-as-a-Judge"""
        
        if not self.evaluation_queries:
            self.load_evaluation_dataset()
        
        queries_to_evaluate = self.evaluation_queries
        if sample_size and sample_size < len(queries_to_evaluate):
            queries_to_evaluate = queries_to_evaluate[:sample_size]
        
        print(f"🤖 Starting RAG Flow Evaluation with LLM-as-a-Judge...")
        print(f"📊 Evaluating {len(queries_to_evaluate)} queries")
        
        relevant_count = 0
        partly_relevant_count = 0
        non_relevant_count = 0
        
        judge_prompt_template = """You are an expert evaluator for a Godot Engine documentation RAG system. 

Your task is to evaluate whether the generated answer is relevant to the question based on the provided context from Godot documentation.

Rate the answer as:
- RELEVANT: The answer directly addresses the question with accurate, helpful information
- PARTLY_RELEVANT: The answer provides some useful information but may be incomplete or tangentially related
- NON_RELEVANT: The answer does not address the question or provides incorrect information

Question: {question}

Generated Answer: {answer}

Context Used: {context}

Evaluation (respond with only RELEVANT, PARTLY_RELEVANT, or NON_RELEVANT):"""
        
        for i, query in enumerate(queries_to_evaluate):
            print(f"🔄 Evaluating query {i+1}/{len(queries_to_evaluate)}: {query.question[:50]}...")
            
            # Generate answer using RAG
            result = self.pipeline.generate_answer(query.question)
            answer = result['answer']
            sources = result.get('source_documents', [])
            
            # Prepare context for judge
            context_text = "\n".join([f"Source {i+1}: {doc['content']}" 
                                    for i, doc in enumerate(sources[:3])])
            
            # Get LLM judge evaluation
            judge_prompt = judge_prompt_template.format(
                question=query.question,
                answer=answer,
                context=context_text
            )
            
            try:
                judge_response = self.judge_llm.invoke(judge_prompt).strip().upper()
                
                if "RELEVANT" in judge_response and "PARTLY" not in judge_response:
                    relevant_count += 1
                    result_type = "RELEVANT"
                elif "PARTLY_RELEVANT" in judge_response:
                    partly_relevant_count += 1
                    result_type = "PARTLY_RELEVANT"
                else:
                    non_relevant_count += 1
                    result_type = "NON_RELEVANT"
                
                print(f"  → {result_type}")
                
            except Exception as e:
                print(f"  → ERROR: {e}")
                non_relevant_count += 1  # Count errors as non-relevant
            
            # Small delay to avoid overwhelming the LLM
            time.sleep(0.5)
        
        total_evaluated = len(queries_to_evaluate)
        
        print("\n🤖 RAG Flow Evaluation Results:")
        print("=" * 50)
        print(f"RELEVANT: {relevant_count} ({(relevant_count/total_evaluated)*100:.1f}%)")
        print(f"PARTLY_RELEVANT: {partly_relevant_count} ({(partly_relevant_count/total_evaluated)*100:.1f}%)")
        print(f"NON_RELEVANT: {non_relevant_count} ({(non_relevant_count/total_evaluated)*100:.1f}%)")
        
        return RAGFlowMetrics(
            relevant_count=relevant_count,
            partly_relevant_count=partly_relevant_count,
            non_relevant_count=non_relevant_count,
            total_queries=total_evaluated
        )
    
    def run_full_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """Run both retrieval and RAG flow evaluation"""
        
        print("🚀 Starting Full Evaluation Suite")
        print("=" * 60)
        
        # Run retrieval evaluation
        retrieval_metrics = self.evaluate_retrieval()
        
        print("\n" + "=" * 60)
        
        # Run RAG flow evaluation  
        rag_metrics = self.evaluate_rag_flow()
        
        # Compile results
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "retrieval_evaluation": {
                "hit_rate": retrieval_metrics.hit_rate,
                "mrr": retrieval_metrics.mrr,
                "precision_at_k": retrieval_metrics.precision_at_k,
                "total_queries": retrieval_metrics.total_queries
            },
            "rag_flow_evaluation": {
                "relevant_percentage": rag_metrics.relevant_percentage,
                "partly_relevant_percentage": rag_metrics.partly_relevant_percentage,
                "non_relevant_percentage": rag_metrics.non_relevant_percentage,
                "total_queries": rag_metrics.total_queries,
                "counts": {
                    "relevant": rag_metrics.relevant_count,
                    "partly_relevant": rag_metrics.partly_relevant_count,
                    "non_relevant": rag_metrics.non_relevant_count
                }
            },
            "summary": {
                "overall_quality": "Good" if rag_metrics.relevant_percentage > 70 else 
                                 "Fair" if rag_metrics.relevant_percentage > 50 else "Needs Improvement",
                "retrieval_quality": "Good" if retrieval_metrics.hit_rate > 80 else
                                   "Fair" if retrieval_metrics.hit_rate > 60 else "Needs Improvement"
            }
        }
        
        if save_results:
            output_file = f"evaluation_results_{int(time.time())}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {output_file}")
        
        print("\n🎯 Evaluation Complete!")
        print("=" * 60)
        
        return results


if __name__ == "__main__":
    # Initialize pipeline and run evaluation
    print("🚀 Initializing Godot RAG Evaluation Suite...")
    
    pipeline = GodotRAGPipeline("config.yaml")
    
    # Initialize components first
    if not pipeline.initialize_components():
        print("❌ Failed to initialize pipeline components")
        exit(1)
    
    # Check if vector store is populated by trying a search
    test_results = pipeline.search_documents("test query", k=1)
    if not test_results:
        print("❌ Vector store appears to be empty. Running pipeline to populate...")
        if not pipeline.run_pipeline():
            print("❌ Pipeline failed to populate vector store")
            exit(1)
        print("✅ Pipeline completed. Vector store is now populated.")
    else:
        print(f"✅ Vector store found with content (test returned {len(test_results)} documents)")
    
    # Initialize evaluation suite
    evaluator = GodotEvaluationSuite(pipeline)
    
    # Run full evaluation
    results = evaluator.run_full_evaluation()
    
    print("\n📊 Final Summary:")
    print(f"Retrieval Quality: {results['summary']['retrieval_quality']}")
    print(f"Overall RAG Quality: {results['summary']['overall_quality']}")
