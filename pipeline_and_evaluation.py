"""
Combined Pipeline + Evaluation Script
Runs the RAG pipeline and immediately evaluates it while vector store is in memory
"""

from godot_rag_pipeline import GodotRAGPipeline
from evaluation_framework import GodotEvaluationSuite

def run_pipeline_and_evaluate():
    """Run pipeline and immediately evaluate while vector store is in memory"""
    
    print("🚀 Combined Pipeline + Evaluation Run")
    print("=" * 60)
    
    # Initialize and run pipeline
    print("1️⃣ Initializing and running RAG pipeline...")
    pipeline = GodotRAGPipeline("config.yaml")
    
    if not pipeline.run_pipeline():
        print("❌ Pipeline failed")
        return
    
    print("\n" + "=" * 60)
    print("2️⃣ Running evaluation with populated vector store...")
    
    # Initialize evaluation suite with the same pipeline instance
    evaluator = GodotEvaluationSuite(pipeline)
    
    # Quick test to confirm vector store has content
    test_results = pipeline.search_documents("How to create a scene", k=1)
    print(f"✅ Vector store test: {len(test_results)} documents found")
    
    if test_results:
        print(f"📖 Sample result: {test_results[0].page_content[:100]}...")
    
    # Create smaller evaluation set for faster results
    from evaluation_framework import EvaluationQuery
    
    quick_queries = [
        EvaluationQuery(
            question="How do I create a new scene in Godot?",
            expected_docs=["scene", "new", "create"],
            category="scene_creation",
            difficulty="basic"
        ),
        EvaluationQuery(
            question="How do I attach a script to a node?",
            expected_docs=["script", "node", "attach"],
            category="scripting", 
            difficulty="basic"
        ),
        EvaluationQuery(
            question="How to make a character move in 2D?",
            expected_docs=["2d", "movement", "character"],
            category="physics",
            difficulty="intermediate"
        ),
        EvaluationQuery(
            question="How do I create a button and connect its signal?",
            expected_docs=["button", "signal", "connect"],
            category="ui",
            difficulty="basic"
        ),
        EvaluationQuery(
            question="How do I create custom shader materials?",
            expected_docs=["shader", "material", "custom"],
            category="rendering",
            difficulty="advanced"
        )
    ]
    
    # Override evaluation queries
    evaluator.evaluation_queries = quick_queries
    
    print("\n3️⃣ Running Retrieval Evaluation...")
    retrieval_metrics = evaluator.evaluate_retrieval(k_values=[1, 3, 5])
    
    print("\n4️⃣ Running RAG Flow Evaluation...")
    rag_metrics = evaluator.evaluate_rag_flow()
    
    # Final summary comparing to your example
    print("\n" + "=" * 60)
    print("📊 FINAL EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"🔍 Retrieval Performance:")
    print(f"   Hit Rate @ 5: {retrieval_metrics.hit_rate:.1f}%")
    print(f"   MRR: {retrieval_metrics.mrr:.1f}%")
    print(f"   Total queries: {retrieval_metrics.total_queries}")
    
    print(f"\n🤖 RAG Flow Performance:")
    print(f"   Relevant: {rag_metrics.relevant_count}/{rag_metrics.total_queries} ({rag_metrics.relevant_percentage:.1f}%)")
    print(f"   Partly Relevant: {rag_metrics.partly_relevant_count}/{rag_metrics.total_queries} ({rag_metrics.partly_relevant_percentage:.1f}%)")
    print(f"   Non-Relevant: {rag_metrics.non_relevant_count}/{rag_metrics.total_queries} ({rag_metrics.non_relevant_percentage:.1f}%)")
    
    # Compare to your reference example
    print(f"\n📋 Comparison to Reference Example:")
    print(f"   Reference Hit Rate: 94% → Your System: {retrieval_metrics.hit_rate:.1f}%")
    print(f"   Reference MRR: 82-90% → Your System: {retrieval_metrics.mrr:.1f}%")
    print(f"   Reference RAG Quality: 83-84% → Your System: {rag_metrics.relevant_percentage:.1f}%")
    
    # Assessment
    print(f"\n💡 Performance Assessment:")
    
    if retrieval_metrics.hit_rate >= 90 and rag_metrics.relevant_percentage >= 80:
        print("   🏆 EXCELLENT - Your system matches or exceeds reference performance!")
    elif retrieval_metrics.hit_rate >= 70 and rag_metrics.relevant_percentage >= 70:
        print("   ✅ GOOD - Your system performs well, close to reference levels")
    elif retrieval_metrics.hit_rate >= 50 and rag_metrics.relevant_percentage >= 60:
        print("   ⚠️  FAIR - Your system needs optimization to match reference performance")
    else:
        print("   ❌ NEEDS WORK - Significant improvements needed")
    
    # Specific recommendations
    if retrieval_metrics.hit_rate < 80:
        print("   🔧 Retrieval improvements needed:")
        print("      - Consider tuning embedding model parameters")
        print("      - Review document chunking strategy")
        print("      - Experiment with different similarity thresholds")
    
    if rag_metrics.relevant_percentage < 80:
        print("   🔧 Generation improvements needed:")
        print("      - Optimize prompt template for Godot-specific queries")
        print("      - Adjust LLM temperature and parameters")
        print("      - Review context window and retrieval count")
    
    print(f"\n🎉 Evaluation Complete!")
    return {
        "retrieval": retrieval_metrics,
        "rag_flow": rag_metrics
    }


if __name__ == "__main__":
    results = run_pipeline_and_evaluate()
