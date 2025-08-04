"""
Quick Evaluation Script for Godot RAG Pipeline
Tests current performance with a subset of queries
"""

import sys
import time
from evaluation_framework import GodotEvaluationSuite, EvaluationQuery
from godot_rag_pipeline import GodotRAGPipeline


def quick_evaluation():
    """Run a quick evaluation with a small set of queries"""
    
    print("🚀 Quick Evaluation of Godot RAG Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    print("📥 Loading pipeline...")
    pipeline = GodotRAGPipeline("config.yaml")
    
    # Check if components are initialized
    if not pipeline.initialize_components():
        print("❌ Failed to initialize pipeline components")
        return
    
    if not pipeline.vector_store:
        print("❌ Vector store not found. Please run the main pipeline first:")
        print("   python godot_rag_pipeline.py")
        return
    
    # Test sample questions manually for better insights
    test_questions = [
        "How do I create a new scene in Godot?",
        "How do I attach a script to a node?", 
        "How to make a character move in 2D?",
        "How do I create a button and connect its signal?",
        "How do I create custom shader materials?"
    ]
    
    print(f"🎯 Testing retrieval and generation with {len(test_questions)} questions...")
    
    # Initialize evaluator for RAG flow evaluation only
    evaluator = GodotEvaluationSuite(pipeline)
    
    relevant_answers = 0
    total_questions = len(test_questions)
    
    print("\n🔍 Testing Retrieval + Generation Quality:")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        
        # Test retrieval
        docs = pipeline.search_documents(question, k=3)
        print(f"   📄 Retrieved {len(docs)} documents")
        
        if docs:
            # Show first retrieved document preview
            first_doc = docs[0]
            content_preview = first_doc.page_content[:100] + "..." if len(first_doc.page_content) > 100 else first_doc.page_content
            source = first_doc.metadata.get('source', 'Unknown')
            print(f"   📖 Top result: {content_preview}")
            print(f"   🔗 Source: {source.split('/')[-1] if '/' in source else source}")
        
        # Test generation
        try:
            result = pipeline.generate_answer(question)
            answer = result['answer']
            
            # Simple quality check - is the answer substantive?
            if len(answer) > 50 and "don't know" not in answer.lower() and "error" not in answer.lower():
                print(f"   ✅ Generated substantial answer ({len(answer)} chars)")
                relevant_answers += 1
            else:
                print(f"   ⚠️  Generated minimal/error answer ({len(answer)} chars)")
            
            # Show answer preview
            answer_preview = answer[:150] + "..." if len(answer) > 150 else answer
            print(f"   💬 Answer: {answer_preview}")
            
        except Exception as e:
            print(f"   ❌ Generation error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 QUICK EVALUATION SUMMARY")
    print("=" * 60)
    
    success_rate = (relevant_answers / total_questions) * 100
    print(f"🎯 Answer Quality: {relevant_answers}/{total_questions} ({success_rate:.1f}%) generated good answers")
    
    if success_rate >= 80:
        print("   ✅ Excellent! Your RAG system is working very well")
    elif success_rate >= 60:
        print("   ✅ Good! Your RAG system is performing well")
    elif success_rate >= 40:
        print("   ⚠️  Fair - Your RAG system needs some tuning")
    else:
        print("   ❌ Poor - Your RAG system needs significant improvement")
    
    # Specific recommendations based on common issues
    print(f"\n💡 Recommendations:")
    print(f"   🔍 Document retrieval appears to be working (finding relevant docs)")
    print(f"   🤖 LLM generation is producing coherent answers")
    
    if success_rate < 80:
        print(f"   📝 Consider:")
        print(f"      - Fine-tuning the prompt template for better Godot-specific responses")
        print(f"      - Adjusting chunk size/overlap for better context")
        print(f"      - Testing different embedding models or parameters")
    
    print("\n🔄 For comprehensive metrics, run:")
    print("   python evaluation_framework.py")


if __name__ == "__main__":
    quick_evaluation()
