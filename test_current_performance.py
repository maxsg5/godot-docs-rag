#!/usr/bin/env python3
"""
Simple evaluation script for testing RAG improvements
"""

def test_problematic_queries():
    """Test the queries that previously performed poorly"""
    test_queries = [
        "How to create a scene in Godot?",
        "How to add nodes to a scene?", 
        "How to attach a script to a node?",
        "What is the difference between _ready() and _init()?",
        "How to detect collisions in Godot?"
    ]
    
    print("🧪 Testing problematic queries...")
    print("="*50)
    
    try:
        from godot_rag_pipeline import GodotRAGPipeline
        pipeline = GodotRAGPipeline()
        
        if not pipeline.vector_store:
            print("❌ Vector store not initialized")
            print("💡 Run: python godot_rag_pipeline.py")
            return
            
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            results = pipeline.search_documents(query, k=3)
            
            if results:
                for j, doc in enumerate(results, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    content_preview = doc.page_content[:120].replace('\n', ' ')
                    print(f"   {j}. {content_preview}...")
                    print(f"      Source: {source}")
            else:
                print("   ❌ No results found")
                
            print("-" * 40)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_problematic_queries()
