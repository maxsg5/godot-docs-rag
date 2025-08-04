"""
Debug script to check vector store status and retrieval functionality
"""

from godot_rag_pipeline import GodotRAGPipeline

def debug_vector_store():
    """Debug vector store functionality"""
    
    print("🔍 Debugging Vector Store Status")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = GodotRAGPipeline("config.yaml")
    
    # Initialize components
    if not pipeline.initialize_components():
        print("❌ Failed to initialize components")
        return
    
    # Check vector store status
    print(f"\n📊 Vector Store Info:")
    print(f"   Type: {type(pipeline.vector_store)}")
    print(f"   Initialized: {pipeline.vector_store is not None}")
    
    if pipeline.vector_store:
        try:
            # Try to get document count (if supported)
            print(f"   Available methods: {[m for m in dir(pipeline.vector_store) if not m.startswith('_')]}")
            
            # Test embedding directly
            print(f"\n🧪 Testing embedding directly...")
            test_embedding = pipeline.embeddings.embed_query("test Godot scene creation")
            print(f"   Embedding size: {len(test_embedding)}")
            
            # Test similarity search
            print(f"\n🔍 Testing similarity search...")
            results = pipeline.vector_store.similarity_search("How to create a scene in Godot", k=3)
            print(f"   Search results: {len(results)} documents")
            
            if results:
                print(f"   First result preview:")
                first_doc = results[0]
                print(f"     Content: {first_doc.page_content[:200]}...")
                print(f"     Metadata: {first_doc.metadata}")
            else:
                print(f"   ❌ No documents returned from search")
                
                # Check if there are any documents in the store
                print(f"\n🔎 Investigating empty results...")
                
                # Try alternative search
                try:
                    all_results = pipeline.vector_store.similarity_search("", k=10)
                    print(f"   Empty query results: {len(all_results)} documents")
                except Exception as e:
                    print(f"   Empty query error: {e}")
                
                # Try with different queries
                test_queries = ["scene", "script", "node", "Godot", "tutorial"]
                for query in test_queries:
                    try:
                        query_results = pipeline.vector_store.similarity_search(query, k=1)
                        print(f"   Query '{query}': {len(query_results)} results")
                    except Exception as e:
                        print(f"   Query '{query}' error: {e}")
                        
        except Exception as e:
            print(f"   ❌ Error checking vector store: {e}")
    
    # Check if we need to rebuild the vector store
    print(f"\n💡 Diagnosis:")
    if not pipeline.vector_store:
        print("   ❌ Vector store not initialized")
    elif pipeline.search_documents("test", k=1):
        print("   ✅ Vector store working correctly")
    else:
        print("   ⚠️  Vector store initialized but empty or not returning results")
        print("   🔄 May need to run the full pipeline to populate documents")
        print("   Run: python godot_rag_pipeline.py")


if __name__ == "__main__":
    debug_vector_store()
