from godot_rag_pipeline import GodotRAGPipeline


def main():
    """Main entry point"""
    try:
        pipeline = GodotRAGPipeline()
        success = pipeline.run_pipeline()
        
        if success:
            print("sucess")
            # # Example search
            # print("\n🔍 Example search:")
            # results = pipeline.search_documents("How to create a scene in Godot?", k=3)
            # for i, doc in enumerate(results, 1):
            #     print(f"\nResult {i}:")
            #     print(f"Content: {doc.page_content[:200]}...")
            #     print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        else :
            print("womp")
            
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
