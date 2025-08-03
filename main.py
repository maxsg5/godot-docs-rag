#!/usr/bin/env python3
"""
Main entry point for Godot Documentation RAG Pipeline
"""
from pipeline import GodotRAGPipeline

def main():
    """Run the complete RAG pipeline"""
    pipeline = GodotRAGPipeline()
    return pipeline.run_pipeline()

if __name__ == "__main__":
    main()
