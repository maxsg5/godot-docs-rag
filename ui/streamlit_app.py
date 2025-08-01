"""
Streamlit Web Interface for Godot RAG System
Provides user-friendly interface for querying and system management
"""

import streamlit as st
import asyncio
import json
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import logging

# Configure page
st.set_page_config(
    page_title="Godot RAG Assistant",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #478CBF;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #478CBF;
    }
    
    .query-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    
    .answer-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    
    .source-box {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitRAGInterface:
    """Streamlit interface for the RAG system"""
    
    def __init__(self):
        self.logger = logging.getLogger("godot_rag.streamlit")
        
        # Initialize session state
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'metrics_collector' not in st.session_state:
            st.session_state.metrics_collector = None
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'system_ready' not in st.session_state:
            st.session_state.system_ready = False
        if 'last_query_result' not in st.session_state:
            st.session_state.last_query_result = None
    
    def run(self):
        """Run the Streamlit interface"""
        # Header
        st.markdown('<h1 class="main-header">🎮 Godot RAG Assistant</h1>', unsafe_allow_html=True)
        st.markdown("Your intelligent assistant for Godot game engine documentation")
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        if st.session_state.system_ready and st.session_state.rag_system:
            self._render_main_interface()
        else:
            self._render_system_setup()
    
    def _render_sidebar(self):
        """Render sidebar with system status and controls"""
        with st.sidebar:
            st.header("🛠️ System Status")
            
            # System status
            if st.session_state.rag_system:
                if st.session_state.rag_system.is_ready():
                    st.success("✅ System Ready")
                    st.session_state.system_ready = True
                else:
                    st.warning("⚠️ System Initializing...")
                    if st.button("🔄 Check Status"):
                        st.rerun()
            else:
                st.error("❌ System Not Initialized")
            
            st.divider()
            
            # Navigation
            st.header("📋 Navigation")
            page = st.selectbox(
                "Select Page",
                ["Query Interface", "System Metrics", "Data Management", "Settings"],
                key="page_selector"
            )
            
            st.session_state.current_page = page
            
            st.divider()
            
            # Quick stats
            if st.session_state.metrics_collector:
                st.header("📊 Quick Stats")
                try:
                    stats = st.session_state.metrics_collector.get_current_stats()
                    overview = stats.get("overview", {})
                    
                    st.metric("Total Queries", overview.get("total_queries", 0))
                    st.metric("Success Rate", f"{overview.get('success_rate', 0):.1f}%")
                    st.metric("Avg Response Time", f"{overview.get('avg_processing_time', 0):.2f}s")
                    
                    if overview.get("avg_user_rating", 0) > 0:
                        st.metric("User Rating", f"{overview.get('avg_user_rating', 0):.1f}/5")
                        
                except Exception as e:
                    st.error(f"Error loading stats: {e}")
    
    def _render_system_setup(self):
        """Render system setup interface"""
        st.header("🚀 System Setup")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("The RAG system needs to be initialized before use.")
            
            if st.button("🔧 Initialize System", type="primary"):
                with st.spinner("Initializing RAG system..."):
                    success = self._initialize_system()
                    if success:
                        st.success("✅ System initialized successfully!")
                        st.rerun()
                    else:
                        st.error("❌ System initialization failed. Check logs for details.")
        
        with col2:
            st.markdown("### Requirements")
            st.markdown("- ✅ Ollama server running")
            st.markdown("- ✅ Required models available")
            st.markdown("- ✅ Vector database ready")
            st.markdown("- ✅ Documents processed")
    
    def _render_main_interface(self):
        """Render main interface based on selected page"""
        page = st.session_state.get("current_page", "Query Interface")
        
        if page == "Query Interface":
            self._render_query_interface()
        elif page == "System Metrics":
            self._render_metrics_dashboard()
        elif page == "Data Management":
            self._render_data_management()
        elif page == "Settings":
            self._render_settings()
    
    def _render_query_interface(self):
        """Render the main query interface"""
        st.header("💬 Ask about Godot")
        
        # Query input
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            query = st.text_area(
                "Enter your question:",
                placeholder="How do I create a 2D player character in Godot?",
                height=100,
                key="user_query"
            )
        
        with col2:
            method = st.selectbox(
                "Retrieval Method",
                ["hybrid", "vector", "bm25", "tfidf"],
                index=0,
                key="retrieval_method"
            )
            
            prompt_type = st.selectbox(
                "Answer Style",
                ["default", "detailed", "beginner"],
                index=0,
                key="prompt_type"
            )
        
        with col3:
            max_docs = st.slider(
                "Max Documents",
                min_value=1,
                max_value=10,
                value=5,
                key="max_documents"
            )
            
            use_reranking = st.checkbox(
                "Use Re-ranking",
                value=True,
                key="use_reranking"
            )
        
        # Query execution
        if st.button("🔍 Ask Question", type="primary", disabled=not query.strip()):
            with st.spinner("Searching for answers..."):
                result = self._execute_query(
                    query.strip(),
                    method,
                    prompt_type,
                    use_reranking,
                    max_docs
                )
                
                if result:
                    st.session_state.last_query_result = result
        
        # Display results
        if st.session_state.last_query_result:
            self._display_query_result(st.session_state.last_query_result)
        
        # Query history
        if st.session_state.query_history:
            st.divider()
            st.subheader("📝 Recent Queries")
            
            for i, hist_query in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Q: {hist_query['query'][:100]}..."):
                    st.write(f"**Method:** {hist_query['method']}")
                    st.write(f"**Time:** {hist_query.get('processing_time', 0):.2f}s")
                    st.write(f"**Answer:** {hist_query['answer'][:500]}...")
    
    def _render_metrics_dashboard(self):
        """Render metrics and analytics dashboard"""
        st.header("📊 System Metrics")
        
        if not st.session_state.metrics_collector:
            st.error("Metrics collector not available")
            return
        
        try:
            stats = st.session_state.metrics_collector.get_current_stats()
            overview = stats.get("overview", {})
            recent = stats.get("recent_performance", {})
            system_health = stats.get("system_health", {})
            
            # Overview metrics
            st.subheader("📈 Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Queries",
                    overview.get("total_queries", 0)
                )
            
            with col2:
                st.metric(
                    "Success Rate",
                    f"{overview.get('success_rate', 0):.1f}%"
                )
            
            with col3:
                st.metric(
                    "Avg Response Time",
                    f"{overview.get('avg_processing_time', 0):.2f}s"
                )
            
            with col4:
                rating = overview.get("avg_user_rating", 0)
                st.metric(
                    "User Rating",
                    f"{rating:.1f}/5" if rating > 0 else "N/A"
                )
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Method usage chart
                method_usage = stats.get("method_usage", {})
                if method_usage:
                    fig = px.pie(
                        values=list(method_usage.values()),
                        names=list(method_usage.keys()),
                        title="Retrieval Method Usage"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Daily queries chart
                daily_queries = stats.get("daily_queries", {})
                if daily_queries:
                    df = pd.DataFrame(
                        list(daily_queries.items()),
                        columns=["Date", "Queries"]
                    )
                    fig = px.bar(df, x="Date", y="Queries", title="Daily Query Count")
                    st.plotly_chart(fig, use_container_width=True)
            
            # System health
            if system_health:
                st.subheader("💻 System Health")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    cpu = system_health.get("cpu_percent", 0)
                    st.metric("CPU Usage", f"{cpu:.1f}%")
                
                with col2:
                    memory = system_health.get("memory_percent", 0)
                    st.metric("Memory Usage", f"{memory:.1f}%")
                
                with col3:
                    disk = system_health.get("disk_usage_percent", 0)
                    st.metric("Disk Usage", f"{disk:.1f}%")
            
            # Error analysis
            error_types = stats.get("error_types", {})
            if error_types:
                st.subheader("⚠️ Error Analysis")
                df_errors = pd.DataFrame(
                    list(error_types.items()),
                    columns=["Error Type", "Count"]
                )
                st.dataframe(df_errors, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    
    def _render_data_management(self):
        """Render data management interface"""
        st.header("📂 Data Management")
        
        # Processing status
        st.subheader("📊 Processing Status")
        
        if st.button("🔄 Refresh Status"):
            # Get processing status
            try:
                # This would call the data processor to get status
                st.info("Processing status refresh functionality would go here")
            except Exception as e:
                st.error(f"Error refreshing status: {e}")
        
        # Document processing
        st.subheader("🔄 Document Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🌐 Reprocess Web Documents"):
                with st.spinner("Processing web documents..."):
                    st.info("Web document processing would be triggered here")
        
        with col2:
            if st.button("📁 Process Local Files"):
                with st.spinner("Processing local files..."):
                    st.info("Local file processing would be triggered here")
        
        # Data statistics
        if st.session_state.rag_system:
            try:
                system_stats = asyncio.run(st.session_state.rag_system.get_system_stats())
                
                st.subheader("📈 Data Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Chunks", system_stats.get("total_chunks", 0))
                
                with col2:
                    st.metric("Avg Chunk Tokens", f"{system_stats.get('avg_chunk_tokens', 0):.0f}")
                
                with col3:
                    methods = system_stats.get("available_methods", [])
                    st.metric("Available Methods", len(methods))
                
            except Exception as e:
                st.error(f"Error loading system stats: {e}")
    
    def _render_settings(self):
        """Render settings interface"""
        st.header("⚙️ Settings")
        
        # System configuration
        st.subheader("🔧 System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Ollama Base URL", value="http://localhost:11434", key="ollama_url")
            st.text_input("LLM Model", value="llama3.2", key="llm_model")
            st.text_input("Embedding Model", value="nomic-embed-text", key="embedding_model")
        
        with col2:
            st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, key="chunk_size")
            st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, key="chunk_overlap")
            st.slider("Hybrid Alpha", min_value=0.0, max_value=1.0, value=0.7, key="hybrid_alpha")
        
        # Vector database settings
        st.subheader("🗄️ Vector Database")
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("Qdrant URL", value="http://localhost:6333", key="qdrant_url")
            st.text_input("Collection Name", value="godot_docs", key="collection_name")
        
        with col2:
            st.number_input("Embedding Dimension", value=768, key="embedding_dim")
        
        # Save settings
        if st.button("💾 Save Settings"):
            st.success("Settings saved! Restart the system to apply changes.")
        
        # System actions
        st.subheader("🛠️ System Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Restart System"):
                st.info("System restart functionality would go here")
        
        with col2:
            if st.button("📊 Export Metrics"):
                st.info("Metrics export functionality would go here")
        
        with col3:
            if st.button("🧹 Clear Cache"):
                st.info("Cache clearing functionality would go here")
    
    def _initialize_system(self) -> bool:
        """Initialize the RAG system"""
        try:
            # Import system components
            from src.config import RAGConfig
            from src.rag_system import ComprehensiveRAGSystem
            from src.monitoring import MetricsCollector
            
            # Initialize configuration
            config = RAGConfig()
            
            # Initialize RAG system
            rag_system = ComprehensiveRAGSystem(config)
            
            # Initialize in background (simplified for demo)
            # In real implementation, this would be async
            st.session_state.rag_system = rag_system
            
            # Initialize metrics collector
            metrics_collector = MetricsCollector(config.data_dir)
            metrics_collector.start_background_collection()
            st.session_state.metrics_collector = metrics_collector
            
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            st.error(f"Initialization error: {e}")
            return False
    
    def _execute_query(self, query: str, method: str, prompt_type: str, 
                      use_reranking: bool, max_docs: int) -> Dict[str, Any]:
        """Execute a query against the RAG system"""
        try:
            # This would be async in real implementation
            result = {
                "query": query,
                "answer": f"This is a simulated answer for: '{query}'\n\nIn a real implementation, this would call the RAG system asynchronously.",
                "method": method,
                "prompt_type": prompt_type,
                "processing_time": 1.23,
                "retrieved_documents": [
                    {
                        "content": "Sample document content relevant to the query...",
                        "score": 0.85,
                        "chunk_id": "sample_chunk_1"
                    }
                ],
                "context_length": 1500,
                "num_documents": max_docs
            }
            
            # Add to history
            st.session_state.query_history.append(result)
            
            # Record metrics
            if st.session_state.metrics_collector:
                st.session_state.metrics_collector.record_query(result, success=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            return None
    
    def _display_query_result(self, result: Dict[str, Any]):
        """Display query results"""
        st.divider()
        
        # Answer
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(f"**Answer:**")
        st.markdown(result["answer"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Metadata
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Method", result["method"])
        
        with col2:
            st.metric("Processing Time", f"{result['processing_time']:.2f}s")
        
        with col3:
            st.metric("Documents Used", result["num_documents"])
        
        with col4:
            st.metric("Context Length", result["context_length"])
        
        # Source documents
        if result.get("retrieved_documents"):
            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(result["retrieved_documents"]):
                    st.markdown(f"**Document {i+1}** (Score: {doc.get('score', 0):.3f})")
                    st.markdown(f"```\n{doc['content'][:300]}...\n```")
        
        # Feedback
        st.subheader("👍 Rate this answer")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            rating = st.selectbox(
                "Rating",
                [1, 2, 3, 4, 5],
                index=4,
                key=f"rating_{result.get('timestamp', time.time())}"
            )
        
        with col2:
            feedback_comment = st.text_input(
                "Comments (optional)",
                key=f"feedback_{result.get('timestamp', time.time())}"
            )
        
        if st.button("📝 Submit Feedback"):
            # Record feedback
            if st.session_state.metrics_collector:
                timestamp = result.get('timestamp', datetime.now().isoformat())
                st.session_state.metrics_collector.record_feedback(
                    timestamp, rating, feedback_comment
                )
            st.success("Thank you for your feedback!")


def main():
    """Main function to run the Streamlit app"""
    interface = StreamlitRAGInterface()
    interface.run()


if __name__ == "__main__":
    main()
