"""
Monitoring and Metrics Collection for RAG System
Tracks performance, usage patterns, and system health
"""

import logging
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@dataclass
class QueryMetrics:
    """Metrics for a single query"""
    timestamp: str
    query: str
    method: str
    prompt_type: str
    processing_time: float
    context_length: int
    num_documents: int
    answer_length: int
    success: bool
    error_message: Optional[str] = None
    user_feedback: Optional[int] = None  # 1-5 rating
    feedback_comment: Optional[str] = None


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_processing_time: float
    avg_context_length: float
    method_usage: Dict[str, int]
    prompt_type_usage: Dict[str, int]


class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, data_dir: Path, max_query_history: int = 10000):
        self.data_dir = Path(data_dir)
        self.metrics_dir = self.data_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("godot_rag.metrics")
        
        # In-memory storage for recent metrics
        self.query_history: deque = deque(maxlen=max_query_history)
        self.system_metrics_history: deque = deque(maxlen=1000)
        
        # Thread-safe locks
        self._lock = threading.Lock()
        
        # Background collection
        self._collection_active = False
        self._collection_thread = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        
        # Aggregated statistics
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "method_counts": defaultdict(int),
            "prompt_type_counts": defaultdict(int),
            "daily_query_counts": defaultdict(int),
            "hourly_query_counts": defaultdict(int),
            "avg_processing_time": 0.0,
            "avg_context_length": 0.0,
            "user_ratings": [],
            "error_types": defaultdict(int)
        }
        
        # Load existing metrics
        self._load_existing_metrics()
    
    def start_background_collection(self, interval_seconds: int = 60):
        """Start background system metrics collection"""
        if self._collection_active:
            return
        
        self._collection_active = True
        self._collection_thread = threading.Thread(
            target=self._background_collection_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._collection_thread.start()
        self.logger.info(f"📊 Started background metrics collection (interval: {interval_seconds}s)")
    
    def stop_background_collection(self):
        """Stop background metrics collection"""
        self._collection_active = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        self.logger.info("🛑 Stopped background metrics collection")
    
    def _background_collection_loop(self, interval_seconds: int):
        """Background loop for collecting system metrics"""
        while self._collection_active:
            try:
                self._collect_system_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                self.logger.error(f"❌ Background metrics collection error: {e}")
                time.sleep(interval_seconds)
    
    def _collect_system_metrics(self):
        """Collect current system metrics"""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # Get system stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Calculate aggregated query stats
            with self._lock:
                recent_queries = list(self.query_history)
                
                if recent_queries:
                    successful = [q for q in recent_queries if q.success]
                    avg_processing_time = sum(q.processing_time for q in successful) / len(successful) if successful else 0
                    avg_context_length = sum(q.context_length for q in successful) / len(successful) if successful else 0
                    
                    method_usage = defaultdict(int)
                    prompt_type_usage = defaultdict(int)
                    
                    for query in recent_queries:
                        method_usage[query.method] += 1
                        prompt_type_usage[query.prompt_type] += 1
                else:
                    avg_processing_time = 0
                    avg_context_length = 0
                    method_usage = {}
                    prompt_type_usage = {}
                
                system_metrics = SystemMetrics(
                    timestamp=datetime.now().isoformat(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024 * 1024),
                    disk_usage_percent=disk.percent,
                    total_queries=self.stats["total_queries"],
                    successful_queries=self.stats["successful_queries"],
                    failed_queries=self.stats["failed_queries"],
                    avg_processing_time=avg_processing_time,
                    avg_context_length=avg_context_length,
                    method_usage=dict(method_usage),
                    prompt_type_usage=dict(prompt_type_usage)
                )
                
                self.system_metrics_history.append(system_metrics)
                
                # Save metrics periodically
                if len(self.system_metrics_history) % 10 == 0:
                    self._save_system_metrics()
                    
        except Exception as e:
            self.logger.error(f"❌ System metrics collection failed: {e}")
    
    def record_query(self, query_data: Dict[str, Any], success: bool = True, error_message: Optional[str] = None):
        """Record a query and its metrics"""
        try:
            metrics = QueryMetrics(
                timestamp=datetime.now().isoformat(),
                query=query_data.get("query", ""),
                method=query_data.get("method", "unknown"),
                prompt_type=query_data.get("prompt_type", "default"),
                processing_time=query_data.get("processing_time", 0.0),
                context_length=query_data.get("context_length", 0),
                num_documents=query_data.get("num_documents", 0),
                answer_length=len(query_data.get("answer", "")),
                success=success,
                error_message=error_message
            )
            
            with self._lock:
                self.query_history.append(metrics)
                
                # Update aggregated stats
                self.stats["total_queries"] += 1
                if success:
                    self.stats["successful_queries"] += 1
                else:
                    self.stats["failed_queries"] += 1
                    if error_message:
                        error_type = type(error_message).__name__ if isinstance(error_message, Exception) else "unknown"
                        self.stats["error_types"][error_type] += 1
                
                self.stats["method_counts"][metrics.method] += 1
                self.stats["prompt_type_counts"][metrics.prompt_type] += 1
                
                # Date-based tracking
                date_key = datetime.now().strftime("%Y-%m-%d")
                hour_key = datetime.now().strftime("%Y-%m-%d-%H")
                self.stats["daily_query_counts"][date_key] += 1
                self.stats["hourly_query_counts"][hour_key] += 1
                
                # Update averages
                successful_queries = [q for q in self.query_history if q.success]
                if successful_queries:
                    self.stats["avg_processing_time"] = sum(q.processing_time for q in successful_queries) / len(successful_queries)
                    self.stats["avg_context_length"] = sum(q.context_length for q in successful_queries) / len(successful_queries)
            
            # Save query metrics periodically
            if len(self.query_history) % 100 == 0:
                self._executor.submit(self._save_query_metrics)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to record query metrics: {e}")
    
    def record_feedback(self, query_timestamp: str, rating: int, comment: Optional[str] = None):
        """Record user feedback for a query"""
        try:
            with self._lock:
                # Find the query by timestamp
                for query in reversed(self.query_history):
                    if query.timestamp == query_timestamp:
                        query.user_feedback = rating
                        query.feedback_comment = comment
                        self.stats["user_ratings"].append(rating)
                        break
                        
            self.logger.info(f"📝 Recorded feedback: {rating}/5 for query at {query_timestamp}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to record feedback: {e}")
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        with self._lock:
            recent_system_metrics = list(self.system_metrics_history)[-1] if self.system_metrics_history else None
            recent_queries = list(self.query_history)[-100:]  # Last 100 queries
            
            # Calculate success rate
            total_queries = self.stats["total_queries"]
            success_rate = (self.stats["successful_queries"] / total_queries * 100) if total_queries > 0 else 0
            
            # Calculate average rating
            ratings = self.stats["user_ratings"]
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            
            # Recent performance
            recent_processing_times = [q.processing_time for q in recent_queries if q.success]
            recent_avg_time = sum(recent_processing_times) / len(recent_processing_times) if recent_processing_times else 0
            
            return {
                "overview": {
                    "total_queries": total_queries,
                    "successful_queries": self.stats["successful_queries"],
                    "failed_queries": self.stats["failed_queries"],
                    "success_rate": round(success_rate, 2),
                    "avg_processing_time": round(self.stats["avg_processing_time"], 3),
                    "avg_context_length": round(self.stats["avg_context_length"], 0),
                    "avg_user_rating": round(avg_rating, 2),
                    "total_ratings": len(ratings)
                },
                "recent_performance": {
                    "recent_avg_time": round(recent_avg_time, 3),
                    "recent_queries_count": len(recent_queries),
                    "recent_success_rate": round(
                        len([q for q in recent_queries if q.success]) / len(recent_queries) * 100
                        if recent_queries else 0, 2
                    )
                },
                "method_usage": dict(self.stats["method_counts"]),
                "prompt_type_usage": dict(self.stats["prompt_type_counts"]),
                "error_types": dict(self.stats["error_types"]),
                "daily_queries": dict(list(self.stats["daily_query_counts"].items())[-7:]),  # Last 7 days
                "hourly_queries": dict(list(self.stats["hourly_query_counts"].items())[-24:]),  # Last 24 hours
                "system_health": {
                    "cpu_percent": recent_system_metrics.cpu_percent if recent_system_metrics else 0,
                    "memory_percent": recent_system_metrics.memory_percent if recent_system_metrics else 0,
                    "memory_used_mb": recent_system_metrics.memory_used_mb if recent_system_metrics else 0,
                    "disk_usage_percent": recent_system_metrics.disk_usage_percent if recent_system_metrics else 0
                } if recent_system_metrics else {}
            }
    
    def get_detailed_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get detailed metrics for the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_iso = cutoff_time.isoformat()
        
        with self._lock:
            # Filter recent queries
            recent_queries = [
                q for q in self.query_history 
                if q.timestamp >= cutoff_iso
            ]
            
            # Filter recent system metrics
            recent_system_metrics = [
                m for m in self.system_metrics_history 
                if m.timestamp >= cutoff_iso
            ]
            
            return {
                "time_period_hours": hours,
                "query_metrics": [asdict(q) for q in recent_queries],
                "system_metrics": [asdict(m) for m in recent_system_metrics],
                "summary": {
                    "total_queries": len(recent_queries),
                    "avg_processing_time": sum(q.processing_time for q in recent_queries if q.success) / len([q for q in recent_queries if q.success]) if recent_queries else 0,
                    "method_breakdown": {
                        method: len([q for q in recent_queries if q.method == method])
                        for method in set(q.method for q in recent_queries)
                    },
                    "success_rate": len([q for q in recent_queries if q.success]) / len(recent_queries) * 100 if recent_queries else 0
                }
            }
    
    def _load_existing_metrics(self):
        """Load existing metrics from disk"""
        try:
            # Load query metrics
            query_file = self.metrics_dir / "query_metrics.jsonl"
            if query_file.exists():
                with open(query_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            metrics = QueryMetrics(**data)
                            self.query_history.append(metrics)
                            
                            # Update stats
                            self.stats["total_queries"] += 1
                            if metrics.success:
                                self.stats["successful_queries"] += 1
                            else:
                                self.stats["failed_queries"] += 1
                            
                            self.stats["method_counts"][metrics.method] += 1
                            self.stats["prompt_type_counts"][metrics.prompt_type] += 1
                            
                            if metrics.user_feedback:
                                self.stats["user_ratings"].append(metrics.user_feedback)
                
                self.logger.info(f"📥 Loaded {len(self.query_history)} existing query metrics")
            
            # Load system metrics
            system_file = self.metrics_dir / "system_metrics.jsonl"
            if system_file.exists():
                with open(system_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            metrics = SystemMetrics(**data)
                            self.system_metrics_history.append(metrics)
                
                self.logger.info(f"📥 Loaded {len(self.system_metrics_history)} existing system metrics")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load existing metrics: {e}")
    
    def _save_query_metrics(self):
        """Save query metrics to disk"""
        try:
            query_file = self.metrics_dir / "query_metrics.jsonl"
            
            with self._lock:
                queries_to_save = list(self.query_history)
            
            with open(query_file, 'w') as f:
                for query in queries_to_save:
                    f.write(json.dumps(asdict(query)) + '\n')
            
            self.logger.debug(f"💾 Saved {len(queries_to_save)} query metrics")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save query metrics: {e}")
    
    def _save_system_metrics(self):
        """Save system metrics to disk"""
        try:
            system_file = self.metrics_dir / "system_metrics.jsonl"
            
            with self._lock:
                metrics_to_save = list(self.system_metrics_history)
            
            with open(system_file, 'w') as f:
                for metrics in metrics_to_save:
                    f.write(json.dumps(asdict(metrics)) + '\n')
            
            self.logger.debug(f"💾 Saved {len(metrics_to_save)} system metrics")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save system metrics: {e}")
    
    def export_metrics(self, filepath: Path, format: str = "json"):
        """Export all metrics to a file"""
        try:
            stats = self.get_current_stats()
            detailed = self.get_detailed_metrics(hours=24*7)  # Last week
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "current_stats": stats,
                "detailed_metrics": detailed
            }
            
            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            self.logger.info(f"📤 Exported metrics to {filepath}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export metrics: {e}")
            raise
    
    def cleanup_old_metrics(self, days: int = 30):
        """Clean up metrics older than specified days"""
        cutoff_time = datetime.now() - timedelta(days=days)
        cutoff_iso = cutoff_time.isoformat()
        
        try:
            with self._lock:
                # Clean query history
                original_query_count = len(self.query_history)
                self.query_history = deque([
                    q for q in self.query_history if q.timestamp >= cutoff_iso
                ], maxlen=self.query_history.maxlen)
                
                # Clean system metrics
                original_system_count = len(self.system_metrics_history)
                self.system_metrics_history = deque([
                    m for m in self.system_metrics_history if m.timestamp >= cutoff_iso
                ], maxlen=self.system_metrics_history.maxlen)
                
                cleaned_queries = original_query_count - len(self.query_history)
                cleaned_system = original_system_count - len(self.system_metrics_history)
                
                self.logger.info(f"🧹 Cleaned up {cleaned_queries} old query metrics and {cleaned_system} old system metrics")
                
                # Save cleaned metrics
                self._save_query_metrics()
                self._save_system_metrics()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to cleanup old metrics: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_background_collection()
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
