"""
Comprehensive Observability System
Provides detailed logging, metrics, and debugging capabilities for the narrative generation pipeline
"""
import json
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
import psutil
import sys
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class EventType(Enum):
    """Types of system events to track"""
    MEMORY_ACCESS = "memory_access"
    GENERATION_START = "generation_start"
    GENERATION_COMPLETE = "generation_complete"
    SUMMARIZATION = "summarization"
    CONSISTENCY_CHECK = "consistency_check"
    ERROR = "error"
    PERFORMANCE_WARNING = "performance_warning"
    STATE_CHANGE = "state_change"

class Severity(Enum):
    """Event severity levels"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemEvent:
    """Represents a system event for tracking"""
    event_id: str
    event_type: EventType
    severity: Severity
    timestamp: datetime
    component: str
    operation: str
    duration_ms: Optional[float]
    metadata: Dict[str, Any]
    error_details: Optional[Dict[str, Any]]
    context: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    memory_usage_mb: float
    cpu_usage_percent: float
    active_threads: int
    response_time_ms: float
    cache_hit_rate: float
    error_rate: float
    tokens_processed: int
    generation_rate: float  # tokens per second

@dataclass
class GenerationSession:
    """Tracks a complete generation session"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    chapter_id: str
    word_count: int
    model_calls: int
    total_tokens: int
    errors: List[str]
    warnings: List[str]
    performance_metrics: PerformanceMetrics

class ObservabilitySystem:
    """
    Comprehensive observability system for the Novel Memory Architecture
    """
    
    def __init__(self, base_path: Path, enable_detailed_logging: bool = True):
        self.base_path = Path(base_path)
        self.logs_path = self.base_path / "logs"
        self.logs_path.mkdir(parents=True, exist_ok=True)
        
        self.enable_detailed_logging = enable_detailed_logging
        
        # Event storage
        self.events: List[SystemEvent] = []
        self.active_sessions: Dict[str, GenerationSession] = {}
        self.completed_sessions: List[GenerationSession] = []
        
        # Metrics tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.operation_timings: Dict[str, List[float]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Setup loggers
        self._setup_loggers()
        
        # Start background metrics collection
        self._start_metrics_collection()
    
    def _setup_loggers(self):
        """Setup structured logging"""
        # Main system logger
        self.logger = logging.getLogger('novel_memory')
        self.logger.setLevel(logging.DEBUG if self.enable_detailed_logging else logging.INFO)
        
        # File handler for persistent logging
        log_file = self.logs_path / f"system_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # JSON formatter for structured logs
        json_formatter = JsonFormatter()
        file_handler.setFormatter(json_formatter)
        
        self.logger.addHandler(file_handler)
        
        # Error-specific logger
        error_file = self.logs_path / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(json_formatter)
        
        self.logger.addHandler(error_handler)
    
    def _start_metrics_collection(self):
        """Start background thread for metrics collection"""
        def collect_metrics():
            while True:
                try:
                    metrics = self._collect_performance_metrics()
                    with self._lock:
                        self.metrics_history.append(metrics)
                        
                        # Keep only recent metrics (last 1000 entries)
                        if len(self.metrics_history) > 1000:
                            self.metrics_history = self.metrics_history[-1000:]
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Error collecting metrics: {e}")
                    time.sleep(60)  # Wait longer on error
        
        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        metrics_thread.start()
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        process = psutil.Process()
        
        # Memory usage
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        
        # CPU usage
        cpu_usage = process.cpu_percent()
        
        # Thread count
        active_threads = threading.active_count()
        
        # Calculate cache hit rate from recent events
        cache_events = [e for e in self.events[-100:] if e.event_type == EventType.MEMORY_ACCESS]
        cache_hits = sum(1 for e in cache_events if e.metadata.get('cache_hit', False))
        cache_hit_rate = cache_hits / len(cache_events) if cache_events else 0.0
        
        # Calculate error rate
        error_events = [e for e in self.events[-100:] if e.severity in [Severity.ERROR, Severity.CRITICAL]]
        error_rate = len(error_events) / max(len(self.events[-100:]), 1)
        
        # Average response time
        recent_timings = []
        for timings in list(self.operation_timings.values())[-10:]:
            recent_timings.extend(timings[-10:])
        avg_response_time = sum(recent_timings) / len(recent_timings) if recent_timings else 0.0
        
        return PerformanceMetrics(
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage,
            active_threads=active_threads,
            response_time_ms=avg_response_time,
            cache_hit_rate=cache_hit_rate,
            error_rate=error_rate,
            tokens_processed=0,  # Would be tracked separately
            generation_rate=0.0   # Would be calculated from active sessions
        )
    
    @contextmanager
    def track_operation(self, operation_name: str, component: str, context: Dict[str, Any] = None):
        """Context manager for tracking operation performance"""
        if context is None:
            context = {}
        
        start_time = time.time()
        event_id = f"{operation_name}_{int(start_time * 1000)}"
        
        # Log operation start
        self.log_event(
            event_type=EventType.GENERATION_START,
            severity=Severity.INFO,
            component=component,
            operation=operation_name,
            metadata={"operation_id": event_id},
            context=context
        )
        
        try:
            yield event_id
            
            # Success - log completion
            duration_ms = (time.time() - start_time) * 1000
            
            with self._lock:
                if operation_name not in self.operation_timings:
                    self.operation_timings[operation_name] = []
                self.operation_timings[operation_name].append(duration_ms)
                
                # Keep only recent timings
                if len(self.operation_timings[operation_name]) > 100:
                    self.operation_timings[operation_name] = self.operation_timings[operation_name][-100:]
            
            self.log_event(
                event_type=EventType.GENERATION_COMPLETE,
                severity=Severity.INFO,
                component=component,
                operation=operation_name,
                duration_ms=duration_ms,
                metadata={"operation_id": event_id, "success": True},
                context=context
            )
            
        except Exception as e:
            # Error - log failure
            duration_ms = (time.time() - start_time) * 1000
            
            error_details = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
            
            self.log_event(
                event_type=EventType.ERROR,
                severity=Severity.ERROR,
                component=component,
                operation=operation_name,
                duration_ms=duration_ms,
                metadata={"operation_id": event_id, "success": False},
                error_details=error_details,
                context=context
            )
            
            raise
    
    def log_event(self, event_type: EventType, severity: Severity, 
                  component: str, operation: str,
                  duration_ms: Optional[float] = None,
                  metadata: Dict[str, Any] = None,
                  error_details: Dict[str, Any] = None,
                  context: Dict[str, Any] = None):
        """Log a system event"""
        if metadata is None:
            metadata = {}
        if context is None:
            context = {}
        
        event_id = f"{component}_{operation}_{int(time.time() * 1000)}"
        
        event = SystemEvent(
            event_id=event_id,
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            duration_ms=duration_ms,
            metadata=metadata,
            error_details=error_details,
            context=context
        )
        
        with self._lock:
            self.events.append(event)
            
            # Keep only recent events (last 10000)
            if len(self.events) > 10000:
                self.events = self.events[-10000:]
        
        # Log to structured logger
        log_data = {
            "event_id": event_id,
            "event_type": event_type.value,
            "component": component,
            "operation": operation,
            "duration_ms": duration_ms,
            "metadata": metadata,
            "context": context
        }
        
        if severity == Severity.DEBUG:
            self.logger.debug(json.dumps(log_data))
        elif severity == Severity.INFO:
            self.logger.info(json.dumps(log_data))
        elif severity == Severity.WARNING:
            self.logger.warning(json.dumps(log_data))
        elif severity == Severity.ERROR:
            log_data["error_details"] = error_details
            self.logger.error(json.dumps(log_data))
        elif severity == Severity.CRITICAL:
            log_data["error_details"] = error_details
            self.logger.critical(json.dumps(log_data))
    
    def start_generation_session(self, chapter_id: str) -> str:
        """Start tracking a generation session"""
        session_id = f"session_{chapter_id}_{int(time.time() * 1000)}"
        
        session = GenerationSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            chapter_id=chapter_id,
            word_count=0,
            model_calls=0,
            total_tokens=0,
            errors=[],
            warnings=[],
            performance_metrics=self._collect_performance_metrics()
        )
        
        with self._lock:
            self.active_sessions[session_id] = session
        
        self.log_event(
            event_type=EventType.STATE_CHANGE,
            severity=Severity.INFO,
            component="session_manager",
            operation="start_session",
            metadata={"session_id": session_id, "chapter_id": chapter_id}
        )
        
        return session_id
    
    def end_generation_session(self, session_id: str, word_count: int = 0,
                             model_calls: int = 0, total_tokens: int = 0):
        """End a generation session"""
        with self._lock:
            if session_id not in self.active_sessions:
                self.logger.warning(f"Attempted to end unknown session: {session_id}")
                return
            
            session = self.active_sessions[session_id]
            session.end_time = datetime.now()
            session.word_count = word_count
            session.model_calls = model_calls
            session.total_tokens = total_tokens
            
            self.completed_sessions.append(session)
            del self.active_sessions[session_id]
        
        duration = (session.end_time - session.start_time).total_seconds()
        
        self.log_event(
            event_type=EventType.STATE_CHANGE,
            severity=Severity.INFO,
            component="session_manager",
            operation="end_session",
            duration_ms=duration * 1000,
            metadata={
                "session_id": session_id,
                "word_count": word_count,
                "model_calls": model_calls,
                "total_tokens": total_tokens,
                "duration_seconds": duration
            }
        )
    
    def add_session_event(self, session_id: str, event_type: str, details: Any):
        """Add an event to an active session"""
        with self._lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                
                if event_type == "error":
                    session.errors.append(str(details))
                elif event_type == "warning":
                    session.warnings.append(str(details))
                elif event_type == "model_call":
                    session.model_calls += 1
                elif event_type == "tokens":
                    session.total_tokens += int(details)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        current_metrics = self._collect_performance_metrics()
        
        # Calculate health indicators
        health_indicators = {
            "memory_health": "good" if current_metrics.memory_usage_mb < 1000 else "warning" if current_metrics.memory_usage_mb < 2000 else "critical",
            "cpu_health": "good" if current_metrics.cpu_usage_percent < 70 else "warning" if current_metrics.cpu_usage_percent < 90 else "critical",
            "error_rate_health": "good" if current_metrics.error_rate < 0.01 else "warning" if current_metrics.error_rate < 0.05 else "critical",
            "response_time_health": "good" if current_metrics.response_time_ms < 1000 else "warning" if current_metrics.response_time_ms < 5000 else "critical"
        }
        
        # Overall health
        health_scores = {"good": 3, "warning": 2, "critical": 1}
        avg_score = sum(health_scores[status] for status in health_indicators.values()) / len(health_indicators)
        
        if avg_score >= 2.5:
            overall_health = "good"
        elif avg_score >= 1.5:
            overall_health = "warning"
        else:
            overall_health = "critical"
        
        return {
            "overall_health": overall_health,
            "health_indicators": health_indicators,
            "current_metrics": asdict(current_metrics),
            "active_sessions": len(self.active_sessions),
            "recent_errors": len([e for e in self.events[-100:] if e.severity in [Severity.ERROR, Severity.CRITICAL]]),
            "uptime_seconds": (datetime.now() - self.events[0].timestamp).total_seconds() if self.events else 0
        }
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate performance report for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent events
        recent_events = [e for e in self.events if e.timestamp >= cutoff_time]
        
        # Event counts by type
        event_counts = {}
        for event in recent_events:
            event_counts[event.event_type.value] = event_counts.get(event.event_type.value, 0) + 1
        
        # Error analysis
        errors = [e for e in recent_events if e.severity in [Severity.ERROR, Severity.CRITICAL]]
        error_types = {}
        for error in errors:
            error_type = error.error_details.get('error_type', 'Unknown') if error.error_details else 'Unknown'
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Performance trends
        recent_metrics = [m for m in self.metrics_history if True]  # Would filter by timestamp
        
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        avg_response = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        
        # Session statistics
        recent_sessions = [s for s in self.completed_sessions if s.start_time >= cutoff_time]
        
        total_words = sum(s.word_count for s in recent_sessions)
        total_tokens = sum(s.total_tokens for s in recent_sessions)
        avg_session_duration = sum((s.end_time - s.start_time).total_seconds() for s in recent_sessions) / len(recent_sessions) if recent_sessions else 0
        
        return {
            "time_period_hours": hours,
            "event_summary": {
                "total_events": len(recent_events),
                "by_type": event_counts,
                "error_count": len(errors),
                "error_types": error_types
            },
            "performance_averages": {
                "memory_usage_mb": avg_memory,
                "cpu_usage_percent": avg_cpu,
                "response_time_ms": avg_response
            },
            "generation_statistics": {
                "sessions_completed": len(recent_sessions),
                "total_words_generated": total_words,
                "total_tokens_processed": total_tokens,
                "average_session_duration_seconds": avg_session_duration
            },
            "operation_timings": {
                op: {
                    "count": len(timings),
                    "avg_ms": sum(timings) / len(timings),
                    "min_ms": min(timings),
                    "max_ms": max(timings)
                }
                for op, timings in self.operation_timings.items() if timings
            }
        }
    
    def export_logs(self, start_date: datetime, end_date: datetime, 
                   event_types: List[EventType] = None) -> str:
        """Export logs for a specific time period"""
        filtered_events = []
        
        for event in self.events:
            if start_date <= event.timestamp <= end_date:
                if event_types is None or event.event_type in event_types:
                    filtered_events.append(event)
        
        # Convert to exportable format
        export_data = {
            "export_info": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "total_events": len(filtered_events),
                "exported_at": datetime.now().isoformat()
            },
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "severity": e.severity.value,
                    "timestamp": e.timestamp.isoformat(),
                    "component": e.component,
                    "operation": e.operation,
                    "duration_ms": e.duration_ms,
                    "metadata": e.metadata,
                    "error_details": e.error_details,
                    "context": e.context
                }
                for e in filtered_events
            ]
        }
        
        # Save to file
        export_file = self.logs_path / f"export_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
        
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return str(export_file)

class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

# Import required for datetime operations
from datetime import timedelta