"""
Processing metrics and quality tracking service.
Captures comprehensive metrics about document processing performance and quality.

REFACTORING HISTORY:
- Converted MetricsService to inherit from BaseService (DRY consolidation)
- Consolidated imports using app.core.common
- Standardized logging patterns via BaseService
- Applied error handling decorators where appropriate
- Estimated code reduction: 20% fewer lines, 35% less duplication
"""
# DRY CONSOLIDATION: Using consolidated imports
from app.core.common import (
    BaseService, with_service_logging, get_utc_datetime,
    Dict, Any, Optional, List, dataclass, field
)
from app.core.exceptions import with_error_handling

# Specific imports that can't be consolidated
import time
from datetime import datetime
from contextlib import contextmanager
import statistics


@dataclass
class MetricSnapshot:
    """Represents a point-in-time metric measurement."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, Any] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class ProcessingMetrics:
    """Comprehensive metrics for a document processing operation."""
    document_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Stage timings (in seconds)
    validation_time: Optional[float] = None
    extraction_time: Optional[float] = None
    chunking_time: Optional[float] = None
    embedding_time: Optional[float] = None
    storage_time: Optional[float] = None
    total_time: Optional[float] = None
    
    # Resource usage
    peak_memory_mb: Optional[float] = None
    avg_memory_mb: Optional[float] = None
    disk_usage_mb: Optional[float] = None
    
    # Quality metrics
    extraction_quality: Optional[float] = None  # 0.0 to 1.0
    chunk_quality_avg: Optional[float] = None
    chunk_quality_min: Optional[float] = None
    chunk_quality_max: Optional[float] = None
    
    # Processing stats
    file_size_bytes: Optional[int] = None
    extracted_chars: Optional[int] = None
    total_chunks: Optional[int] = None
    quality_chunks: Optional[int] = None
    duplicate_chunks: Optional[int] = None
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    retries: int = 0
    
    # Success indicators
    validation_success: bool = False
    extraction_success: bool = False
    chunking_success: bool = False
    embedding_success: bool = False
    storage_success: bool = False
    overall_success: bool = False
    
    def calculate_totals(self) -> None:
        """Calculate total metrics."""
        if self.end_time and self.start_time:
            self.total_time = (self.end_time - self.start_time).total_seconds()
        
        # Calculate overall success
        self.overall_success = all([
            self.validation_success,
            self.extraction_success,
            self.chunking_success,
            self.storage_success
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging/storage."""
        self.calculate_totals()
        
        return {
            "document_id": self.document_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "timings": {
                "validation_ms": int(self.validation_time * 1000) if self.validation_time else None,
                "extraction_ms": int(self.extraction_time * 1000) if self.extraction_time else None,
                "chunking_ms": int(self.chunking_time * 1000) if self.chunking_time else None,
                "embedding_ms": int(self.embedding_time * 1000) if self.embedding_time else None,
                "storage_ms": int(self.storage_time * 1000) if self.storage_time else None,
                "total_ms": int(self.total_time * 1000) if self.total_time else None,
            },
            "resources": {
                "peak_memory_mb": self.peak_memory_mb,
                "avg_memory_mb": self.avg_memory_mb,
                "disk_usage_mb": self.disk_usage_mb,
            },
            "quality": {
                "extraction_quality": self.extraction_quality,
                "chunk_quality_avg": self.chunk_quality_avg,
                "chunk_quality_min": self.chunk_quality_min,
                "chunk_quality_max": self.chunk_quality_max,
            },
            "stats": {
                "file_size_bytes": self.file_size_bytes,
                "extracted_chars": self.extracted_chars,
                "total_chunks": self.total_chunks,
                "quality_chunks": self.quality_chunks,
                "duplicate_chunks": self.duplicate_chunks,
                "extraction_ratio": (
                    self.extracted_chars / self.file_size_bytes
                    if self.file_size_bytes and self.extracted_chars else None
                ),
            },
            "success": {
                "validation": self.validation_success,
                "extraction": self.extraction_success,
                "chunking": self.chunking_success,
                "embedding": self.embedding_success,
                "storage": self.storage_success,
                "overall": self.overall_success,
            },
            "issues": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "retries": self.retries,
                "error_details": self.errors[:5],  # First 5 errors
                "warning_details": self.warnings[:5],  # First 5 warnings
            }
        }


class MetricsService(BaseService):
    """Service for tracking processing metrics and quality."""
    
    def __init__(self):
        # DRY CONSOLIDATION: Using BaseService initialization
        super().__init__("metrics")
        
        self.current_metrics: Dict[str, ProcessingMetrics] = {}
        self.historical_metrics: List[ProcessingMetrics] = []
        self.memory_samples: Dict[str, List[float]] = {}
    
    @with_service_logging("start_processing")
    @with_error_handling("start_processing")
    def start_processing(self, document_id: str, file_size_bytes: Optional[int] = None) -> ProcessingMetrics:
        """Start tracking metrics for a document processing operation."""
        metrics = ProcessingMetrics(
            document_id=document_id,
            start_time=get_utc_datetime(),
            file_size_bytes=file_size_bytes
        )
        
        self.current_metrics[document_id] = metrics
        self.memory_samples[document_id] = []
        
        self.logger.info(
            "metrics_tracking_started",
            document_id=document_id,
            file_size_bytes=file_size_bytes
        )
        
        return metrics
    
    @with_service_logging("end_processing")
    @with_error_handling("end_processing")
    def end_processing(self, document_id: str) -> Optional[ProcessingMetrics]:
        """End tracking and finalize metrics."""
        metrics = self.current_metrics.get(document_id)
        if not metrics:
            self.logger.warning("no_metrics_found_to_end", document_id=document_id)
            return None
        
        metrics.end_time = get_utc_datetime()
        metrics.calculate_totals()
        
        # Calculate average memory if samples exist
        if document_id in self.memory_samples and self.memory_samples[document_id]:
            metrics.avg_memory_mb = statistics.mean(self.memory_samples[document_id])
            metrics.peak_memory_mb = max(self.memory_samples[document_id])
        
        # Move to historical
        self.historical_metrics.append(metrics)
        del self.current_metrics[document_id]
        if document_id in self.memory_samples:
            del self.memory_samples[document_id]
        
        # Log final metrics
        self.logger.info(
            "processing_metrics_final",
            **metrics.to_dict()
        )
        
        return metrics
    
    @contextmanager
    def track_stage(self, document_id: str, stage: str):
        """Context manager to track timing for a processing stage."""
        metrics = self.current_metrics.get(document_id)
        if not metrics:
            self.logger.warning("no_metrics_found_for_stage", document_id=document_id, stage=stage)
            yield
            return
        
        start_time = time.time()
        
        try:
            yield metrics
            # Stage succeeded
            setattr(metrics, f"{stage}_success", True)
        except Exception as e:
            # Stage failed
            metrics.errors.append(f"{stage}: {str(e)}")
            raise
        finally:
            # Record timing
            elapsed = time.time() - start_time
            setattr(metrics, f"{stage}_time", elapsed)
            
            self.logger.debug(
                "stage_timing_recorded",
                document_id=document_id,
                stage=stage,
                elapsed_ms=int(elapsed * 1000)
            )
    
    def record_memory_sample(self, document_id: str, memory_mb: float) -> None:
        """Record a memory usage sample."""
        if document_id in self.memory_samples:
            self.memory_samples[document_id].append(memory_mb)
            
            # Update peak memory immediately
            metrics = self.current_metrics.get(document_id)
            if metrics:
                if not metrics.peak_memory_mb or memory_mb > metrics.peak_memory_mb:
                    metrics.peak_memory_mb = memory_mb
    
    def record_quality_metrics(
        self,
        document_id: str,
        extraction_quality: Optional[float] = None,
        chunk_qualities: Optional[List[float]] = None
    ) -> None:
        """Record quality metrics for processed content."""
        metrics = self.current_metrics.get(document_id)
        if not metrics:
            return
        
        if extraction_quality is not None:
            metrics.extraction_quality = extraction_quality
        
        if chunk_qualities:
            metrics.chunk_quality_avg = statistics.mean(chunk_qualities)
            metrics.chunk_quality_min = min(chunk_qualities)
            metrics.chunk_quality_max = max(chunk_qualities)
    
    def record_processing_stats(
        self,
        document_id: str,
        extracted_chars: Optional[int] = None,
        total_chunks: Optional[int] = None,
        quality_chunks: Optional[int] = None,
        duplicate_chunks: Optional[int] = None
    ) -> None:
        """Record processing statistics."""
        metrics = self.current_metrics.get(document_id)
        if not metrics:
            return
        
        if extracted_chars is not None:
            metrics.extracted_chars = extracted_chars
        if total_chunks is not None:
            metrics.total_chunks = total_chunks
        if quality_chunks is not None:
            metrics.quality_chunks = quality_chunks
        if duplicate_chunks is not None:
            metrics.duplicate_chunks = duplicate_chunks
    
    def add_warning(self, document_id: str, warning: str) -> None:
        """Add a warning to the metrics."""
        metrics = self.current_metrics.get(document_id)
        if metrics:
            metrics.warnings.append(warning)
            self.logger.warning("processing_warning_recorded", document_id=document_id, warning=warning)
    
    def increment_retries(self, document_id: str) -> None:
        """Increment retry count."""
        metrics = self.current_metrics.get(document_id)
        if metrics:
            metrics.retries += 1
    
    def get_aggregate_metrics(self, last_n: int = 100) -> Dict[str, Any]:
        """Get aggregate metrics from recent processing operations."""
        recent_metrics = self.historical_metrics[-last_n:] if len(self.historical_metrics) > last_n else self.historical_metrics
        
        if not recent_metrics:
            return {"message": "No historical metrics available"}
        
        # Calculate aggregates
        success_rate = sum(1 for m in recent_metrics if m.overall_success) / len(recent_metrics)
        
        total_times = [m.total_time for m in recent_metrics if m.total_time]
        avg_processing_time = statistics.mean(total_times) if total_times else 0
        
        extraction_times = [m.extraction_time for m in recent_metrics if m.extraction_time]
        avg_extraction_time = statistics.mean(extraction_times) if extraction_times else 0
        
        quality_scores = [m.extraction_quality for m in recent_metrics if m.extraction_quality]
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        return {
            "sample_size": len(recent_metrics),
            "success_rate": success_rate,
            "average_timings": {
                "total_seconds": avg_processing_time,
                "extraction_seconds": avg_extraction_time,
            },
            "average_quality": avg_quality,
            "error_rate": 1 - success_rate,
            "common_errors": self._get_common_errors(recent_metrics),
        }
    
    def _get_common_errors(self, metrics_list: List[ProcessingMetrics], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get most common errors from metrics."""
        error_counts: Dict[str, int] = {}
        
        for metrics in metrics_list:
            for error in metrics.errors:
                # Simplify error for grouping
                error_type = error.split(":")[0] if ":" in error else error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Sort by frequency
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"error": error, "count": count}
            for error, count in sorted_errors[:top_n]
        ]


# Global metrics service instance
metrics_service = MetricsService()