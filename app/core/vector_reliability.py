"""
Vector Storage Reliability Layer - Day 3
Provides comprehensive reliability, monitoring, and safety mechanisms for vector storage operations.

This module implements critical reliability patterns:
- Memory management with exhaustion prevention
- Performance monitoring with timeout enforcement
- Circuit breakers for external API calls
- Index validation and corruption detection
- Rate limiting and backoff strategies
"""
import asyncio
import time
import os
import psutil
import hashlib
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import threading
from collections import defaultdict, deque

# Avoid circular import by importing directly
from app.core.logging import get_logger

def get_service_logger(service_name: str):
    """Get a logger for a service with consistent naming."""
    return get_logger(f"app.services.{service_name}")
from app.core.exceptions import (
    VectorStoreError,
    ExternalServiceError,
    ValidationError
)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    current_usage_mb: float
    peak_usage_mb: float
    available_mb: float
    usage_percentage: float
    system_total_mb: float
    process_memory_mb: float
    is_critical: bool = False
    is_warning: bool = False


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    operation_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    timeout_count: int = 0
    error_count: int = 0
    success_rate: float = 100.0


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    timeout_seconds: float = 60.0
    half_open_max_calls: int = 3
    reset_timeout: float = 300.0


class MemoryManager:
    """
    Manages memory usage monitoring and exhaustion prevention.
    
    Provides real-time memory monitoring with configurable thresholds
    to prevent system memory exhaustion during vector operations.
    """
    
    def __init__(
        self,
        warning_threshold: float = 80.0,
        critical_threshold: float = 90.0,
        max_memory_mb: Optional[float] = None
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.max_memory_mb = max_memory_mb
        self.logger = get_service_logger("memory_manager")
        self._peak_usage = 0.0
        
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics."""
        # System memory
        memory = psutil.virtual_memory()
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info().rss / 1024 / 1024
        
        # Calculate peak
        self._peak_usage = max(self._peak_usage, process_memory)
        
        # Determine status
        usage_pct = memory.percent
        is_warning = usage_pct >= self.warning_threshold
        is_critical = usage_pct >= self.critical_threshold
        
        stats = MemoryStats(
            current_usage_mb=memory.used / 1024 / 1024,
            peak_usage_mb=self._peak_usage,
            available_mb=memory.available / 1024 / 1024,
            usage_percentage=usage_pct,
            system_total_mb=memory.total / 1024 / 1024,
            process_memory_mb=process_memory,
            is_critical=is_critical,
            is_warning=is_warning
        )
        
        if is_critical:
            self.logger.error(
                "critical_memory_usage",
                usage_pct=usage_pct,
                available_mb=stats.available_mb
            )
        elif is_warning:
            self.logger.warning(
                "high_memory_usage",
                usage_pct=usage_pct,
                available_mb=stats.available_mb
            )
            
        return stats
    
    def check_memory_available(self, required_mb: float) -> bool:
        """Check if sufficient memory is available for operation."""
        stats = self.get_memory_stats()
        
        if stats.is_critical:
            self.logger.error(
                "operation_blocked_critical_memory",
                required_mb=required_mb,
                available_mb=stats.available_mb
            )
            return False
            
        if required_mb > stats.available_mb * 0.8:  # Leave 20% buffer
            self.logger.warning(
                "operation_blocked_insufficient_memory",
                required_mb=required_mb,
                available_mb=stats.available_mb
            )
            return False
            
        return True
    
    def enforce_memory_limits(self, operation: str) -> bool:
        """Enforce memory limits for vector operations."""
        stats = self.get_memory_stats()
        
        if stats.is_critical:
            raise VectorStoreError(
                f"Operation '{operation}' blocked: Critical memory usage "
                f"({stats.usage_percentage:.1f}%)",
                "memory_exhaustion"
            )
            
        return True


class PerformanceMonitor:
    """
    Monitors and enforces performance limits for vector operations.
    
    Tracks query times, detects performance degradation, and enforces
    timeout limits to prevent hanging operations.
    """
    
    def __init__(self, default_timeout: float = 30.0):
        self.default_timeout = default_timeout
        self.logger = get_service_logger("performance_monitor")
        self._metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self._recent_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    @asynccontextmanager
    async def track_operation(self, operation: str, timeout: Optional[float] = None):
        """Context manager to track operation performance with timeout."""
        timeout = timeout or self.default_timeout
        start_time = time.time()
        
        try:
            # Run operation with timeout
            async with asyncio.timeout(timeout):
                yield
                
            # Track successful operation
            elapsed = time.time() - start_time
            self._update_metrics(operation, elapsed, success=True)
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            self._update_metrics(operation, elapsed, success=False, timeout=True)
            self.logger.error(
                "operation_timeout",
                operation=operation,
                timeout=timeout,
                elapsed=elapsed
            )
            raise VectorStoreError(
                f"Operation '{operation}' timed out after {timeout}s",
                "timeout"
            )
        except Exception as e:
            elapsed = time.time() - start_time
            self._update_metrics(operation, elapsed, success=False)
            raise
    
    def _update_metrics(
        self,
        operation: str,
        elapsed: float,
        success: bool = True,
        timeout: bool = False
    ):
        """Update performance metrics for operation."""
        metrics = self._metrics[operation]
        recent = self._recent_times[operation]
        
        metrics.operation_count += 1
        metrics.total_time += elapsed
        metrics.avg_time = metrics.total_time / metrics.operation_count
        metrics.min_time = min(metrics.min_time, elapsed)
        metrics.max_time = max(metrics.max_time, elapsed)
        
        if timeout:
            metrics.timeout_count += 1
        if not success:
            metrics.error_count += 1
            
        metrics.success_rate = (
            (metrics.operation_count - metrics.error_count) / 
            metrics.operation_count * 100
        )
        
        recent.append(elapsed)
        
        # Log performance degradation
        if len(recent) >= 10:
            recent_avg = sum(recent) / len(recent)
            if recent_avg > metrics.avg_time * 1.5:  # 50% slower than average
                self.logger.warning(
                    "performance_degradation_detected",
                    operation=operation,
                    recent_avg=recent_avg,
                    historical_avg=metrics.avg_time
                )
    
    def get_metrics(self, operation: str) -> PerformanceMetrics:
        """Get performance metrics for operation."""
        return self._metrics[operation]
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """Get all performance metrics."""
        return dict(self._metrics)


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.
    
    Prevents cascading failures by temporarily blocking calls to failing services
    and allowing them to recover before re-enabling access.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = get_service_logger(f"circuit_breaker_{name}")
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function call through circuit breaker."""
        with self._lock:
            if self._should_reject_call():
                raise ExternalServiceError(
                    self.name,
                    f"Circuit breaker OPEN for {self.name}"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_reject_call(self) -> bool:
        """Determine if call should be rejected based on circuit state."""
        now = time.time()
        
        if self._state == CircuitState.CLOSED:
            return False
            
        elif self._state == CircuitState.OPEN:
            # Check if we should transition to half-open
            if now - self._last_failure_time >= self.config.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self.logger.info(f"circuit_breaker_half_open", name=self.name)
                return False
            return True
            
        elif self._state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            return self._half_open_calls >= self.config.half_open_max_calls
    
    def _on_success(self):
        """Handle successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Successful call in half-open state - close circuit
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self.logger.info(f"circuit_breaker_closed", name=self.name)
            else:
                # Reset failure count on success
                self._failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open state - back to open
                self._state = CircuitState.OPEN
                self.logger.warning(f"circuit_breaker_reopened", name=self.name)
                
            elif self._failure_count >= self.config.failure_threshold:
                # Too many failures - open circuit
                self._state = CircuitState.OPEN
                self.logger.error(
                    f"circuit_breaker_opened",
                    name=self.name,
                    failure_count=self._failure_count
                )
            
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
            "half_open_calls": self._half_open_calls
        }


class RateLimiter:
    """
    Rate limiter with token bucket algorithm.
    
    Prevents API quota exhaustion and controls request rates to external services.
    """
    
    def __init__(self, max_calls: int, time_window: float = 60.0):
        self.max_calls = max_calls
        self.time_window = time_window
        self.logger = get_service_logger("rate_limiter")
        
        self._tokens = max_calls
        self._last_refill = time.time()
        self._lock = threading.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the rate limiter."""
        with self._lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self._last_refill
            if elapsed > 0:
                new_tokens = int(elapsed * (self.max_calls / self.time_window))
                self._tokens = min(self.max_calls, self._tokens + new_tokens)
                self._last_refill = now
            
            # Check if we have enough tokens
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            else:
                self.logger.warning(
                    "rate_limit_exceeded",
                    requested=tokens,
                    available=self._tokens,
                    max_calls=self.max_calls,
                    window=self.time_window
                )
                return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get rate limiter status."""
        return {
            "max_calls": self.max_calls,
            "time_window": self.time_window,
            "available_tokens": self._tokens,
            "last_refill": self._last_refill
        }


class IndexValidator:
    """
    Validates vector index integrity and detects corruption.
    
    Provides checksum validation, consistency checks, and corruption detection
    for vector indices to ensure data integrity.
    """
    
    def __init__(self):
        self.logger = get_service_logger("index_validator")
    
    def calculate_index_checksum(self, index_data: bytes) -> str:
        """Calculate checksum for index data."""
        return hashlib.sha256(index_data).hexdigest()
    
    def validate_index_integrity(
        self,
        index_path: str,
        expected_checksum: Optional[str] = None
    ) -> bool:
        """Validate index file integrity."""
        try:
            if not os.path.exists(index_path):
                self.logger.error("index_file_missing", path=index_path)
                return False
            
            # Read and validate file
            with open(index_path, 'rb') as f:
                data = f.read()
            
            if len(data) == 0:
                self.logger.error("index_file_empty", path=index_path)
                return False
            
            # Calculate current checksum
            current_checksum = self.calculate_index_checksum(data)
            
            if expected_checksum and current_checksum != expected_checksum:
                self.logger.error(
                    "index_checksum_mismatch",
                    path=index_path,
                    expected=expected_checksum,
                    actual=current_checksum
                )
                return False
            
            self.logger.info(
                "index_validation_passed",
                path=index_path,
                checksum=current_checksum,
                size_bytes=len(data)
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "index_validation_failed",
                path=index_path,
                error=str(e)
            )
            return False
    
    def validate_vector_dimensions(
        self,
        vectors: List[List[float]],
        expected_dimension: int
    ) -> bool:
        """Validate that all vectors have consistent dimensions."""
        if not vectors:
            return True
            
        for i, vector in enumerate(vectors):
            if len(vector) != expected_dimension:
                self.logger.error(
                    "dimension_mismatch",
                    vector_index=i,
                    expected=expected_dimension,
                    actual=len(vector)
                )
                return False
        
        self.logger.info(
            "dimension_validation_passed",
            vector_count=len(vectors),
            dimension=expected_dimension
        )
        return True
    
    def validate_metadata_alignment(
        self,
        vectors: List[List[float]],
        metadatas: List[Dict[str, Any]]
    ) -> bool:
        """Validate that vectors and metadata are properly aligned."""
        if len(vectors) != len(metadatas):
            self.logger.error(
                "metadata_alignment_failed",
                vector_count=len(vectors),
                metadata_count=len(metadatas)
            )
            return False
        
        self.logger.info(
            "metadata_alignment_validated",
            count=len(vectors)
        )
        return True


# Global instances for shared use
memory_manager = MemoryManager()
performance_monitor = PerformanceMonitor()
index_validator = IndexValidator()