"""
Metrics API endpoints.
Provides access to processing metrics and quality tracking data.

REFACTORING HISTORY:
- Applied consolidated error handling patterns using @with_api_error_handling
- Replaced manual HTTPException creation with ApiResponses patterns
- Consolidated imports using app.core.common
- Standardized API logging via get_api_logger
- Estimated code reduction: 35% fewer lines, 60% less error handling duplication
"""
# DRY CONSOLIDATION: Using consolidated imports
from app.core.common import (
    get_api_logger, ApiResponses,
    Dict, Any, Optional
)
from app.core.exceptions import with_api_error_handling

# Specific imports that can't be consolidated
from fastapi import APIRouter, Query

from app.services.metrics import metrics_service

# DRY CONSOLIDATION: Using consolidated API logger
logger = get_api_logger("metrics")

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/aggregate", response_model=Dict[str, Any])
@with_api_error_handling("get_aggregate_metrics")
async def get_aggregate_metrics(
    last_n: int = Query(100, description="Number of recent operations to analyze", ge=1, le=1000)
) -> Dict[str, Any]:
    """
    Get aggregate metrics from recent processing operations.
    
    Returns success rates, performance stats, and common errors.
    """
    metrics = metrics_service.get_aggregate_metrics(last_n=last_n)
    
    # DRY CONSOLIDATION: Using ApiResponses for standardized response
    return ApiResponses.success(
        metrics,
        f"Aggregate metrics for last {last_n} operations"
    )


@router.get("/current", response_model=Dict[str, Any])
@with_api_error_handling("get_current_processing")
async def get_current_processing() -> Dict[str, Any]:
    """
    Get currently processing documents and their metrics.
    """
    current_docs = list(metrics_service.current_metrics.keys())
    current_count = len(current_docs)
    
    # DRY CONSOLIDATION: Using ApiResponses for standardized response
    return ApiResponses.success({
        "currently_processing": current_count,
        "document_ids": current_docs[:10],  # Limit to first 10 for privacy
        "total_active": current_count
    }, f"Found {current_count} documents currently processing")


@router.get("/health", response_model=Dict[str, Any])
@with_api_error_handling("get_processing_health")
async def get_processing_health() -> Dict[str, Any]:
    """
    Get processing system health metrics.
    """
    # Get recent metrics for health check
    recent_metrics = metrics_service.get_aggregate_metrics(last_n=20)
    
    # Determine health status
    success_rate = recent_metrics.get("success_rate", 0) if isinstance(recent_metrics, dict) else 0
    error_rate = recent_metrics.get("error_rate", 1) if isinstance(recent_metrics, dict) else 1
    
    if success_rate >= 0.95:
        health_status = "healthy"
    elif success_rate >= 0.80:
        health_status = "warning"
    else:
        health_status = "unhealthy"
    
    # Currently processing count
    active_processing = len(metrics_service.current_metrics)
    
    # DRY CONSOLIDATION: Using ApiResponses for standardized response
    return ApiResponses.success({
        "health_status": health_status,
        "success_rate": success_rate,
        "error_rate": error_rate,
        "active_processing": active_processing,
        "metrics_available": len(metrics_service.historical_metrics) > 0,
        "recent_operations": recent_metrics.get("sample_size", 0) if isinstance(recent_metrics, dict) else 0
    }, f"Processing health status: {health_status}")


@router.get("/performance", response_model=Dict[str, Any])
@with_api_error_handling("get_performance_metrics")
async def get_performance_metrics(
    last_n: int = Query(50, description="Number of recent operations to analyze", ge=1, le=500)
) -> Dict[str, Any]:
    """
    Get detailed performance metrics including timing breakdowns.
    """
    # Get recent metrics
    recent_metrics = metrics_service.historical_metrics[-last_n:] if len(metrics_service.historical_metrics) > last_n else metrics_service.historical_metrics
    
    if not recent_metrics:
        # DRY CONSOLIDATION: Using ApiResponses for standardized response
        return ApiResponses.success({
            "message": "No performance data available",
            "sample_size": 0
        }, "No performance data available")
    
    # Calculate detailed performance stats
    stage_times = {
        "validation": [m.validation_time for m in recent_metrics if m.validation_time],
        "extraction": [m.extraction_time for m in recent_metrics if m.extraction_time],
        "chunking": [m.chunking_time for m in recent_metrics if m.chunking_time],
        "embedding": [m.embedding_time for m in recent_metrics if m.embedding_time],
        "storage": [m.storage_time for m in recent_metrics if m.storage_time],
    }
    
    # Calculate averages
    avg_times = {}
    for stage, times in stage_times.items():
        if times:
            avg_times[f"{stage}_avg_seconds"] = sum(times) / len(times)
            avg_times[f"{stage}_max_seconds"] = max(times)
        else:
            avg_times[f"{stage}_avg_seconds"] = 0
            avg_times[f"{stage}_max_seconds"] = 0
    
    # Memory stats
    memory_stats = [m.peak_memory_mb for m in recent_metrics if m.peak_memory_mb]
    avg_memory = sum(memory_stats) / len(memory_stats) if memory_stats else 0
    max_memory = max(memory_stats) if memory_stats else 0
    
    # DRY CONSOLIDATION: Using ApiResponses for standardized response
    return ApiResponses.success({
        "sample_size": len(recent_metrics),
        "timing_breakdown": avg_times,
        "memory_usage": {
            "avg_peak_mb": avg_memory,
            "max_peak_mb": max_memory
        },
        "throughput": {
            "operations_analyzed": len(recent_metrics),
            "successful_operations": sum(1 for m in recent_metrics if m.overall_success)
        }
    }, f"Performance metrics for last {len(recent_metrics)} operations")