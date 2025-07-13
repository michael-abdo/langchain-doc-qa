"""
Metrics API endpoints.
Provides access to processing metrics and quality tracking data.
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional

from app.services.metrics import metrics_service
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/aggregate", response_model=Dict[str, Any])
async def get_aggregate_metrics(
    last_n: int = Query(100, description="Number of recent operations to analyze", ge=1, le=1000)
) -> Dict[str, Any]:
    """
    Get aggregate metrics from recent processing operations.
    
    Returns success rates, performance stats, and common errors.
    """
    try:
        metrics = metrics_service.get_aggregate_metrics(last_n=last_n)
        return {
            "status": "success",
            "data": metrics,
            "description": f"Aggregate metrics for last {last_n} operations"
        }
    except Exception as e:
        logger.error("failed_to_get_aggregate_metrics", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve aggregate metrics: {str(e)}"
        )


@router.get("/current", response_model=Dict[str, Any])
async def get_current_processing() -> Dict[str, Any]:
    """
    Get currently processing documents and their metrics.
    """
    try:
        current_docs = list(metrics_service.current_metrics.keys())
        current_count = len(current_docs)
        
        return {
            "status": "success",
            "data": {
                "currently_processing": current_count,
                "document_ids": current_docs[:10],  # Limit to first 10 for privacy
                "total_active": current_count
            }
        }
    except Exception as e:
        logger.error("failed_to_get_current_processing", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve current processing metrics: {str(e)}"
        )


@router.get("/health", response_model=Dict[str, Any])
async def get_processing_health() -> Dict[str, Any]:
    """
    Get processing system health metrics.
    """
    try:
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
        
        return {
            "status": "success",
            "data": {
                "health_status": health_status,
                "success_rate": success_rate,
                "error_rate": error_rate,
                "active_processing": active_processing,
                "metrics_available": len(metrics_service.historical_metrics) > 0,
                "recent_operations": recent_metrics.get("sample_size", 0) if isinstance(recent_metrics, dict) else 0
            }
        }
    except Exception as e:
        logger.error("failed_to_get_processing_health", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve processing health: {str(e)}"
        )


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    last_n: int = Query(50, description="Number of recent operations to analyze", ge=1, le=500)
) -> Dict[str, Any]:
    """
    Get detailed performance metrics including timing breakdowns.
    """
    try:
        # Get recent metrics
        recent_metrics = metrics_service.historical_metrics[-last_n:] if len(metrics_service.historical_metrics) > last_n else metrics_service.historical_metrics
        
        if not recent_metrics:
            return {
                "status": "success",
                "data": {
                    "message": "No performance data available",
                    "sample_size": 0
                }
            }
        
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
        
        return {
            "status": "success",
            "data": {
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
            }
        }
    except Exception as e:
        logger.error("failed_to_get_performance_metrics", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve performance metrics: {str(e)}"
        )