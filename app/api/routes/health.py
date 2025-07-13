"""
Health check and monitoring endpoints.
Provides comprehensive system health information.
"""
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import psutil
import os
from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    checks: Dict[str, Dict[str, Any]]


class SystemInfo(BaseModel):
    """System information model."""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    python_version: str
    process_memory_mb: float


# Track application start time
APP_START_TIME = datetime.now(timezone.utc)


def get_uptime_seconds() -> float:
    """Calculate application uptime in seconds."""
    return (datetime.now(timezone.utc) - APP_START_TIME).total_seconds()


def check_database_health() -> Dict[str, Any]:
    """Check database connectivity (placeholder for now)."""
    # TODO: Implement actual database health check
    return {
        "status": "healthy" if not settings.DATABASE_URL else "not_configured",
        "message": "Database not configured" if not settings.DATABASE_URL else "Database is healthy"
    }


def check_vector_store_health() -> Dict[str, Any]:
    """Check vector store health."""
    # TODO: Implement actual vector store health check
    return {
        "status": "healthy",
        "type": settings.VECTOR_STORE_TYPE,
        "message": f"{settings.VECTOR_STORE_TYPE} vector store is ready"
    }


def check_llm_health() -> Dict[str, Any]:
    """Check LLM provider connectivity."""
    has_key = False
    if settings.LLM_PROVIDER == "openai":
        has_key = bool(settings.OPENAI_API_KEY)
    elif settings.LLM_PROVIDER == "anthropic":
        has_key = bool(settings.ANTHROPIC_API_KEY)
    
    return {
        "status": "healthy" if has_key else "unhealthy",
        "provider": settings.LLM_PROVIDER,
        "model": settings.LLM_MODEL,
        "api_key_configured": has_key,
        "message": "LLM provider is configured" if has_key else "LLM API key not configured"
    }


def get_system_info() -> SystemInfo:
    """Get system resource information."""
    process = psutil.Process(os.getpid())
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return SystemInfo(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        memory_percent=memory.percent,
        memory_available_mb=memory.available / 1024 / 1024,
        disk_usage_percent=disk.percent,
        python_version=f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        process_memory_mb=process.memory_info().rss / 1024 / 1024
    )


@router.get("/health", response_model=HealthStatus, tags=["health"])
async def health_check(response: Response):
    """
    Comprehensive health check endpoint.
    Returns detailed system health information.
    """
    checks = {
        "database": check_database_health(),
        "vector_store": check_vector_store_health(),
        "llm": check_llm_health()
    }
    
    # Determine overall health status
    all_healthy = all(check.get("status") == "healthy" for check in checks.values())
    overall_status = "healthy" if all_healthy else "degraded"
    
    # Set response status code
    if not all_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    health_status = HealthStatus(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=settings.APP_VERSION,
        uptime_seconds=get_uptime_seconds(),
        checks=checks
    )
    
    logger.info(
        "health_check_performed",
        status=overall_status,
        checks=checks
    )
    
    return health_status


@router.get("/health/live", tags=["health"])
async def liveness_check():
    """
    Simple liveness check for container orchestration.
    Returns 200 if the application is running.
    """
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get("/health/ready", tags=["health"])
async def readiness_check(response: Response):
    """
    Readiness check for container orchestration.
    Returns 200 if the application is ready to serve requests.
    """
    # Check critical dependencies
    llm_check = check_llm_health()
    vector_store_check = check_vector_store_health()
    
    is_ready = (
        llm_check.get("status") == "healthy" and
        vector_store_check.get("status") == "healthy"
    )
    
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "reason": "Critical services not healthy",
            "checks": {
                "llm": llm_check,
                "vector_store": vector_store_check
            }
        }
    
    return {
        "status": "ready",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/health/system", response_model=SystemInfo, tags=["health"])
async def system_info():
    """
    Get system resource information.
    Useful for monitoring and debugging.
    """
    return get_system_info()


@router.get("/metrics", tags=["monitoring"])
async def metrics():
    """
    Prometheus-compatible metrics endpoint.
    Returns application metrics in Prometheus format.
    """
    # TODO: Implement proper Prometheus metrics
    # For now, return basic metrics
    metrics_data = []
    
    # Uptime metric
    metrics_data.append(f"# HELP app_uptime_seconds Application uptime in seconds")
    metrics_data.append(f"# TYPE app_uptime_seconds gauge")
    metrics_data.append(f"app_uptime_seconds {get_uptime_seconds()}")
    
    # System metrics
    system = get_system_info()
    metrics_data.append(f"# HELP app_cpu_usage_percent CPU usage percentage")
    metrics_data.append(f"# TYPE app_cpu_usage_percent gauge")
    metrics_data.append(f"app_cpu_usage_percent {system.cpu_percent}")
    
    metrics_data.append(f"# HELP app_memory_usage_percent Memory usage percentage")
    metrics_data.append(f"# TYPE app_memory_usage_percent gauge")
    metrics_data.append(f"app_memory_usage_percent {system.memory_percent}")
    
    return Response(
        content="\n".join(metrics_data),
        media_type="text/plain; version=0.0.4"
    )