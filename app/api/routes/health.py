"""
Health check and monitoring endpoints.
Provides comprehensive system health information.
"""
from typing import Dict, Any, Optional
import psutil
import os
from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from app.core.config import settings
from app.core.logging import get_logger, get_utc_timestamp, get_utc_datetime

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


# Track application start time using centralized utility
APP_START_TIME = get_utc_datetime()


def get_uptime_seconds() -> float:
    """Calculate application uptime in seconds."""
    return (get_utc_datetime() - APP_START_TIME).total_seconds()


def check_database_health() -> Dict[str, Any]:
    """Check database connectivity using centralized validation."""
    return settings.validate_database_health()


def check_vector_store_health() -> Dict[str, Any]:
    """Check vector store health using centralized validation."""
    return settings.validate_vector_store_health()


def check_llm_health() -> Dict[str, Any]:
    """Check LLM provider connectivity using centralized validation."""
    return settings.validate_llm_health()


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
        timestamp=get_utc_timestamp(),
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
    return {"status": "alive", "timestamp": get_utc_timestamp()}


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
        "timestamp": get_utc_timestamp()
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