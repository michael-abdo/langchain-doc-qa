"""
Main FastAPI application with comprehensive error handling.
Entry point for the LangChain Document Q&A API.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
import asyncio

from app.core.config import settings
from app.core.logging import get_logger, LoggingMiddleware, set_correlation_id
from app.core.exceptions import (
    BaseAppException, 
    create_http_exception,
    ConfigurationError,
    ExternalServiceError
)

# Get logger
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    Runs startup and shutdown events.
    """
    # Startup
    logger.info("application_startup", version=settings.APP_VERSION)
    
    try:
        # Validate critical configuration using centralized method
        settings.validate_critical_startup_config()
        logger.info("configuration_validated", provider=settings.LLM_PROVIDER)
        
        # Future: Initialize vector store, database connections, etc.
        
        yield
        
    except Exception as e:
        logger.error("startup_failed", error=str(e), exc_info=True)
        raise
    
    finally:
        # Shutdown
        logger.info("application_shutdown")
        
        # Future: Close database connections, cleanup resources


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Document Q&A system powered by LangChain",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(LoggingMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["x-correlation-id"],
)

# Security middleware
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure based on your domain
    )


# Global exception handlers
@app.exception_handler(BaseAppException)
async def handle_app_exception(request: Request, exc: BaseAppException):
    """Handle custom application exceptions."""
    logger.error(
        "application_error",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
        method=request.method,
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "correlation_id": request.state.correlation_id if hasattr(request.state, "correlation_id") else None
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(x) for x in error["loc"])
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.error(
        "validation_error",
        errors=errors,
        path=request.url.path,
        method=request.method,
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"validation_errors": errors},
                "correlation_id": request.state.correlation_id if hasattr(request.state, "correlation_id") else None
            }
        }
    )


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(request: Request, exc: StarletteHTTPException):
    """Handle generic HTTP exceptions."""
    logger.error(
        "http_error",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail or "An error occurred",
                "correlation_id": request.state.correlation_id if hasattr(request.state, "correlation_id") else None
            }
        }
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """Handle unexpected errors - fail loud but safely."""
    logger.error(
        "unexpected_error",
        error=str(exc),
        error_type=type(exc).__name__,
        path=request.url.path,
        method=request.method,
        exc_info=True,
    )
    
    # In production, return generic error
    if not settings.DEBUG:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred. Please try again later.",
                    "correlation_id": request.state.correlation_id if hasattr(request.state, "correlation_id") else None
                }
            }
        )
    
    # In debug mode, return detailed error
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": str(exc),
                "type": type(exc).__name__,
                "correlation_id": request.state.correlation_id if hasattr(request.state, "correlation_id") else None
            }
        }
    )


# Request middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    
    # Set correlation ID for request
    correlation_id = request.headers.get("x-correlation-id")
    if not correlation_id:
        correlation_id = set_correlation_id()
    else:
        set_correlation_id(correlation_id)
    
    # Store in request state for access in handlers
    request.state.correlation_id = correlation_id
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2)) + "ms"
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            "request_processing_failed",
            error=str(e),
            process_time_ms=round(process_time * 1000, 2),
            exc_info=True
        )
        raise


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - redirects to API docs."""
    return {
        "message": "LangChain Document Q&A API",
        "version": settings.APP_VERSION,
        "docs": f"{settings.API_PREFIX}/docs"
    }


# Import and include routers
from app.api.routes import health
app.include_router(health.router, prefix=settings.API_PREFIX, tags=["health"])
# from app.api.routes import documents, query
# app.include_router(documents.router, prefix=settings.API_PREFIX, tags=["documents"])
# app.include_router(query.router, prefix=settings.API_PREFIX, tags=["query"])


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )