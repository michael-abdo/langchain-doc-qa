"""
Main FastAPI application with comprehensive error handling.
Entry point for the LangChain Document Q&A API.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
import asyncio

from app.core.config import settings
from app.core.logging import get_logger, LoggingMiddleware, set_correlation_id, log_request_error
from app.core.database import init_database, close_database
from app.core.exceptions import (
    BaseAppException, 
    create_http_exception,
    ConfigurationError,
    ExternalServiceError,
    create_app_exception_response,
    create_validation_error_response,
    create_http_error_response,
    create_unexpected_error_response
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
        
        # Initialize database if configured
        if settings.DATABASE_URL:
            await init_database()
            logger.info("database_initialized")
        
        # Future: Initialize vector store, etc.
        
        yield
        
    except Exception as e:
        logger.error("startup_failed", error=str(e), exc_info=True)
        raise
    
    finally:
        # Shutdown
        logger.info("application_shutdown")
        
        # Close database connections
        if settings.DATABASE_URL:
            await close_database()
            logger.info("database_closed")
        
        # Future: Cleanup other resources


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
    """Handle custom application exceptions using centralized response factory."""
    log_request_error(
        logger, 
        "application_error",
        request,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details
    )
    
    return create_app_exception_response(request, exc)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    """Handle request validation errors using centralized response factory."""
    errors = []
    for error in exc.errors():
        field_path = " -> ".join(str(x) for x in error["loc"])
        errors.append({
            "field": field_path,
            "message": error["msg"],
            "type": error["type"]
        })
    
    log_request_error(logger, "validation_error", request, errors=errors)
    
    return create_validation_error_response(request, errors)


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(request: Request, exc: StarletteHTTPException):
    """Handle generic HTTP exceptions using centralized response factory."""
    log_request_error(logger, "http_error", request, status_code=exc.status_code, detail=exc.detail)
    
    return create_http_error_response(request, exc.status_code, exc.detail)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """Handle unexpected errors using centralized response factory - fail loud but safely."""
    log_request_error(
        logger, 
        "unexpected_error", 
        request, 
        error=str(exc), 
        error_type=type(exc).__name__, 
        exc_info=True
    )
    
    # Use centralized response factory with debug-aware error detail
    error_detail = f"{str(exc)} (Type: {type(exc).__name__})" if settings.DEBUG else None
    return create_unexpected_error_response(request, debug=settings.DEBUG, error_detail=error_detail)


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