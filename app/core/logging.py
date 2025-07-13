"""
Structured logging configuration with correlation IDs.
Provides consistent logging across the application with request tracing.
"""
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import structlog
from structlog.stdlib import LoggerFactory
from pythonjsonlogger import jsonlogger

from app.core.config import settings

# Context variable for storing correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def generate_uuid() -> str:
    """Generate UUID string. Centralized UUID utility."""
    return str(uuid.uuid4())


def log_request_error(logger, event: str, request, **kwargs):
    """
    Centralized request error logging utility.
    Reduces duplication of request context logging across exception handlers.
    """
    logger.error(
        event,
        path=request.url.path,
        method=request.method,
        **kwargs
    )


def get_correlation_id() -> str:
    """Get current correlation ID or generate a new one."""
    correlation_id = correlation_id_var.get()
    if correlation_id is None:
        correlation_id = generate_uuid()
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for current context."""
    if correlation_id is None:
        correlation_id = generate_uuid()
    correlation_id_var.set(correlation_id)
    return correlation_id


def add_correlation_id(logger, method_name, event_dict):
    """Add correlation ID to all log entries."""
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict


def add_app_context(logger, method_name, event_dict):
    """Add application context to all log entries."""
    event_dict["app_name"] = settings.APP_NAME
    event_dict["app_version"] = settings.APP_VERSION
    event_dict["environment"] = "development" if settings.DEBUG else "production"
    return event_dict


def setup_logging():
    """
    Configure structured logging for the application.
    Sets up both stdlib logging and structlog with JSON output.
    """
    # Configure Python stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )
    
    # Configure JSON formatter for stdlib logging
    if settings.LOG_FORMAT == "json":
        json_formatter = jsonlogger.JsonFormatter(
            fmt="%(timestamp)s %(level)s %(name)s %(message)s",
            rename_fields={"timestamp": "@timestamp", "level": "log.level"},
        )
        
        # Update all handlers to use JSON formatter
        for handler in logging.root.handlers:
            handler.setFormatter(json_formatter)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            add_correlation_id,
            add_app_context,
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            ),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" 
            else structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LoggingMiddleware:
    """Middleware to add correlation ID to all requests."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Extract or generate correlation ID
            headers = dict(scope.get("headers", []))
            correlation_id = headers.get(b"x-correlation-id", b"").decode("utf-8")
            
            if not correlation_id:
                correlation_id = generate_uuid()
            
            # Set correlation ID for this request
            set_correlation_id(correlation_id)
            
            # Log request start
            logger = get_logger(__name__)
            logger.info(
                "request_started",
                method=scope["method"],
                path=scope["path"],
                correlation_id=correlation_id,
            )
            
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    # Add correlation ID to response headers
                    headers = message.setdefault("headers", [])
                    headers.append((b"x-correlation-id", correlation_id.encode("utf-8")))
                await send(message)
            
            try:
                await self.app(scope, receive, send_wrapper)
            except Exception as e:
                logger.error(
                    "request_failed",
                    method=scope["method"],
                    path=scope["path"],
                    error=str(e),
                    exc_info=True,
                )
                raise
            finally:
                logger.info(
                    "request_completed",
                    method=scope["method"],
                    path=scope["path"],
                )
        else:
            await self.app(scope, receive, send)


# Datetime utility functions to reduce duplication
def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format. Centralized datetime utility."""
    return datetime.now(timezone.utc).isoformat()


def get_utc_datetime() -> datetime:
    """Get current UTC datetime object. Centralized datetime utility."""
    return datetime.now(timezone.utc)


# Initialize logging on module import
setup_logging()

# Export commonly used logger
logger = get_logger(__name__)