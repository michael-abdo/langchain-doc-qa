"""
Custom exceptions for the application.
Follows fail-fast principle with clear, actionable error messages.
Includes common error handling patterns to eliminate duplication.
"""
from typing import Optional, Dict, Any, Callable, TypeVar, Union
from functools import wraps
import asyncio
from fastapi import HTTPException, status, Request
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

# Type hint for decorated functions
F = TypeVar('F', bound=Callable[..., Any])

logger = get_logger(__name__)


class BaseAppException(Exception):
    """Base exception for all application exceptions."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(BaseAppException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        details = {"field": field} if field else {}
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class ValidationError(BaseAppException):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class DocumentProcessingError(BaseAppException):
    """Raised when document processing fails."""
    
    def __init__(self, message: str, filename: Optional[str] = None, error_type: Optional[str] = None):
        details = {}
        if filename:
            details["filename"] = filename
        if error_type:
            details["error_type"] = error_type
            
        super().__init__(
            message=message,
            error_code="DOCUMENT_PROCESSING_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class FileTooLargeError(DocumentProcessingError):
    """Raised when uploaded file exceeds size limit."""
    
    def __init__(self, filename: str, size_mb: float, max_size_mb: int):
        super().__init__(
            message=f"File '{filename}' is too large ({size_mb:.1f}MB). Maximum allowed size is {max_size_mb}MB.",
            filename=filename,
            error_type="file_too_large"
        )
        self.details["size_mb"] = size_mb
        self.details["max_size_mb"] = max_size_mb


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when uploaded file type is not supported."""
    
    def __init__(self, filename: str, file_type: str, supported_types: list):
        super().__init__(
            message=f"File type '{file_type}' is not supported. Supported types: {', '.join(supported_types)}",
            filename=filename,
            error_type="unsupported_file_type"
        )
        self.details["file_type"] = file_type
        self.details["supported_types"] = supported_types


class VectorStoreError(BaseAppException):
    """Raised when vector store operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        details = {"operation": operation} if operation else {}
        super().__init__(
            message=message,
            error_code="VECTOR_STORE_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )


class LLMError(BaseAppException):
    """Raised when LLM operations fail."""
    
    def __init__(self, message: str, provider: Optional[str] = None, model: Optional[str] = None):
        details = {}
        if provider:
            details["provider"] = provider
        if model:
            details["model"] = model
            
        super().__init__(
            message=message,
            error_code="LLM_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )


class RateLimitError(BaseAppException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        details = {"retry_after": retry_after} if retry_after else {}
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )


class AuthenticationError(BaseAppException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class AuthorizationError(BaseAppException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN
        )


class NotFoundError(BaseAppException):
    """Raised when requested resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: Any):
        super().__init__(
            message=f"{resource_type} with ID '{resource_id}' not found",
            error_code="NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource_type": resource_type, "resource_id": str(resource_id)}
        )


class ExternalServiceError(BaseAppException):
    """Raised when external service calls fail."""
    
    def __init__(self, service_name: str, message: str, status_code: Optional[int] = None):
        details = {"service": service_name}
        if status_code:
            details["service_status_code"] = status_code
            
        super().__init__(
            message=f"External service error ({service_name}): {message}",
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )


def get_correlation_id_from_request(request: Request) -> Optional[str]:
    """
    Extract correlation ID from request state.
    Centralized utility to reduce duplication across exception handlers.
    """
    return getattr(request.state, "correlation_id", None) if hasattr(request, "state") else None


def create_error_response(
    status_code: int,
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> JSONResponse:
    """
    Centralized error response factory.
    Creates consistent JSON error responses across all exception handlers.
    
    Args:
        status_code: HTTP status code
        error_code: Application-specific error code
        message: Human-readable error message
        details: Optional additional error details
        correlation_id: Request correlation ID for tracing
    
    Returns:
        JSONResponse with standardized error format
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {},
                "correlation_id": correlation_id
            }
        }
    )


def create_app_exception_response(request: Request, exc: BaseAppException) -> JSONResponse:
    """
    Create error response for custom application exceptions.
    Uses centralized response factory to ensure consistency.
    """
    return create_error_response(
        status_code=exc.status_code,
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        correlation_id=get_correlation_id_from_request(request)
    )


def create_validation_error_response(request: Request, errors: list) -> JSONResponse:
    """
    Create error response for validation errors.
    Uses centralized response factory to ensure consistency.
    """
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        error_code="VALIDATION_ERROR",
        message="Request validation failed",
        details={"validation_errors": errors},
        correlation_id=get_correlation_id_from_request(request)
    )


def create_http_error_response(request: Request, status_code: int, detail: str) -> JSONResponse:
    """
    Create error response for HTTP errors.
    Uses centralized response factory to ensure consistency.
    """
    return create_error_response(
        status_code=status_code,
        error_code=f"HTTP_{status_code}",
        message=detail or "An error occurred",
        correlation_id=get_correlation_id_from_request(request)
    )


def create_unexpected_error_response(request: Request, debug: bool = False, error_detail: str = None) -> JSONResponse:
    """
    Create error response for unexpected errors.
    Uses centralized response factory to ensure consistency.
    """
    if debug and error_detail:
        return create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_SERVER_ERROR",
            message=error_detail,
            correlation_id=get_correlation_id_from_request(request)
        )
    else:
        return create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred. Please try again later.",
            correlation_id=get_correlation_id_from_request(request)
        )


def create_http_exception(exc: BaseAppException) -> HTTPException:
    """Convert custom exception to FastAPI HTTPException."""
    return HTTPException(
        status_code=exc.status_code,
        detail={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


# ============================================================================
# COMMON ERROR HANDLING PATTERNS - DRY CONSOLIDATION
# ============================================================================

def handle_service_error(
    operation: str,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    raise_as: Optional[type] = None
) -> None:
    """
    Consolidated error handling pattern for services.
    Replaces repeated try-catch-log-raise patterns.
    
    Args:
        operation: Description of operation that failed
        error: The original exception
        context: Additional context for logging
        raise_as: Exception class to raise (defaults to DocumentProcessingError)
    """
    error_context = context or {}
    error_context.update({
        "operation": operation,
        "error": str(error),
        "error_type": type(error).__name__
    })
    
    logger.error(f"{operation}_failed", **error_context, exc_info=True)
    
    # Determine what exception to raise
    if raise_as:
        raise raise_as(f"{operation} failed: {str(error)}")
    else:
        raise DocumentProcessingError(f"{operation} failed: {str(error)}")


def with_error_handling(
    operation: str,
    context: Optional[Dict[str, Any]] = None,
    raise_as: Optional[type] = None,
    reraise_if: Optional[tuple] = None
):
    """
    Decorator to add consistent error handling to service methods.
    
    Args:
        operation: Description of operation for logging
        context: Additional context for logging
        raise_as: Exception class to raise on error
        reraise_if: Tuple of exception types to reraise without modification
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Re-raise specific exceptions without modification
                if reraise_if and isinstance(e, reraise_if):
                    raise
                
                handle_service_error(operation, e, context, raise_as)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Re-raise specific exceptions without modification
                if reraise_if and isinstance(e, reraise_if):
                    raise
                
                handle_service_error(operation, e, context, raise_as)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def handle_api_error(
    operation: str,
    error: Exception,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    context: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """
    Consolidated API error handling pattern.
    Replaces repeated HTTPException raising patterns.
    
    Args:
        operation: Description of operation that failed
        error: The original exception  
        status_code: HTTP status code to return
        context: Additional context for logging
        
    Returns:
        HTTPException to raise
    """
    error_context = context or {}
    error_context.update({
        "operation": operation,
        "error": str(error),
        "error_type": type(error).__name__
    })
    
    logger.error(f"api_{operation}_failed", **error_context, exc_info=True)
    
    return HTTPException(
        status_code=status_code,
        detail=f"Failed to {operation}: {str(error)}"
    )


def with_api_error_handling(
    operation: str,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    context: Optional[Dict[str, Any]] = None
):
    """
    Decorator to add consistent API error handling to endpoint functions.
    
    Args:
        operation: Description of operation for error messages
        status_code: HTTP status code to return on error
        context: Additional context for logging
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTPExceptions without modification
                raise
            except BaseAppException as e:
                # Convert app exceptions to HTTP exceptions
                raise HTTPException(
                    status_code=e.status_code,
                    detail=e.message
                )
            except Exception as e:
                # Handle unexpected errors
                raise handle_api_error(operation, e, status_code, context)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTPExceptions without modification
                raise
            except BaseAppException as e:
                # Convert app exceptions to HTTP exceptions
                raise HTTPException(
                    status_code=e.status_code,
                    detail=e.message
                )
            except Exception as e:
                # Handle unexpected errors
                raise handle_api_error(operation, e, status_code, context)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator