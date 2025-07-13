"""
Common imports and utilities module.
Consolidates frequently used imports and patterns to eliminate duplication.
"""

# ============================================================================
# COMMON IMPORTS - DRY CONSOLIDATION
# ============================================================================

# Standard library
import os
import asyncio
import hashlib
from typing import Optional, Dict, Any, List, Tuple, Union, Set
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from functools import wraps
from dataclasses import dataclass, field
from uuid import UUID

# Export commonly used types for importing modules
__all__ = [
    # Standard library exports
    'os', 'asyncio', 'hashlib', 'Optional', 'Dict', 'Any', 'List', 
    'Tuple', 'Union', 'Set', 'Path', 'datetime', 'asynccontextmanager', 
    'dataclass', 'field', 'UUID',
    # Application exports
    'BaseService', 'with_service_logging', 'CommonValidators', 
    'get_service_logger', 'get_api_logger', 'create_safe_filename',
    'calculate_content_hash', 'validate_file_size', 'validate_file_type',
    'create_error_context', 'ApiResponses'
]

# Third-party
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete

# Application core
from app.core.config import settings, config
from app.core.logging import get_logger, get_utc_datetime
# Note: Database imports removed to avoid circular import with vector_reliability
from app.core.exceptions import (
    BaseAppException,
    DocumentProcessingError,
    ValidationError,
    NotFoundError,
    handle_service_error,
    handle_api_error,
    with_error_handling,
    with_api_error_handling
)

# ============================================================================
# COMMON UTILITY FUNCTIONS - DRY CONSOLIDATION
# ============================================================================

def get_service_logger(service_name: str):
    """Get a logger for a service with consistent naming."""
    return get_logger(f"app.services.{service_name}")


def get_api_logger(endpoint_name: str):
    """Get a logger for an API endpoint with consistent naming."""
    return get_logger(f"app.api.{endpoint_name}")


def create_safe_filename(original_filename: str, timestamp: Optional[datetime] = None) -> str:
    """
    Create a safe filename with timestamp prefix.
    Consolidates filename generation logic.
    """
    if not timestamp:
        timestamp = get_utc_datetime()
    
    # Remove dangerous characters
    safe_name = "".join(c for c in original_filename if c.isalnum() or c in "._-")
    name_part, ext_part = os.path.splitext(safe_name)
    
    # Add timestamp prefix
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    return f"{timestamp_str}_{name_part[:50]}{ext_part}"


def calculate_content_hash(content: bytes) -> str:
    """Calculate SHA-256 hash of content."""
    return hashlib.sha256(content).hexdigest()


def validate_file_size(file_size_bytes: int, max_size_mb: Optional[int] = None) -> None:
    """
    Validate file size against limits.
    
    Args:
        file_size_bytes: File size in bytes
        max_size_mb: Maximum size in MB (defaults to config)
        
    Raises:
        ValidationError: If file is too large
    """
    max_size = max_size_mb or config.processing_config["max_file_size_mb"]
    max_bytes = max_size * 1024 * 1024
    
    if file_size_bytes > max_bytes:
        raise ValidationError(
            f"File size {file_size_bytes / (1024*1024):.1f}MB exceeds limit of {max_size}MB"
        )


def validate_file_type(filename: str, allowed_types: Optional[List[str]] = None) -> str:
    """
    Validate and return file extension.
    
    Args:
        filename: Original filename
        allowed_types: List of allowed extensions (defaults to config)
        
    Returns:
        Validated file extension
        
    Raises:
        ValidationError: If file type not supported
    """
    file_extension = Path(filename).suffix.lower()
    allowed = set(allowed_types or config.processing_config["allowed_types"])
    
    if file_extension not in allowed:
        raise ValidationError(
            f"File type '{file_extension}' not supported. Allowed: {sorted(allowed)}"
        )
    
    return file_extension


def create_error_context(operation: str, **kwargs) -> Dict[str, Any]:
    """Create standardized error context for logging."""
    context = {
        "operation": operation,
        "timestamp": get_utc_datetime().isoformat()
    }
    context.update(kwargs)
    return context


# ============================================================================
# COMMON VALIDATION PATTERNS - DRY CONSOLIDATION
# ============================================================================

class CommonValidators:
    """Common validation functions used across services."""
    
    @staticmethod
    def validate_document_id(document_id: str) -> str:
        """Validate document ID format."""
        if not document_id or not isinstance(document_id, str):
            raise ValidationError("Document ID must be a non-empty string")
        return document_id.strip()
    
    @staticmethod
    def validate_content_not_empty(content: bytes, operation: str = "processing") -> None:
        """Validate that content is not empty."""
        if not content or len(content) == 0:
            raise ValidationError(f"Empty content provided for {operation}")
    
    @staticmethod
    def validate_text_extraction(text: str, min_length: int = 10) -> None:
        """Validate extracted text quality."""
        if not text or len(text.strip()) < min_length:
            raise ValidationError(
                f"Extracted text too short: {len(text.strip()) if text else 0} chars "
                f"(minimum {min_length})"
            )


# ============================================================================
# COMMON RESPONSE PATTERNS - DRY CONSOLIDATION  
# ============================================================================

class ApiResponses:
    """Standardized API response patterns."""
    
    @staticmethod
    def success(data: Any, message: str = "Operation completed successfully") -> Dict[str, Any]:
        """Create success response."""
        return {
            "status": "success",
            "message": message,
            "data": data,
            "timestamp": get_utc_datetime().isoformat()
        }
    
    @staticmethod
    def error(message: str, error_code: str = "OPERATION_FAILED", details: Optional[Dict] = None) -> Dict[str, Any]:
        """Create error response."""
        return {
            "status": "error",
            "message": message,
            "error_code": error_code,
            "details": details or {},
            "timestamp": get_utc_datetime().isoformat()
        }
    
    @staticmethod
    def processing(operation: str, resource_id: str) -> Dict[str, Any]:
        """Create processing response."""
        return {
            "status": "processing",
            "message": f"{operation} started",
            "resource_id": resource_id,
            "timestamp": get_utc_datetime().isoformat()
        }


# ============================================================================
# COMMON SERVICE PATTERNS - DRY CONSOLIDATION
# ============================================================================

class BaseService:
    """
    Base service class with common initialization patterns.
    Eliminates repeated service setup code.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = get_service_logger(service_name)
        self.config = config
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize service with common setup."""
        if self._initialized:
            return
        
        self.logger.info(f"{self.service_name}_service_initializing")
        self._setup_service()
        self._initialized = True
        self.logger.info(f"{self.service_name}_service_initialized")
    
    def _setup_service(self) -> None:
        """Override in subclasses for specific setup."""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized


def with_service_logging(operation: str):
    """
    Decorator to add consistent service operation logging.
    Eliminates repeated logging patterns.
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            logger = getattr(self, 'logger', get_logger(__name__))
            service_name = getattr(self, 'service_name', 'unknown')
            
            logger.info(f"{service_name}_{operation}_started")
            
            try:
                result = await func(self, *args, **kwargs)
                logger.info(f"{service_name}_{operation}_completed")
                return result
            except Exception as e:
                logger.error(
                    f"{service_name}_{operation}_failed",
                    error=str(e),
                    exc_info=True
                )
                raise
        
        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            logger = getattr(self, 'logger', get_logger(__name__))
            service_name = getattr(self, 'service_name', 'unknown')
            
            logger.info(f"{service_name}_{operation}_started")
            
            try:
                result = func(self, *args, **kwargs)
                logger.info(f"{service_name}_{operation}_completed")
                return result
            except Exception as e:
                logger.error(
                    f"{service_name}_{operation}_failed",
                    error=str(e),
                    exc_info=True
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator