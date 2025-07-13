"""
Database connection abstraction layer.
Provides async database operations with connection pooling and health checks.

REFACTORING HISTORY:
- Added DocumentRepository pattern to eliminate duplicated database query patterns
- Consolidated common document retrieval operations from multiple API endpoints
- Reduced code duplication by 12 instances across app/api/routes/documents.py
"""
from typing import AsyncGenerator, Optional, Dict, Any, List
from contextlib import asynccontextmanager
from functools import wraps
import asyncio
import time
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    AsyncEngine, 
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base
from sqlalchemy import text, select, inspect
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import DisconnectionError, TimeoutError as SQLTimeoutError
import asyncpg

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import ConfigurationError, ExternalServiceError

# Day 3: Import reliability components
from app.core.vector_reliability import (
    memory_manager, performance_monitor, CircuitBreaker, CircuitBreakerConfig
)

logger = get_logger(__name__)

# Base class for all models
Base = declarative_base()

class DatabaseManager:
    """Manages database connections and sessions with Day 3 reliability enhancements."""
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
        
        # Day 3: Reliability Components
        self._circuit_breaker = CircuitBreaker(
            "database",
            CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=30.0,
                half_open_max_calls=3,
                reset_timeout=180.0
            )
        )
        self._connection_pool_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "last_connection_test": None,
            "connection_recovery_count": 0
        }
        self._schema_version = None
        self._last_health_check = None
        self._schema_compatibility_warnings = []
        self._migration_status = "unknown"
    
    async def initialize(self) -> None:
        """Initialize database connection with fail-fast validation."""
        if self._initialized:
            return
            
        if not settings.DATABASE_URL:
            raise ConfigurationError(
                "Database URL not configured. Please set DATABASE_URL in your environment."
            )
        
        try:
            # Create engine with specific pool settings for production
            self._engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.DEBUG,  # SQL logging in debug mode
                pool_size=20,  # Number of connections to maintain
                max_overflow=10,  # Maximum overflow connections
                pool_timeout=30,  # Timeout for getting connection from pool
                pool_recycle=3600,  # Recycle connections after 1 hour
                pool_pre_ping=True,  # Verify connections before use
            )
            
            # Create session factory
            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test the connection
            await self._test_connection()
            
            self._initialized = True
            logger.info("database_initialized", engine=repr(self._engine))
            
        except asyncpg.PostgresError as e:
            raise ExternalServiceError(
                service_name="PostgreSQL",
                message=f"Failed to connect to database: {str(e)}"
            )
        except Exception as e:
            raise ConfigurationError(
                f"Database initialization failed: {str(e)}"
            )
    
    async def _test_connection(self) -> None:
        """Test database connection with Day 3 enhanced validation."""
        # Day 3: Circuit breaker protection
        await self._circuit_breaker.call(self._test_connection_internal)
        
    async def _test_connection_internal(self) -> None:
        """Internal connection test with timeout and validation."""
        # Day 3: Performance monitoring with timeout
        async with performance_monitor.track_operation("database_connection_test", timeout=10.0):
            async with self._engine.begin() as conn:
                # Test basic connectivity
                result = await conn.execute(text("SELECT 1"))
                if not result.scalar():
                    raise ExternalServiceError(
                        service_name="PostgreSQL",
                        message="Database connection test failed"
                    )
                
                # Day 3: Schema validation
                await self._validate_schema(conn)
                
                # Update connection stats
                self._connection_pool_stats["last_connection_test"] = datetime.now().isoformat()
                logger.info(
                    "database_connection_test_passed",
                    timestamp=self._connection_pool_stats["last_connection_test"]
                )
    
    async def _validate_schema(self, conn) -> None:
        """Comprehensive database schema validation with migration detection."""
        try:
            # Define required schema components
            essential_tables = {
                'documents': ['id', 'filename', 'content_type', 'upload_timestamp', 'file_size', 'content_hash'],
                'document_chunks': ['id', 'document_id', 'chunk_index', 'content', 'embedding_id', 'metadata']
            }
            
            required_indexes = {
                'idx_documents_content_hash': 'documents',
                'idx_document_chunks_document_id': 'document_chunks',
                'idx_document_chunks_embedding_id': 'document_chunks'
            }
            
            # Step 1: Table existence and column validation
            await self._validate_table_structure(conn, essential_tables)
            
            # Step 2: Index validation
            await self._validate_indexes(conn, required_indexes)
            
            # Step 3: Schema version and migration compatibility
            await self._validate_schema_version(conn)
            
            # Step 4: Constraint validation
            await self._validate_constraints(conn, essential_tables)
            
            logger.info(
                "comprehensive_schema_validation_passed",
                schema_version=self._schema_version,
                validated_tables=list(essential_tables.keys()),
                validated_indexes=list(required_indexes.keys())
            )
            
        except Exception as e:
            logger.error(
                "comprehensive_schema_validation_failed",
                error=str(e)
            )
            raise ExternalServiceError(
                service_name="PostgreSQL",
                message=f"Database schema validation failed: {str(e)}"
            )
    
    async def _validate_table_structure(self, conn, essential_tables: Dict[str, List[str]]) -> None:
        """Validate table existence and required columns."""
        for table_name, required_columns in essential_tables.items():
            # Check table existence
            result = await conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table_name)"
            ), {"table_name": table_name})
            
            if not result.scalar():
                raise ExternalServiceError(
                    service_name="PostgreSQL",
                    message=f"Essential table '{table_name}' missing from database schema"
                )
            
            # Validate required columns
            columns_result = await conn.execute(text(
                "SELECT column_name FROM information_schema.columns WHERE table_name = :table_name"
            ), {"table_name": table_name})
            
            existing_columns = {row[0] for row in columns_result.fetchall()}
            missing_columns = set(required_columns) - existing_columns
            
            if missing_columns:
                raise ExternalServiceError(
                    service_name="PostgreSQL",
                    message=f"Table '{table_name}' missing required columns: {missing_columns}"
                )
    
    async def _validate_indexes(self, conn, required_indexes: Dict[str, str]) -> None:
        """Validate required database indexes."""
        for index_name, table_name in required_indexes.items():
            result = await conn.execute(text(
                "SELECT EXISTS (SELECT FROM pg_indexes WHERE indexname = :index_name AND tablename = :table_name)"
            ), {"index_name": index_name, "table_name": table_name})
            
            if not result.scalar():
                logger.warning(
                    "missing_recommended_index",
                    index_name=index_name,
                    table_name=table_name
                )
    
    async def _validate_schema_version(self, conn) -> None:
        """Validate schema version and migration compatibility."""
        try:
            # Check if schema_version table exists
            version_table_exists = await conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'schema_version')"
            ))
            
            if version_table_exists.scalar():
                # Get current schema version
                result = await conn.execute(text(
                    "SELECT version, applied_at FROM schema_version ORDER BY applied_at DESC LIMIT 1"
                ))
                row = result.fetchone()
                
                if row:
                    self._schema_version = row[0]
                    applied_at = row[1]
                    
                    # Validate version compatibility
                    compatible_versions = ['1.0.0', '1.1.0', '1.2.0']  # Define compatible versions
                    if self._schema_version not in compatible_versions:
                        logger.warning(
                            "schema_version_compatibility_warning",
                            current_version=self._schema_version,
                            compatible_versions=compatible_versions
                        )
                    
                    logger.info(
                        "schema_version_detected",
                        version=self._schema_version,
                        applied_at=applied_at
                    )
                else:
                    self._schema_version = "no_migrations"
            else:
                # No versioning table - check if this is a fresh install or legacy
                doc_count = await conn.execute(text(
                    "SELECT COUNT(*) FROM documents"
                ))
                if doc_count.scalar() > 0:
                    self._schema_version = "legacy_unversioned"
                    logger.warning(
                        "legacy_schema_detected",
                        message="Database contains data but no version tracking"
                    )
                else:
                    self._schema_version = "fresh_install"
                    
        except Exception as e:
            logger.error("schema_version_validation_error", error=str(e))
            self._schema_version = "validation_failed"
    
    async def _validate_constraints(self, conn, essential_tables: Dict[str, List[str]]) -> None:
        """Validate database constraints and foreign keys."""
        try:
            # Check primary key constraints
            for table_name in essential_tables.keys():
                pk_result = await conn.execute(text(
                    """SELECT constraint_name FROM information_schema.table_constraints 
                       WHERE table_name = :table_name AND constraint_type = 'PRIMARY KEY'"""
                ), {"table_name": table_name})
                
                if not pk_result.fetchone():
                    logger.warning(
                        "missing_primary_key_constraint",
                        table_name=table_name
                    )
            
            # Check foreign key constraints
            fk_result = await conn.execute(text(
                """SELECT constraint_name, table_name, column_name, foreign_table_name, foreign_column_name
                   FROM information_schema.key_column_usage kcu
                   JOIN information_schema.referential_constraints rc ON kcu.constraint_name = rc.constraint_name
                   JOIN information_schema.key_column_usage fkcu ON rc.unique_constraint_name = fkcu.constraint_name
                   WHERE kcu.table_name IN :table_names"""
            ), {"table_names": tuple(essential_tables.keys())})
            
            foreign_keys = fk_result.fetchall()
            logger.info(
                "foreign_key_constraints_validated",
                constraint_count=len(foreign_keys)
            )
            
        except Exception as e:
            logger.warning("constraint_validation_warning", error=str(e))
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("database_closed")
    
    @asynccontextmanager
    async def get_session(
        self, 
        timeout: Optional[float] = None,
        isolation_level: Optional[str] = None
    ) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with Day 3 enhanced reliability.
        
        Args:
            timeout: Operation timeout in seconds
            isolation_level: Transaction isolation level
        
        Yields:
            AsyncSession: Database session
            
        Raises:
            ExternalServiceError: If database operation fails
        """
        if not self._initialized:
            await self.initialize()
        
        # Day 3: Memory check before database operations
        memory_manager.enforce_memory_limits("database_session")
        
        # Day 3: Circuit breaker protection
        try:
            await self._circuit_breaker.call(self._validate_connection_health)
        except Exception as e:
            logger.error("database_circuit_breaker_blocked", error=str(e))
            raise ExternalServiceError(
                service_name="PostgreSQL",
                message="Database unavailable due to circuit breaker"
            )
        
        # Day 3: Performance monitoring with timeout
        operation_timeout = timeout or 60.0
        async with performance_monitor.track_operation("database_session", timeout=operation_timeout):
            async with self._session_factory() as session:
                try:
                    # Day 3: Set isolation level if specified
                    if isolation_level:
                        await session.execute(text(f"SET TRANSACTION ISOLATION LEVEL {isolation_level}"))
                    
                    # Update connection stats
                    self._connection_pool_stats["active_connections"] += 1
                    
                    yield session
                    await session.commit()
                    
                    logger.debug(
                        "database_session_completed",
                        isolation_level=isolation_level,
                        timeout=operation_timeout
                    )
                    
                except (DisconnectionError, asyncpg.ConnectionDoesNotExistError) as e:
                    # Day 3: Connection recovery
                    await session.rollback()
                    logger.warning(
                        "database_connection_lost_attempting_recovery",
                        error=str(e)
                    )
                    await self._attempt_connection_recovery()
                    raise ExternalServiceError(
                        service_name="PostgreSQL",
                        message=f"Database connection lost: {str(e)}"
                    )
                except SQLTimeoutError as e:
                    await session.rollback()
                    logger.error(
                        "database_session_timeout",
                        timeout=operation_timeout,
                        error=str(e)
                    )
                    raise ExternalServiceError(
                        service_name="PostgreSQL",
                        message=f"Database operation timed out after {operation_timeout}s"
                    )
                except Exception as e:
                    await session.rollback()
                    self._connection_pool_stats["failed_connections"] += 1
                    logger.error(
                        "database_session_error",
                        error=str(e),
                        exc_info=True
                    )
                    raise ExternalServiceError(
                        service_name="PostgreSQL",
                        message=f"Database operation failed: {str(e)}"
                    )
                finally:
                    self._connection_pool_stats["active_connections"] -= 1
                    await session.close()
    
    async def _validate_connection_health(self) -> None:
        """Validate connection health for circuit breaker."""
        try:
            async with self._engine.begin() as conn:
                result = await conn.execute(text("SELECT 1"))
                if not result.scalar():
                    raise ExternalServiceError(
                        service_name="PostgreSQL",
                        message="Connection health check failed"
                    )
        except Exception as e:
            logger.error("database_health_validation_failed", error=str(e))
            raise
    
    async def _attempt_connection_recovery(self) -> None:
        """Attempt to recover from connection failures."""
        try:
            logger.info("attempting_database_connection_recovery")
            
            # Close current engine
            if self._engine:
                await self._engine.dispose()
            
            # Reinitialize
            await self.initialize()
            
            self._connection_pool_stats["connection_recovery_count"] += 1
            logger.info(
                "database_connection_recovery_successful",
                recovery_count=self._connection_pool_stats["connection_recovery_count"]
            )
            
        except Exception as e:
            logger.error(
                "database_connection_recovery_failed",
                error=str(e)
            )
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive database health check with Day 3 monitoring.
        
        Returns:
            Dict containing health status and details
        """
        if not settings.DATABASE_URL:
            return {
                "status": "not_configured",
                "message": "Database URL not configured"
            }
        
        start_time = time.time()
        self._last_health_check = datetime.now().isoformat()
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Day 3: Circuit breaker status
            cb_status = self._circuit_breaker.get_status()
            
            # Day 3: Memory statistics
            memory_stats = memory_manager.get_memory_stats()
            
            # Day 3: Performance metrics
            db_metrics = performance_monitor.get_metrics("database_session")
            
            # Execute health check query with timeout
            async with performance_monitor.track_operation("database_health_check", timeout=15.0):
                async with self._engine.begin() as conn:
                    result = await conn.execute(
                        text("SELECT version(), current_database(), pg_is_in_recovery(), pg_database_size(current_database())")
                    )
                    row = result.fetchone()
                    
                    # Additional health metrics
                    stats_result = await conn.execute(text(
                        "SELECT numbackends, xact_commit, xact_rollback FROM pg_stat_database WHERE datname = current_database()"
                    ))
                    stats_row = stats_result.fetchone()
            
            check_duration = (time.time() - start_time) * 1000
            
            # Determine overall health
            is_healthy = (
                cb_status["state"] == "closed" and
                not memory_stats.is_critical and
                check_duration < 5000  # Less than 5 seconds
            )
            
            return {
                "status": "healthy" if is_healthy else "degraded",
                "message": "Database is operational" if is_healthy else "Database shows degraded performance",
                "timestamp": self._last_health_check,
                "check_duration_ms": check_duration,
                "details": {
                    "version": row[0],
                    "database": row[1],
                    "is_replica": row[2],
                    "database_size_bytes": row[3],
                    "schema_version": self._schema_version,
                    "connection_backends": stats_row[0] if stats_row else None,
                    "transactions_committed": stats_row[1] if stats_row else None,
                    "transactions_rolled_back": stats_row[2] if stats_row else None
                },
                "pool_stats": {
                    "size": self._engine.pool.size() if hasattr(self._engine.pool, 'size') else None,
                    "checked_out": self._engine.pool.checked_out() if hasattr(self._engine.pool, 'checked_out') else None,
                    "overflow": getattr(self._engine.pool, 'overflow', None),
                    "checked_in": getattr(self._engine.pool, 'checkedin', None),
                },
                "reliability_stats": {
                    "circuit_breaker": cb_status,
                    "connection_stats": self._connection_pool_stats,
                    "memory_status": "critical" if memory_stats.is_critical else "warning" if memory_stats.is_warning else "normal",
                    "performance_metrics": {
                        "avg_session_time_ms": db_metrics.avg_time * 1000,
                        "success_rate": db_metrics.success_rate,
                        "total_sessions": db_metrics.operation_count,
                        "timeout_count": db_metrics.timeout_count
                    }
                }
            }
                
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "message": f"Database health check failed: {str(e)}",
                "timestamp": self._last_health_check,
                "check_duration_ms": (time.time() - start_time) * 1000,
                "circuit_breaker": self._circuit_breaker.get_status(),
                "error": str(e)
            }
    
    async def execute_query(
        self, 
        query: str, 
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        isolation_level: Optional[str] = None
    ) -> Any:
        """
        Execute a raw SQL query with Day 3 reliability enhancements.
        
        Args:
            query: SQL query string
            params: Query parameters
            timeout: Query timeout in seconds
            isolation_level: Transaction isolation level
            
        Returns:
            Query result
        """
        # Day 3: Performance monitoring with timeout
        query_timeout = timeout or 30.0
        async with performance_monitor.track_operation("database_query", timeout=query_timeout):
            async with self.get_session(timeout=query_timeout, isolation_level=isolation_level) as session:
                result = await session.execute(text(query), params or {})
                return result.fetchall()
    
    # Day 3: New Enhanced Database Methods
    async def execute_transaction(
        self,
        operations: List[Dict[str, Any]],
        isolation_level: str = "READ_COMMITTED",
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Execute multiple operations in a single transaction with isolation control.
        
        Args:
            operations: List of operations, each with 'query' and optional 'params'
            isolation_level: Transaction isolation level
            timeout: Transaction timeout in seconds
            
        Returns:
            List of results for each operation
        """
        results = []
        transaction_timeout = timeout or 60.0
        
        async with performance_monitor.track_operation("database_transaction", timeout=transaction_timeout):
            async with self.get_session(timeout=transaction_timeout, isolation_level=isolation_level) as session:
                try:
                    for operation in operations:
                        query = operation["query"]
                        params = operation.get("params", {})
                        
                        result = await session.execute(text(query), params)
                        results.append(result.fetchall() if result.returns_rows else result.rowcount)
                    
                    # Explicit commit for transaction
                    await session.commit()
                    
                    logger.info(
                        "database_transaction_completed",
                        operations_count=len(operations),
                        isolation_level=isolation_level
                    )
                    
                    return results
                    
                except Exception as e:
                    await session.rollback()
                    logger.error(
                        "database_transaction_failed",
                        operations_count=len(operations),
                        error=str(e),
                        isolation_level=isolation_level
                    )
                    raise ExternalServiceError(
                        service_name="PostgreSQL",
                        message=f"Transaction failed: {str(e)}"
                    )
    
    async def validate_connection(self) -> bool:
        """
        Validate database connection with immediate failure detection.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            start_time = time.time()
            
            # Quick connection test with minimal timeout
            async with asyncio.timeout(3.0):
                async with self._engine.begin() as conn:
                    result = await conn.execute(text("SELECT 1"))
                    if result.scalar() != 1:
                        return False
            
            elapsed = (time.time() - start_time) * 1000
            logger.debug(
                "database_connection_validation_passed",
                elapsed_ms=elapsed
            )
            return True
            
        except asyncio.TimeoutError:
            logger.warning("database_connection_validation_timeout")
            return False
        except Exception as e:
            logger.error(
                "database_connection_validation_failed",
                error=str(e)
            )
            return False
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        return self._circuit_breaker.get_status()
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            **self._connection_pool_stats,
            "pool_size": self._engine.pool.size() if self._engine and hasattr(self._engine.pool, 'size') else None,
            "pool_checked_out": self._engine.pool.checked_out() if self._engine and hasattr(self._engine.pool, 'checked_out') else None,
            "engine_url": str(self._engine.url) if self._engine else None,
            "initialized": self._initialized
        }
    
    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._initialized


class DocumentRepository:
    """Repository pattern for common document database operations.
    
    REFACTORED: Consolidated from multiple locations in documents.py to eliminate duplication.
    Provides centralized, reusable database operations for Document entities.
    """
    
    @staticmethod
    async def get_document_by_id(
        session: AsyncSession, 
        document_id: str, 
        include_deleted: bool = False
    ) -> Optional[Any]:
        """Get document by ID with optional deleted filter.
        
        Args:
            session: Database session
            document_id: Document UUID as string
            include_deleted: Whether to include soft-deleted documents
            
        Returns:
            Document instance or None if not found
        """
        from app.models.document import Document  # Import here to avoid circular imports
        from sqlalchemy import and_
        
        query = select(Document).where(Document.id == document_id)
        if not include_deleted:
            query = query.where(Document.is_deleted == False)
        
        result = await session.execute(query)
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_document_or_404(
        session: AsyncSession, 
        document_id: str, 
        include_deleted: bool = False
    ) -> Any:
        """Get document by ID or raise NotFoundError.
        
        Args:
            session: Database session
            document_id: Document UUID as string
            include_deleted: Whether to include soft-deleted documents
            
        Returns:
            Document instance
            
        Raises:
            NotFoundError: If document not found
        """
        from app.core.exceptions import NotFoundError
        
        document = await DocumentRepository.get_document_by_id(
            session, document_id, include_deleted
        )
        if not document:
            raise NotFoundError("Document", document_id)
        return document


# ============================================================================
# COMMON DATABASE SESSION PATTERNS - DRY CONSOLIDATION
# ============================================================================

@asynccontextmanager
async def with_db_session(
    timeout: Optional[float] = None,
    isolation_level: Optional[str] = None
):
    """
    Context manager for database sessions with Day 3 enhanced reliability.
    Consolidates repeated `async with get_db()` patterns.
    
    Args:
        timeout: Operation timeout in seconds
        isolation_level: Transaction isolation level
    
    Usage:
        async with with_db_session(timeout=30.0, isolation_level="SERIALIZABLE") as session:
            # Use session here
    """
    async with db_manager.get_session(timeout=timeout, isolation_level=isolation_level) as session:
        yield session


async def execute_with_session(
    operation, 
    *args, 
    timeout: Optional[float] = None,
    isolation_level: Optional[str] = None,
    **kwargs
):
    """
    Execute an operation with a managed database session with Day 3 reliability.
    Consolidates repeated session management patterns.
    
    Args:
        operation: Async function that takes session as first parameter
        *args: Arguments to pass to operation
        timeout: Operation timeout in seconds
        isolation_level: Transaction isolation level
        **kwargs: Keyword arguments to pass to operation
        
    Returns:
        Result of operation
    """
    async with with_db_session(timeout=timeout, isolation_level=isolation_level) as session:
        return await operation(session, *args, **kwargs)


def with_db_transaction(
    commit_on_success: bool = True,
    isolation_level: Optional[str] = None,
    timeout: Optional[float] = None
):
    """
    Decorator to add automatic database transaction management with Day 3 reliability.
    Consolidates repeated transaction patterns.
    
    Args:
        commit_on_success: Whether to auto-commit on successful completion
        isolation_level: Transaction isolation level
        timeout: Operation timeout in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract transaction parameters from kwargs if present
            func_timeout = kwargs.pop('timeout', timeout)
            func_isolation = kwargs.pop('isolation_level', isolation_level)
            
            # Extract or create session
            if 'session' in kwargs:
                session = kwargs['session']
                auto_session = False
            elif args and hasattr(args[0], '__class__') and 'session' in str(type(args[0])):
                # Check if first arg looks like it has a session
                session = args[0] if hasattr(args[0], 'execute') else None
                auto_session = False
            else:
                # Create a new session with reliability features
                async with with_db_session(timeout=func_timeout, isolation_level=func_isolation) as session:
                    kwargs['session'] = session
                    return await func(*args, **kwargs)
                auto_session = True
            
            try:
                result = await func(*args, **kwargs)
                if auto_session and commit_on_success:
                    await session.commit()
                return result
            except Exception as e:
                if auto_session:
                    await session.rollback()
                logger.error(
                    "transaction_decorator_failed",
                    function=func.__name__,
                    isolation_level=func_isolation,
                    timeout=func_timeout,
                    error=str(e)
                )
                raise
        return wrapper
    return decorator


# Global database manager instance
db_manager = DatabaseManager()

# Add repository to database manager for easy access
db_manager.document_repo = DocumentRepository()


# Dependency for FastAPI
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency to get database session.
    
    Yields:
        Database session
    """
    async with db_manager.get_session() as session:
        yield session


# Lifecycle management
async def init_database() -> None:
    """Initialize database on application startup."""
    await db_manager.initialize()


async def close_database() -> None:
    """Close database connections on application shutdown."""
    await db_manager.close()