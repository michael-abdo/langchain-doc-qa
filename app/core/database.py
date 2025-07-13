"""
Database connection abstraction layer.
Provides async database operations with connection pooling and health checks.
"""
from typing import AsyncGenerator, Optional, Dict, Any
from contextlib import asynccontextmanager
import asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    AsyncEngine, 
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base
from sqlalchemy import text, select
from sqlalchemy.pool import NullPool
import asyncpg

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import ConfigurationError, ExternalServiceError

logger = get_logger(__name__)

# Base class for all models
Base = declarative_base()

class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self):
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker] = None
        self._initialized = False
    
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
        """Test database connection with a simple query."""
        async with self._engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            if not result.scalar():
                raise ExternalServiceError(
                    service_name="PostgreSQL",
                    message="Database connection test failed"
                )
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("database_closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get database session with automatic transaction handling.
        
        Yields:
            AsyncSession: Database session
            
        Raises:
            ExternalServiceError: If database operation fails
        """
        if not self._initialized:
            await self.initialize()
        
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
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
                await session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.
        
        Returns:
            Dict containing health status and details
        """
        if not settings.DATABASE_URL:
            return {
                "status": "not_configured",
                "message": "Database URL not configured"
            }
        
        try:
            if not self._initialized:
                await self.initialize()
            
            # Execute health check query
            async with self._engine.begin() as conn:
                result = await conn.execute(
                    text("SELECT version(), current_database(), pg_is_in_recovery()")
                )
                row = result.fetchone()
                
                return {
                    "status": "healthy",
                    "message": "Database is operational",
                    "details": {
                        "version": row[0],
                        "database": row[1],
                        "is_replica": row[2],
                        "pool_size": self._engine.pool.size() if hasattr(self._engine.pool, 'size') else None,
                        "pool_checked_out": self._engine.pool.checked_out() if hasattr(self._engine.pool, 'checked_out') else None
                    }
                }
                
        except Exception as e:
            logger.error("database_health_check_failed", error=str(e))
            return {
                "status": "unhealthy",
                "message": f"Database health check failed: {str(e)}"
            }
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute a raw SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query result
        """
        async with self.get_session() as session:
            result = await session.execute(text(query), params or {})
            return result.fetchall()
    
    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized."""
        return self._initialized


# Global database manager instance
db_manager = DatabaseManager()


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