"""
Core configuration module with environment variable validation.
Follows fail-fast principle - validates all config at startup.
"""
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator, ValidationError
import sys
import os


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Application settings
    APP_NAME: str = Field(default="LangChain Document Q&A", description="Application name")
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    DEBUG: bool = Field(default=False, description="Debug mode")
    
    # API settings
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port", ge=1, le=65535)
    API_PREFIX: str = Field(default="/api/v1", description="API prefix")
    
    # Security settings
    SECRET_KEY: str = Field(..., description="Secret key for JWT tokens")
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Token expiration time")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins"
    )
    
    # Database settings (for future PostgreSQL)
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="PostgreSQL connection URL"
    )
    
    # Vector store settings
    VECTOR_STORE_TYPE: str = Field(
        default="faiss",
        description="Vector store type (faiss, chroma, pgvector)"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-ada-002",
        description="OpenAI embedding model"
    )
    
    # LLM settings
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, description="Anthropic API key")
    LLM_PROVIDER: str = Field(default="openai", description="LLM provider (openai, anthropic)")
    LLM_MODEL: str = Field(default="gpt-4-turbo-preview", description="LLM model name")
    LLM_TEMPERATURE: float = Field(default=0.7, description="LLM temperature", ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=2000, description="Max tokens for LLM response", ge=1)
    
    # Document processing settings
    CHUNK_SIZE: int = Field(default=1000, description="Document chunk size", ge=100)
    CHUNK_OVERLAP: int = Field(default=200, description="Document chunk overlap", ge=0)
    MAX_FILE_SIZE_MB: int = Field(default=10, description="Max file size in MB", ge=1)
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=[".pdf", ".docx", ".txt"],
        description="Allowed file extensions"
    )
    
    # Logging settings
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format (json, text)")
    
    # AWS settings (for deployment)
    AWS_REGION: Optional[str] = Field(default=None, description="AWS region")
    AWS_ACCESS_KEY_ID: Optional[str] = Field(default=None, description="AWS access key")
    AWS_SECRET_ACCESS_KEY: Optional[str] = Field(default=None, description="AWS secret key")
    S3_BUCKET_NAME: Optional[str] = Field(default=None, description="S3 bucket for file storage")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="forbid",  # Fail on unknown environment variables
        validate_default=True,
        use_enum_values=True
    )
    
    @validator("OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    def validate_api_keys(cls, v, values):
        """Ensure at least one LLM API key is provided."""
        if v is None and values.get("LLM_PROVIDER") in ["openai", "anthropic"]:
            # Check if the other key is provided
            if values.get("OPENAI_API_KEY") is None and values.get("ANTHROPIC_API_KEY") is None:
                raise ValueError("At least one LLM API key must be provided")
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        """Validate PostgreSQL connection URL format."""
        if v and not v.startswith(("postgresql://", "postgres://")):
            raise ValueError("Database URL must be a valid PostgreSQL connection string")
        return v
    
    @validator("CHUNK_OVERLAP")
    def validate_chunk_overlap(cls, v, values):
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = values.get("CHUNK_SIZE", 1000)
        if v >= chunk_size:
            raise ValueError(f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})")
        return v


    def validate_llm_health(self) -> Dict[str, Any]:
        """
        Centralized LLM health validation logic.
        Used by both startup validation and health checks.
        """
        has_key = False
        if self.LLM_PROVIDER == "openai":
            has_key = bool(self.OPENAI_API_KEY)
        elif self.LLM_PROVIDER == "anthropic":
            has_key = bool(self.ANTHROPIC_API_KEY)
        
        return {
            "status": "healthy" if has_key else "unhealthy",
            "provider": self.LLM_PROVIDER,
            "model": self.LLM_MODEL,
            "api_key_configured": has_key,
            "message": "LLM provider is configured" if has_key else "LLM API key not configured"
        }
    
    def validate_database_health(self) -> Dict[str, Any]:
        """
        Centralized database health validation logic.
        Used by health checks and startup validation.
        """
        # TODO: Implement actual database health check
        return {
            "status": "not_configured" if not self.DATABASE_URL else "healthy",
            "message": "Database not configured" if not self.DATABASE_URL else "Database is healthy",
            "url_configured": bool(self.DATABASE_URL)
        }
    
    def validate_vector_store_health(self) -> Dict[str, Any]:
        """
        Centralized vector store health validation logic.
        Used by health checks and startup validation.
        """
        # TODO: Implement actual vector store health check
        return {
            "status": "healthy",
            "type": self.VECTOR_STORE_TYPE,
            "message": f"{self.VECTOR_STORE_TYPE} vector store is ready"
        }

    def validate_critical_startup_config(self) -> None:
        """
        Validates critical configuration at startup.
        Raises ConfigurationError if validation fails.
        """
        from app.core.exceptions import ConfigurationError
        
        llm_health = self.validate_llm_health()
        if llm_health["status"] != "healthy":
            provider = self.LLM_PROVIDER
            if provider == "openai":
                raise ConfigurationError("OpenAI API key is required when using OpenAI provider")
            elif provider == "anthropic":
                raise ConfigurationError("Anthropic API key is required when using Anthropic provider")
            else:
                raise ConfigurationError(f"Invalid LLM provider: {provider}")

    @staticmethod
    def create_health_response(
        service_name: str,
        is_healthy: bool,
        details: Dict[str, Any],
        message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Standardized health check response factory.
        
        REFACTORED: Consolidated from multiple locations to eliminate duplicated health response patterns.
        Provides consistent structure for all health check responses across the application.
        
        Args:
            service_name: Name of the service being checked
            is_healthy: Boolean indicating if service is healthy
            details: Dictionary of detailed health information
            message: Optional custom message (defaults to generated message)
            
        Returns:
            Standardized health response dictionary
        """
        status = "healthy" if is_healthy else "unhealthy"
        default_message = f"{service_name} is {'operational' if is_healthy else 'experiencing issues'}"
        
        return {
            "status": status,
            "message": message or default_message,
            "details": details
        }


def load_settings() -> Settings:
    """
    Load and validate settings.
    Fails fast if configuration is invalid.
    """
    try:
        settings = Settings()
        return settings
    except ValidationError as e:
        print("❌ Configuration Error - Invalid settings detected:")
        for error in e.errors():
            field = " -> ".join(str(x) for x in error["loc"])
            msg = error["msg"]
            print(f"  - {field}: {msg}")
        print("\n💡 Please check your .env file and environment variables")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected configuration error: {str(e)}")
        sys.exit(1)


# Create a singleton instance
settings = load_settings()

# Export commonly used settings
DEBUG = settings.DEBUG
SECRET_KEY = settings.SECRET_KEY
API_PREFIX = settings.API_PREFIX