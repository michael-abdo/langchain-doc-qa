"""
Document schemas for API validation and serialization.
Follows existing Pydantic patterns for consistency.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from uuid import UUID
from pydantic import BaseModel, Field, validator


class ProcessingStatus(str, Enum):
    """Document processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentBase(BaseModel):
    """Base document schema with common fields."""
    tags: Optional[List[str]] = Field(None, description="Document tags for categorization")
    document_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional document metadata")


class DocumentCreate(DocumentBase):
    """Schema for creating a new document (typically through file upload)."""
    # Most fields will be populated during file processing
    pass


class DocumentUpdate(BaseModel):
    """Schema for updating an existing document."""
    tags: Optional[List[str]] = Field(None, description="Updated document tags")
    document_metadata: Optional[Dict[str, Any]] = Field(None, description="Updated document metadata")


class DocumentChunkResponse(BaseModel):
    """Schema for document chunk responses."""
    id: UUID = Field(..., description="Unique chunk identifier")
    chunk_index: int = Field(..., description="Order of chunk within document")
    content: str = Field(..., description="Chunk text content")
    content_length: int = Field(..., description="Length of chunk content")
    start_page: Optional[int] = Field(None, description="Starting page number (for PDFs)")
    end_page: Optional[int] = Field(None, description="Ending page number (for PDFs)")
    start_char: Optional[int] = Field(None, description="Starting character position")
    end_char: Optional[int] = Field(None, description="Ending character position")
    chunk_metadata: Optional[Dict[str, Any]] = Field(None, description="Chunk-specific metadata")
    created_at: datetime = Field(..., description="Chunk creation timestamp")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "chunk_index": 0,
                "content": "This is the first chunk of the document...",
                "content_length": 1000,
                "start_page": 1,
                "end_page": 1,
                "start_char": 0,
                "end_char": 999,
                "chunk_metadata": {"section": "introduction"},
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class DocumentResponse(DocumentBase):
    """Schema for document responses."""
    id: UUID = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Document filename")
    original_filename: str = Field(..., description="Original uploaded filename")
    file_type: str = Field(..., description="File extension (.pdf, .docx, .txt)")
    file_size_bytes: int = Field(..., description="File size in bytes")
    file_size_mb: float = Field(..., description="File size in megabytes")
    processing_status: ProcessingStatus = Field(..., description="Document processing status")
    processing_error: Optional[str] = Field(None, description="Processing error message if failed")
    processing_started_at: Optional[datetime] = Field(None, description="When processing started")
    processing_completed_at: Optional[datetime] = Field(None, description="When processing completed")
    content_preview: Optional[str] = Field(None, description="Preview of document content")
    total_chunks: int = Field(..., description="Number of chunks created")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    created_at: datetime = Field(..., description="Document creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    chunks: Optional[List[DocumentChunkResponse]] = Field(None, description="Document chunks")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "sample_document.pdf",
                "original_filename": "Sample Document.pdf",
                "file_type": ".pdf",
                "file_size_bytes": 1048576,
                "file_size_mb": 1.0,
                "processing_status": "completed",
                "processing_error": None,
                "processing_started_at": "2024-01-15T10:30:00Z",
                "processing_completed_at": "2024-01-15T10:32:00Z",
                "content_preview": "This is a sample document...",
                "total_chunks": 5,
                "embedding_model": "text-embedding-ada-002",
                "tags": ["research", "important"],
                "document_metadata": {"author": "John Doe"},
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:32:00Z"
            }
        }


class DocumentListResponse(BaseModel):
    """Schema for paginated document list responses."""
    documents: List[DocumentResponse] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Documents per page")
    pages: int = Field(..., description="Total number of pages")
    
    @validator('pages', always=True)
    def calculate_pages(cls, v, values):
        """Calculate total pages based on total and per_page."""
        total = values.get('total', 0)
        per_page = values.get('per_page', 1)
        return (total + per_page - 1) // per_page if per_page > 0 else 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [],
                "total": 25,
                "page": 1,
                "per_page": 10,
                "pages": 3
            }
        }


class DocumentUploadResponse(BaseModel):
    """Schema for document upload responses."""
    document_id: UUID = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Document filename")
    file_size_mb: float = Field(..., description="File size in megabytes")
    processing_status: ProcessingStatus = Field(..., description="Initial processing status")
    message: str = Field(..., description="Upload success message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "filename": "sample_document.pdf",
                "file_size_mb": 1.0,
                "processing_status": "pending",
                "message": "Document uploaded successfully and queued for processing"
            }
        }


class DocumentProcessingStatus(BaseModel):
    """Schema for document processing status responses."""
    document_id: UUID = Field(..., description="Unique document identifier")
    processing_status: ProcessingStatus = Field(..., description="Current processing status")
    processing_error: Optional[str] = Field(None, description="Processing error message if failed")
    processing_started_at: Optional[datetime] = Field(None, description="When processing started")
    processing_completed_at: Optional[datetime] = Field(None, description="When processing completed")
    total_chunks: int = Field(..., description="Number of chunks created")
    progress_percentage: Optional[float] = Field(None, description="Processing progress (0-100)")
    
    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "processing_status": "processing",
                "processing_error": None,
                "processing_started_at": "2024-01-15T10:30:00Z",
                "processing_completed_at": None,
                "total_chunks": 3,
                "progress_percentage": 60.0
            }
        }


class DocumentSearchRequest(BaseModel):
    """Schema for document search requests."""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    file_types: Optional[List[str]] = Field(None, description="Filter by file types")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    processing_status: Optional[ProcessingStatus] = Field(None, description="Filter by processing status")
    page: int = Field(1, ge=1, description="Page number")
    per_page: int = Field(10, ge=1, le=100, description="Results per page")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning",
                "file_types": [".pdf", ".docx"],
                "tags": ["research"],
                "processing_status": "completed",
                "page": 1,
                "per_page": 10
            }
        }