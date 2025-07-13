"""
Document database models.
Integrates with existing async SQLAlchemy setup.
"""
from typing import Optional, List
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, ForeignKey, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from app.core.database import Base


class Document(Base):
    """Document model for storing uploaded documents and metadata."""
    
    __tablename__ = "documents"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Document metadata
    filename = Column(String(255), nullable=False, index=True)
    original_filename = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False, index=True)  # .pdf, .docx, .txt
    file_size_bytes = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)  # SHA-256 hash
    
    # Processing status
    processing_status = Column(String(20), nullable=False, default="pending", index=True)
    # Status values: pending, processing, completed, failed
    processing_error = Column(Text, nullable=True)
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Document content
    content = Column(Text, nullable=True)  # Extracted text content
    content_preview = Column(String(500), nullable=True)  # First 500 chars for quick preview
    
    # Vector store information
    vector_store_id = Column(String(100), nullable=True, index=True)
    embedding_model = Column(String(100), nullable=True)
    total_chunks = Column(Integer, nullable=False, default=0)
    
    # Metadata and tags
    document_metadata = Column(JSONB, nullable=True)  # Additional metadata as JSON
    tags = Column(JSONB, nullable=True)  # Tags for categorization
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Soft delete
    is_deleted = Column(Boolean, nullable=False, default=False, index=True)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.processing_status}')>"
    
    @property
    def is_processed(self) -> bool:
        """Check if document has been successfully processed."""
        return self.processing_status == "completed"
    
    @property
    def is_processing(self) -> bool:
        """Check if document is currently being processed."""
        return self.processing_status == "processing"
    
    @property
    def has_failed(self) -> bool:
        """Check if document processing has failed."""
        return self.processing_status == "failed"
    
    @property
    def file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return round(self.file_size_bytes / 1024 / 1024, 2)


class DocumentChunk(Base):
    """Document chunk model for storing processed document chunks."""
    
    __tablename__ = "document_chunks"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Foreign key to document
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Chunk information
    chunk_index = Column(Integer, nullable=False, index=True)  # Order within document
    content = Column(Text, nullable=False)
    content_length = Column(Integer, nullable=False)
    
    # Vector information
    embedding = Column(LargeBinary, nullable=True)  # Serialized embedding vector
    embedding_model = Column(String(100), nullable=True)
    vector_store_id = Column(String(100), nullable=True, index=True)
    
    # Chunk metadata
    start_page = Column(Integer, nullable=True)  # For PDFs
    end_page = Column(Integer, nullable=True)    # For PDFs
    start_char = Column(Integer, nullable=True)  # Character position in original document
    end_char = Column(Integer, nullable=True)    # Character position in original document
    
    # Additional metadata
    chunk_metadata = Column(JSONB, nullable=True)  # Chunk-specific metadata
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"
    
    @property
    def preview(self) -> str:
        """Get a preview of the chunk content (first 100 characters)."""
        return self.content[:100] + "..." if len(self.content) > 100 else self.content