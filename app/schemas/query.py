"""
Query and Answer API schemas.
Defines request/response models for RAG query processing endpoints.
"""
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, validator
from enum import Enum


class QueryStatus(str, Enum):
    """Query processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class QueryType(str, Enum):
    """Type of query being processed."""
    QUESTION = "question"
    SEARCH = "search"
    SUMMARY = "summary"
    CHAT = "chat"


class QueryRequest(BaseModel):
    """Request model for submitting a query."""
    query: str = Field(..., min_length=1, max_length=1000, description="The question or query to process")
    query_type: QueryType = Field(default=QueryType.QUESTION, description="Type of query being submitted")
    context_documents: Optional[List[UUID]] = Field(None, description="Specific documents to search within")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of search results to consider")
    include_sources: bool = Field(default=True, description="Whether to include source citations in response")
    stream_response: bool = Field(default=False, description="Whether to stream the response in real-time")
    session_id: Optional[str] = Field(None, description="Session ID for conversational context")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional query metadata")

    @validator('query')
    def validate_query_content(cls, v):
        """Validate query content."""
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        # Basic SQL injection prevention
        dangerous_patterns = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
        query_upper = v.upper()
        for pattern in dangerous_patterns:
            if pattern in query_upper:
                raise ValueError(f'Query contains potentially dangerous content: {pattern}')
        return v.strip()


class QueryResponse(BaseModel):
    """Response model for query submission."""
    query_id: UUID = Field(..., description="Unique identifier for the query")
    status: QueryStatus = Field(..., description="Current processing status")
    estimated_completion_time: Optional[float] = Field(None, description="Estimated seconds until completion")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Query creation timestamp")


class AnswerSource(BaseModel):
    """Source citation for an answer."""
    document_id: UUID = Field(..., description="Source document ID")
    document_name: str = Field(..., description="Source document filename")
    chunk_id: str = Field(..., description="Specific chunk reference")
    relevance_score: float = Field(..., ge=0, le=1, description="Relevance score for this source")
    page_number: Optional[int] = Field(None, description="Page number if available")
    excerpt: str = Field(..., description="Relevant text excerpt")


class Answer(BaseModel):
    """Generated answer model."""
    content: str = Field(..., description="The generated answer content")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in the answer accuracy")
    sources: List[AnswerSource] = Field(default_factory=list, description="Source citations")
    generation_time: float = Field(..., description="Time taken to generate answer in seconds")
    model_used: str = Field(..., description="LLM model used for generation")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Answer quality metrics")


class QueryResult(BaseModel):
    """Complete query result model."""
    query_id: UUID = Field(..., description="Query identifier")
    query: str = Field(..., description="Original query text")
    query_type: QueryType = Field(..., description="Type of query processed")
    status: QueryStatus = Field(..., description="Processing status")
    answer: Optional[Answer] = Field(None, description="Generated answer (if completed)")
    error_message: Optional[str] = Field(None, description="Error message (if failed)")
    processing_time: float = Field(..., description="Total processing time in seconds")
    created_at: datetime = Field(..., description="Query creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Query completion timestamp")
    session_id: Optional[str] = Field(None, description="Associated session ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class QueryListResponse(BaseModel):
    """Response model for listing queries."""
    queries: List[QueryResult] = Field(..., description="List of query results")
    total: int = Field(..., description="Total number of queries")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")


# Chat-specific schemas

class ChatMessage(BaseModel):
    """Individual chat message."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatSession(BaseModel):
    """Chat session model."""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(None, description="Associated user ID")
    title: Optional[str] = Field(None, description="Session title")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    message_count: int = Field(default=0, description="Number of messages in session")
    is_active: bool = Field(default=True, description="Whether session is active")


class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID (creates new if not provided)")
    include_context: bool = Field(default=True, description="Whether to include conversation context")
    max_context_messages: int = Field(default=10, ge=1, le=50, description="Maximum context messages to include")
    stream_response: bool = Field(default=False, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    session_id: str = Field(..., description="Session identifier")
    message_id: str = Field(..., description="Unique message identifier")
    response: str = Field(..., description="Assistant response")
    sources: List[AnswerSource] = Field(default_factory=list, description="Source citations")
    confidence_score: float = Field(..., ge=0, le=1, description="Response confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Token usage statistics")


class ChatHistory(BaseModel):
    """Chat session history."""
    session: ChatSession = Field(..., description="Session information")
    messages: List[ChatMessage] = Field(..., description="Message history")


# Streaming response models

class StreamChunk(BaseModel):
    """Individual chunk in a streaming response."""
    chunk_id: int = Field(..., description="Chunk sequence number")
    content: str = Field(..., description="Partial content")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class StreamResponse(BaseModel):
    """Streaming response wrapper."""
    query_id: UUID = Field(..., description="Query identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    chunk: StreamChunk = Field(..., description="Content chunk")
    timestamp: datetime = Field(..., description="Chunk timestamp")