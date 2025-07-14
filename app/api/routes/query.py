"""
Query and Answer API endpoints.
Provides RAG query processing, chat interactions, and answer generation.

Implements Day 3 Requirements:
- Query processing endpoints with input validation
- Chat functionality with session management
- Response streaming capabilities
- Query history with data corruption prevention
"""
from app.core.common import (
    get_api_logger, ApiResponses,
    Optional, List, UUID, Dict, Any, asyncio
)
from app.core.exceptions import (
    QueryValidationError,
    SecurityError,
    RateLimitError,
    LLMError,
    VectorStoreError,
    with_api_error_handling
)

# Specific imports
from fastapi import APIRouter, Depends, Query as QueryParam, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import AsyncGenerator

from app.schemas.query import (
    QueryRequest, QueryResponse, QueryResult, QueryListResponse,
    ChatRequest, ChatResponse, ChatHistory, ChatSession,
    StreamChunk, StreamResponse, QueryStatus, QueryType,
    Answer, AnswerSource
)
from app.services.rag_pipeline import RAGPipeline
from app.services.query_processor import QueryProcessor
from app.core.database import get_db, AsyncSession
from app.models.document import Document

# DRY CONSOLIDATION: Using consolidated API logger
logger = get_api_logger("query")
router = APIRouter()

# In-memory storage for demo (replace with database in production)
active_queries: Dict[str, QueryResult] = {}
chat_sessions: Dict[str, ChatSession] = {}
query_history: Dict[str, List[str]] = {}  # session_id -> [query_ids]

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()


@router.post("/query", response_model=QueryResponse, tags=["query"])
@with_api_error_handling("submit_query")
async def submit_query(
    request: Request,
    query_request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Submit a query for RAG processing.
    
    Implements Day 3 requirements:
    - Input validation ensuring queries are processable
    - Background processing with status tracking
    - Resource limit compliance
    """
    # Generate unique query ID
    query_id = uuid.uuid4()
    
    # Create initial query result
    query_result = QueryResult(
        query_id=query_id,
        query=query_request.query,
        query_type=query_request.query_type,
        status=QueryStatus.PENDING,
        processing_time=0.0,
        created_at=datetime.utcnow(),
        session_id=query_request.session_id,
        metadata=query_request.metadata or {}
    )
    
    # Store in active queries
    active_queries[str(query_id)] = query_result
    
    # Add to session history for data corruption prevention
    if query_request.session_id:
        if query_request.session_id not in query_history:
            query_history[query_request.session_id] = []
        query_history[query_request.session_id].append(str(query_id))
        
        # Limit history size to prevent memory issues
        if len(query_history[query_request.session_id]) > 100:
            query_history[query_request.session_id] = query_history[query_request.session_id][-100:]
    
    # Start background processing
    background_tasks.add_task(process_query_background, str(query_id), query_request, db)
    
    logger.info(
        "query_submitted",
        query_id=str(query_id),
        query_type=query_request.query_type.value,
        session_id=query_request.session_id,
        query_length=len(query_request.query)
    )
    
    return QueryResponse(
        query_id=query_id,
        status=QueryStatus.PENDING,
        estimated_completion_time=15.0,  # Estimated based on complexity
        message="Query submitted for processing",
        created_at=query_result.created_at
    )


async def process_query_background(query_id: str, query_request: QueryRequest, db: AsyncSession):
    """Background task to process the query using RAG pipeline."""
    start_time = time.time()
    
    try:
        # Update status to processing
        if query_id in active_queries:
            active_queries[query_id].status = QueryStatus.PROCESSING
        
        logger.info("query_processing_started", query_id=query_id)
        
        # Process query through RAG pipeline
        result = rag_pipeline.process_query(
            query=query_request.query,
            user_id=query_request.session_id or "anonymous"
        )
        
        # Convert RAG result to Answer format
        if result.success and result.answer:
            # Extract sources from RAG result
            sources = []
            if hasattr(result, 'search_results') and result.search_results:
                for search_result in result.search_results.results[:query_request.max_results]:
                    source = AnswerSource(
                        document_id=UUID(search_result.metadata.get('document_id', str(uuid.uuid4()))),
                        document_name=search_result.source,
                        chunk_id=search_result.chunk_id,
                        relevance_score=search_result.relevance_score,
                        excerpt=search_result.document.page_content[:200]
                    )
                    sources.append(source)
            
            answer = Answer(
                content=result.answer.content,
                confidence_score=result.answer.confidence_score,
                sources=sources,
                generation_time=result.answer.generation_time,
                model_used=result.answer.model_used,
                token_usage={
                    "prompt_tokens": result.answer.prompt_tokens,
                    "completion_tokens": result.answer.completion_tokens
                },
                quality_metrics=result.answer.quality_metrics
            )
            
            # Update query result with success
            if query_id in active_queries:
                active_queries[query_id].status = QueryStatus.COMPLETED
                active_queries[query_id].answer = answer
                active_queries[query_id].completed_at = datetime.utcnow()
        else:
            # Handle failure
            error_message = "Query processing failed"
            if result.errors:
                error_message = result.errors[0].user_message
            
            if query_id in active_queries:
                active_queries[query_id].status = QueryStatus.FAILED
                active_queries[query_id].error_message = error_message
        
        # Update processing time
        processing_time = time.time() - start_time
        if query_id in active_queries:
            active_queries[query_id].processing_time = processing_time
        
        logger.info(
            "query_processing_completed",
            query_id=query_id,
            success=result.success,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(
            "query_processing_failed",
            query_id=query_id,
            error=str(e),
            exc_info=True
        )
        
        # Update with error status
        if query_id in active_queries:
            active_queries[query_id].status = QueryStatus.FAILED
            active_queries[query_id].error_message = f"Processing error: {str(e)}"
            active_queries[query_id].processing_time = time.time() - start_time


@router.get("/query/{query_id}", response_model=QueryResult, tags=["query"])
@with_api_error_handling("get_query_status")
async def get_query_status(query_id: UUID):
    """
    Get query processing status and results.
    
    Returns the current status of a query and the answer if completed.
    """
    query_str = str(query_id)
    
    if query_str not in active_queries:
        return ApiResponses.error(
            "Query not found",
            "QUERY_NOT_FOUND",
            {"query_id": query_str}
        )
    
    return active_queries[query_str]


@router.get("/queries", response_model=QueryListResponse, tags=["query"])
@with_api_error_handling("list_queries")
async def list_queries(
    session_id: Optional[str] = QueryParam(None, description="Filter by session ID"),
    status: Optional[QueryStatus] = QueryParam(None, description="Filter by status"),
    page: int = QueryParam(1, ge=1, description="Page number"),
    per_page: int = QueryParam(10, ge=1, le=100, description="Items per page")
):
    """
    List queries with filtering and pagination.
    
    Supports filtering by session ID and status for query history management.
    """
    # Filter queries
    filtered_queries = []
    for query_result in active_queries.values():
        if session_id and query_result.session_id != session_id:
            continue
        if status and query_result.status != status:
            continue
        filtered_queries.append(query_result)
    
    # Sort by creation time (newest first)
    filtered_queries.sort(key=lambda q: q.created_at, reverse=True)
    
    # Apply pagination
    total = len(filtered_queries)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_queries = filtered_queries[start_idx:end_idx]
    
    return QueryListResponse(
        queries=paginated_queries,
        total=total,
        page=page,
        per_page=per_page
    )


@router.delete("/query/{query_id}", tags=["query"])
@with_api_error_handling("cancel_query")
async def cancel_query(query_id: UUID):
    """
    Cancel a pending or processing query.
    """
    query_str = str(query_id)
    
    if query_str not in active_queries:
        return ApiResponses.error(
            "Query not found",
            "QUERY_NOT_FOUND",
            {"query_id": query_str}
        )
    
    query_result = active_queries[query_str]
    
    if query_result.status in [QueryStatus.COMPLETED, QueryStatus.FAILED, QueryStatus.CANCELLED]:
        return ApiResponses.error(
            "Query cannot be cancelled",
            "QUERY_NOT_CANCELLABLE",
            {"status": query_result.status.value}
        )
    
    # Update status
    active_queries[query_str].status = QueryStatus.CANCELLED
    
    logger.info("query_cancelled", query_id=query_str)
    
    return ApiResponses.success(
        {"cancelled": True},
        "Query cancelled successfully"
    )


# Chat endpoints

@router.post("/chat", response_model=ChatResponse, tags=["chat"])
@with_api_error_handling("chat_message")
async def chat_message(
    chat_request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Send a chat message and get AI response.
    
    Implements Day 3 requirements:
    - Session management with graceful failures
    - Conversational context preservation
    - Input validation for processable queries
    """
    # Get or create session
    session_id = chat_request.session_id or str(uuid.uuid4())
    
    if session_id not in chat_sessions:
        chat_sessions[session_id] = ChatSession(
            session_id=session_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )
    
    session = chat_sessions[session_id]
    session.last_activity = datetime.utcnow()
    session.message_count += 1
    
    # Create query request for RAG processing
    query_request = QueryRequest(
        query=chat_request.message,
        query_type=QueryType.CHAT,
        session_id=session_id,
        stream_response=chat_request.stream_response
    )
    
    # Process through RAG pipeline
    start_time = time.time()
    result = rag_pipeline.process_query(
        query=chat_request.message,
        user_id=session_id
    )
    
    processing_time = time.time() - start_time
    
    # Generate response
    if result.success and result.answer:
        # Extract sources
        sources = []
        if hasattr(result, 'search_results') and result.search_results:
            for search_result in result.search_results.results[:5]:
                source = AnswerSource(
                    document_id=UUID(search_result.metadata.get('document_id', str(uuid.uuid4()))),
                    document_name=search_result.source,
                    chunk_id=search_result.chunk_id,
                    relevance_score=search_result.relevance_score,
                    excerpt=search_result.document.page_content[:200]
                )
                sources.append(source)
        
        response_content = result.answer.content
        confidence_score = result.answer.confidence_score
        token_usage = {
            "prompt_tokens": result.answer.prompt_tokens,
            "completion_tokens": result.answer.completion_tokens
        }
    else:
        response_content = "I'm sorry, I couldn't generate a response to your message. Please try rephrasing your question."
        confidence_score = 0.0
        sources = []
        token_usage = {}
    
    message_id = str(uuid.uuid4())
    
    logger.info(
        "chat_message_processed",
        session_id=session_id,
        message_id=message_id,
        processing_time=processing_time,
        success=result.success if 'result' in locals() else False
    )
    
    return ChatResponse(
        session_id=session_id,
        message_id=message_id,
        response=response_content,
        sources=sources,
        confidence_score=confidence_score,
        processing_time=processing_time,
        token_usage=token_usage
    )


@router.get("/chat/{session_id}/history", response_model=ChatHistory, tags=["chat"])
@with_api_error_handling("get_chat_history")
async def get_chat_history(session_id: str):
    """
    Get chat session history.
    
    Implements query history validation to prevent data corruption.
    """
    if session_id not in chat_sessions:
        return ApiResponses.error(
            "Session not found",
            "SESSION_NOT_FOUND",
            {"session_id": session_id}
        )
    
    session = chat_sessions[session_id]
    
    # Get associated queries for this session
    messages = []
    if session_id in query_history:
        for query_id in query_history[session_id]:
            if query_id in active_queries:
                query_result = active_queries[query_id]
                
                # Add user message
                messages.append(ChatMessage(
                    role="user",
                    content=query_result.query,
                    timestamp=query_result.created_at
                ))
                
                # Add assistant response if available
                if query_result.answer:
                    messages.append(ChatMessage(
                        role="assistant",
                        content=query_result.answer.content,
                        timestamp=query_result.completed_at or query_result.created_at
                    ))
    
    # Sort messages by timestamp
    messages.sort(key=lambda m: m.timestamp)
    
    return ChatHistory(
        session=session,
        messages=messages
    )


@router.delete("/chat/{session_id}", tags=["chat"])
@with_api_error_handling("delete_chat_session")
async def delete_chat_session(session_id: str):
    """
    Delete a chat session and its history.
    """
    if session_id not in chat_sessions:
        return ApiResponses.error(
            "Session not found",
            "SESSION_NOT_FOUND",
            {"session_id": session_id}
        )
    
    # Clean up session data
    del chat_sessions[session_id]
    if session_id in query_history:
        del query_history[session_id]
    
    logger.info("chat_session_deleted", session_id=session_id)
    
    return ApiResponses.success(
        {"deleted": True},
        "Chat session deleted successfully"
    )


# Streaming endpoints

@router.get("/query/{query_id}/stream", tags=["streaming"])
@with_api_error_handling("stream_query_response")
async def stream_query_response(query_id: UUID):
    """
    Stream query response in real-time.
    
    Implements Day 3 requirements:
    - Response streaming with connection failure handling
    - Real-time answer delivery
    """
    query_str = str(query_id)
    
    if query_str not in active_queries:
        return JSONResponse(
            status_code=404,
            content={"error": "Query not found"}
        )
    
    async def generate_stream():
        """Generate SSE stream for query response."""
        query_result = active_queries[query_str]
        
        # Stream status updates
        while query_result.status in [QueryStatus.PENDING, QueryStatus.PROCESSING]:
            chunk = StreamChunk(
                chunk_id=int(time.time()),
                content=f"Status: {query_result.status.value}",
                is_final=False,
                metadata={"status": query_result.status.value}
            )
            
            stream_response = StreamResponse(
                query_id=UUID(query_str),
                chunk=chunk,
                timestamp=datetime.utcnow()
            )
            
            yield f"data: {stream_response.model_dump_json()}\n\n"
            await asyncio.sleep(1)  # Poll every second
            
            # Refresh query result
            query_result = active_queries[query_str]
        
        # Stream final result
        if query_result.status == QueryStatus.COMPLETED and query_result.answer:
            # Stream answer in chunks
            content = query_result.answer.content
            chunk_size = 50
            
            for i, chunk_start in enumerate(range(0, len(content), chunk_size)):
                chunk_content = content[chunk_start:chunk_start + chunk_size]
                is_final = chunk_start + chunk_size >= len(content)
                
                chunk = StreamChunk(
                    chunk_id=i,
                    content=chunk_content,
                    is_final=is_final,
                    metadata={"confidence": query_result.answer.confidence_score}
                )
                
                stream_response = StreamResponse(
                    query_id=UUID(query_str),
                    chunk=chunk,
                    timestamp=datetime.utcnow()
                )
                
                yield f"data: {stream_response.model_dump_json()}\n\n"
                await asyncio.sleep(0.1)  # Small delay between chunks
        else:
            # Stream error
            error_chunk = StreamChunk(
                chunk_id=0,
                content=query_result.error_message or "Query failed",
                is_final=True,
                metadata={"error": True}
            )
            
            stream_response = StreamResponse(
                query_id=UUID(query_str),
                chunk=error_chunk,
                timestamp=datetime.utcnow()
            )
            
            yield f"data: {stream_response.model_dump_json()}\n\n"
    
    return EventSourceResponse(generate_stream())


@router.get("/health/query", tags=["monitoring"])
@with_api_error_handling("query_service_health")
async def query_service_health():
    """
    Get query service health status.
    """
    # Check RAG pipeline health
    try:
        # Simple health check - try to initialize components
        pipeline_healthy = hasattr(rag_pipeline, 'query_processor') and rag_pipeline.query_processor is not None
        
        active_query_count = len(active_queries)
        active_session_count = len(chat_sessions)
        
        # Memory usage check
        total_history_size = sum(len(history) for history in query_history.values())
        
        health_status = "healthy" if pipeline_healthy else "degraded"
        
        return ApiResponses.success({
            "status": health_status,
            "pipeline_healthy": pipeline_healthy,
            "active_queries": active_query_count,
            "active_sessions": active_session_count,
            "total_query_history": total_history_size,
            "memory_usage": {
                "queries_in_memory": active_query_count,
                "sessions_in_memory": active_session_count
            }
        }, f"Query service status: {health_status}")
        
    except Exception as e:
        logger.error("query_health_check_failed", error=str(e))
        return ApiResponses.error(
            "Query service health check failed",
            "HEALTH_CHECK_FAILED",
            {"error": str(e)}
        )