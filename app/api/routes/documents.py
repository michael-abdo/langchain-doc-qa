"""
Document management API endpoints.
Provides document upload, processing, and retrieval functionality.
"""
from typing import List, Optional
from uuid import UUID
import asyncio

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, func, and_, or_
from sqlalchemy.orm import selectinload

from app.core.database import get_db, db_manager
from app.core.logging import get_logger
from app.core.exceptions import (
    DocumentProcessingError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    NotFoundError,
    ValidationError
)
from app.models.document import Document, DocumentChunk
from app.schemas.document import (
    DocumentResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    DocumentProcessingStatus,
    DocumentUpdate,
    DocumentSearchRequest,
    ProcessingStatus
)
from app.services.document_processor import document_processor
from app.services.vector_store import vector_store_manager
from app.services.chunking import chunking_service

logger = get_logger(__name__)
router = APIRouter()


async def process_document_background(document_id: str):
    """Background task to process uploaded document."""
    try:
        async with get_db() as session:
            # Get document
            result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            document = result.scalar_one_or_none()
            if not document:
                logger.error("document_not_found_for_processing", document_id=document_id)
                return
            
            # Update status to processing
            await document_processor.update_processing_status(
                session, document_id, ProcessingStatus.PROCESSING
            )
            await session.commit()
            
            # Simulate file loading (in real app, load from storage)
            sample_content = f"Sample content for document {document.original_filename}"
            
            # Extract text content
            extracted_text, extraction_error = await document_processor.extract_text_content(
                document.original_filename, sample_content.encode()
            )
            
            if extraction_error:
                await document_processor.update_processing_status(
                    session, document_id, ProcessingStatus.FAILED, extraction_error
                )
                await session.commit()
                return
            
            # Create chunks
            chunks = chunking_service.chunk_document(
                text=extracted_text,
                document_metadata={
                    "document_id": str(document.id),
                    "filename": document.filename,
                    "file_type": document.file_type
                }
            )
            
            # Store chunks in database
            for i, chunk in enumerate(chunks):
                db_chunk = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk.content,
                    content_length=len(chunk.content),
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    chunk_metadata=chunk.metadata
                )
                session.add(db_chunk)
            
            # Update document with results
            document.content = extracted_text
            document.content_preview = extracted_text[:500] if extracted_text else None
            document.total_chunks = len(chunks)
            document.embedding_model = vector_store_manager.embedding_service.model_name
            
            # Create embeddings and store in vector store
            if chunks:
                chunk_texts = [chunk.content for chunk in chunks]
                chunk_ids = [f"{document.id}_{i}" for i in range(len(chunks))]
                chunk_metadatas = [
                    {
                        "document_id": str(document.id),
                        "chunk_index": i,
                        "filename": document.filename
                    }
                    for i in range(len(chunks))
                ]
                
                await vector_store_manager.add_texts(
                    texts=chunk_texts,
                    metadatas=chunk_metadatas,
                    ids=chunk_ids
                )
            
            # Mark as completed
            await document_processor.update_processing_status(
                session, document_id, ProcessingStatus.COMPLETED
            )
            await session.commit()
            
            logger.info(
                "document_processing_completed",
                document_id=document_id,
                chunks_created=len(chunks)
            )
            
    except Exception as e:
        logger.error(
            "document_processing_failed",
            document_id=document_id,
            error=str(e),
            exc_info=True
        )
        
        # Mark as failed
        async with get_db() as session:
            await document_processor.update_processing_status(
                session, document_id, ProcessingStatus.FAILED, str(e)
            )
            await session.commit()


@router.post("/documents/upload", response_model=DocumentUploadResponse, tags=["documents"])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a document for processing.
    Accepts PDF, DOCX, and TXT files.
    """
    try:
        # Read file content
        content = await file.read()
        
        # Validate file
        await document_processor.validate_file(file.filename, content)
        
        # Generate safe filename
        safe_filename = document_processor.generate_safe_filename(file.filename)
        
        # Parse tags
        parsed_tags = []
        if tags:
            parsed_tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        # Create document record
        document = await document_processor.create_document_record(
            session=db,
            filename=safe_filename,
            original_filename=file.filename,
            content=content,
            document_data=None  # Could add DocumentCreate if needed
        )
        
        # Add tags if provided
        if parsed_tags:
            document.tags = parsed_tags
        
        await db.commit()
        
        # Start background processing
        background_tasks.add_task(process_document_background, str(document.id))
        
        logger.info(
            "document_uploaded",
            document_id=str(document.id),
            filename=file.filename,
            size_bytes=len(content)
        )
        
        return DocumentUploadResponse(
            document_id=document.id,
            filename=safe_filename,
            file_size_mb=document.file_size_mb,
            processing_status=ProcessingStatus.PENDING,
            message="Document uploaded successfully and queued for processing"
        )
        
    except (FileTooLargeError, UnsupportedFileTypeError, DocumentProcessingError) as e:
        logger.warning("document_upload_validation_failed", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("document_upload_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to upload document")


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    processing_status: Optional[ProcessingStatus] = Query(None, description="Filter by processing status"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    db: AsyncSession = Depends(get_db)
):
    """List documents with filtering and pagination."""
    try:
        # Build query
        query = select(Document).where(Document.is_deleted == False)
        
        # Apply filters
        if file_type:
            query = query.where(Document.file_type == file_type)
        
        if processing_status:
            query = query.where(Document.processing_status == processing_status.value)
        
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
            if tag_list:
                # PostgreSQL JSONB contains operation
                tag_conditions = [Document.tags.contains([tag]) for tag in tag_list]
                query = query.where(or_(*tag_conditions))
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * per_page
        query = query.offset(offset).limit(per_page)
        query = query.order_by(Document.created_at.desc())
        
        # Execute query
        result = await db.execute(query)
        documents = result.scalars().all()
        
        # Convert to response models
        document_responses = [
            DocumentResponse.model_validate(doc) for doc in documents
        ]
        
        return DocumentListResponse(
            documents=document_responses,
            total=total,
            page=page,
            per_page=per_page
        )
        
    except Exception as e:
        logger.error("list_documents_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.get("/documents/{document_id}", response_model=DocumentResponse, tags=["documents"])
async def get_document(
    document_id: UUID,
    include_chunks: bool = Query(False, description="Include document chunks in response"),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific document by ID."""
    try:
        # REFACTORED: Using DocumentRepository to eliminate duplicated query pattern
        if include_chunks:
            # For chunks, need custom query with selectinload
            query = select(Document).where(
                and_(Document.id == document_id, Document.is_deleted == False)
            ).options(selectinload(Document.chunks))
            result = await db.execute(query)
            document = result.scalar_one_or_none()
            if not document:
                raise NotFoundError("Document", document_id)
        else:
            # Use repository for simple document retrieval
            document = await db_manager.document_repo.get_document_or_404(
                db, str(document_id)
            )
        
        return DocumentResponse.model_validate(document)
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    except Exception as e:
        logger.error("get_document_failed", document_id=str(document_id), error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@router.put("/documents/{document_id}", response_model=DocumentResponse, tags=["documents"])
async def update_document(
    document_id: UUID,
    document_update: DocumentUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update document metadata (tags, metadata)."""
    try:
        # REFACTORED: Using DocumentRepository to eliminate duplicated query pattern
        document = await db_manager.document_repo.get_document_or_404(
            db, str(document_id)
        )
        
        # Update fields
        update_data = {}
        if document_update.tags is not None:
            update_data["tags"] = document_update.tags
        if document_update.document_metadata is not None:
            update_data["document_metadata"] = document_update.document_metadata
        
        if update_data:
            await db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(**update_data)
            )
            await db.commit()
        
        # Return updated document
        await db.refresh(document)
        return DocumentResponse.model_validate(document)
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    except Exception as e:
        logger.error("update_document_failed", document_id=str(document_id), error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update document")


@router.delete("/documents/{document_id}", tags=["documents"])
async def delete_document(
    document_id: UUID,
    hard_delete: bool = Query(False, description="Permanently delete (vs soft delete)"),
    db: AsyncSession = Depends(get_db)
):
    """Delete a document (soft delete by default)."""
    try:
        # REFACTORED: Using DocumentRepository to eliminate duplicated query pattern
        document = await db_manager.document_repo.get_document_or_404(
            db, str(document_id)
        )
        
        if hard_delete:
            # Hard delete - remove from vector store and database
            try:
                # Remove from vector store
                chunk_ids = [f"{document_id}_{i}" for i in range(document.total_chunks)]
                await vector_store_manager.delete_texts(chunk_ids)
            except Exception as e:
                logger.warning("vector_store_deletion_failed", document_id=str(document_id), error=str(e))
            
            # Delete from database
            await db.delete(document)
        else:
            # Soft delete
            document.is_deleted = True
            document.deleted_at = func.now()
        
        await db.commit()
        
        logger.info(
            "document_deleted",
            document_id=str(document_id),
            hard_delete=hard_delete
        )
        
        return {"message": "Document deleted successfully"}
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    except Exception as e:
        logger.error("delete_document_failed", document_id=str(document_id), error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.get("/documents/{document_id}/status", response_model=DocumentProcessingStatus, tags=["documents"])
async def get_document_processing_status(
    document_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """Get document processing status."""
    try:
        # REFACTORED: Using DocumentRepository to eliminate duplicated query pattern
        document = await db_manager.document_repo.get_document_or_404(
            db, str(document_id)
        )
        
        # Calculate progress percentage
        progress = None
        if document.processing_status == ProcessingStatus.COMPLETED.value:
            progress = 100.0
        elif document.processing_status == ProcessingStatus.PROCESSING.value:
            progress = 50.0  # Simplified progress calculation
        elif document.processing_status == ProcessingStatus.FAILED.value:
            progress = 0.0
        
        return DocumentProcessingStatus(
            document_id=document.id,
            processing_status=ProcessingStatus(document.processing_status),
            processing_error=document.processing_error,
            processing_started_at=document.processing_started_at,
            processing_completed_at=document.processing_completed_at,
            total_chunks=document.total_chunks,
            progress_percentage=progress
        )
        
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    except Exception as e:
        logger.error("get_document_status_failed", document_id=str(document_id), error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get document status")


@router.post("/documents/search", tags=["documents"])
async def search_documents(
    search_request: DocumentSearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """Search documents using vector similarity."""
    try:
        # Search in vector store
        search_results = await vector_store_manager.search(
            query=search_request.query,
            k=search_request.per_page,
            filter_dict={
                "filename": search_request.file_types[0] if search_request.file_types and len(search_request.file_types) == 1 else None
            } if search_request.file_types else None
        )
        
        # Extract document IDs from results
        document_ids = []
        for result in search_results:
            doc_id = result.get("metadata", {}).get("document_id")
            if doc_id:
                document_ids.append(doc_id)
        
        # Get corresponding documents from database
        documents = []
        if document_ids:
            query = select(Document).where(
                and_(
                    Document.id.in_(document_ids),
                    Document.is_deleted == False
                )
            )
            
            # Apply additional filters
            if search_request.processing_status:
                query = query.where(Document.processing_status == search_request.processing_status.value)
            
            result = await db.execute(query)
            documents = result.scalars().all()
        
        # Convert to response format
        response_data = []
        for doc in documents:
            doc_data = DocumentResponse.model_validate(doc).model_dump()
            
            # Add search score if available
            for search_result in search_results:
                if search_result.get("metadata", {}).get("document_id") == str(doc.id):
                    doc_data["search_score"] = search_result.get("score")
                    break
            
            response_data.append(doc_data)
        
        return {
            "results": response_data,
            "total": len(response_data),
            "query": search_request.query,
            "search_type": "vector_similarity"
        }
        
    except Exception as e:
        logger.error("search_documents_failed", query=search_request.query, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to search documents")