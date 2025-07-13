"""
Document chunking service.
Handles intelligent splitting of documents into chunks for vector storage.
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Represents a document chunk."""
    content: str
    start_char: int
    end_char: int
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def length(self) -> int:
        """Get chunk content length."""
        return len(self.content)


class ChunkerInterface(ABC):
    """Abstract interface for document chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into chunks."""
        pass


class RecursiveCharacterTextSplitter(ChunkerInterface):
    """
    Recursive character-based text splitter.
    Tries to split on sentences, then words, then characters.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "! ",    # Exclamations
            "? ",    # Questions
            "; ",    # Semicolons
            ", ",    # Commas
            " ",     # Words
            ""       # Characters
        ]
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into chunks using recursive character splitting."""
        if not text.strip():
            return []
        
        chunks = []
        current_chunks = [text]
        
        for separator in self.separators:
            new_chunks = []
            
            for chunk in current_chunks:
                if len(chunk) <= self.chunk_size:
                    new_chunks.append(chunk)
                else:
                    # Split this chunk
                    split_chunks = self._split_text_with_separator(chunk, separator)
                    new_chunks.extend(split_chunks)
            
            current_chunks = new_chunks
            
            # Check if all chunks are small enough
            if all(len(chunk) <= self.chunk_size for chunk in current_chunks):
                break
        
        # Create Chunk objects with proper positioning
        result_chunks = []
        current_position = 0
        
        for i, chunk_content in enumerate(current_chunks):
            if not chunk_content.strip():
                continue
            
            # Find the actual position in original text
            start_pos = text.find(chunk_content, current_position)
            if start_pos == -1:
                # Fallback: use approximate position
                start_pos = current_position
            
            end_pos = start_pos + len(chunk_content)
            
            chunk = Chunk(
                content=chunk_content.strip(),
                start_char=start_pos,
                end_char=end_pos,
                metadata=metadata.copy() if metadata else {}
            )
            
            result_chunks.append(chunk)
            current_position = max(current_position, end_pos - self.chunk_overlap)
        
        # Merge overlapping chunks if needed
        final_chunks = self._merge_small_chunks(result_chunks)
        
        logger.info(
            "text_chunked",
            original_length=len(text),
            chunks_created=len(final_chunks),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        return final_chunks
    
    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """Split text using a specific separator."""
        if not separator:
            # Character-level splitting
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
        splits = text.split(separator)
        
        # If no splits occurred, return original text
        if len(splits) == 1:
            return [text]
        
        # Reconstruct chunks while respecting size limits
        chunks = []
        current_chunk = ""
        
        for split in splits:
            test_chunk = current_chunk + (separator if current_chunk else "") + split
            
            if len(test_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = test_chunk
            else:
                # Current chunk is full, start a new one
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = split
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks that are too small."""
        if not chunks:
            return []
        
        min_chunk_size = max(100, self.chunk_size // 10)  # Minimum 100 chars or 10% of chunk size
        merged_chunks = []
        current_chunk = None
        
        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
            elif (
                len(current_chunk.content) < min_chunk_size and 
                len(current_chunk.content) + len(chunk.content) <= self.chunk_size
            ):
                # Merge with current chunk
                merged_content = current_chunk.content + "\n\n" + chunk.content
                current_chunk = Chunk(
                    content=merged_content,
                    start_char=current_chunk.start_char,
                    end_char=chunk.end_char,
                    start_page=current_chunk.start_page,
                    end_page=chunk.end_page,
                    metadata=current_chunk.metadata
                )
            else:
                # Finalize current chunk and start new one
                merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            merged_chunks.append(current_chunk)
        
        return merged_chunks


class SentenceAwareTextSplitter(ChunkerInterface):
    """
    Sentence-aware text splitter.
    Attempts to keep sentences intact while respecting chunk size limits.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Improved sentence boundary detection
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into chunks while preserving sentence boundaries."""
        if not text.strip():
            return []
        
        # Split into sentences
        sentences = self.sentence_pattern.split(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(test_chunk) <= self.chunk_size or not current_chunk:
                current_chunk = test_chunk
            else:
                # Finalize current chunk
                if current_chunk:
                    chunk_end = current_start + len(current_chunk)
                    chunk = Chunk(
                        content=current_chunk,
                        start_char=current_start,
                        end_char=chunk_end,
                        metadata=metadata.copy() if metadata else {}
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    current_start = max(0, chunk_end - self.chunk_overlap)
                
                current_chunk = sentence
        
        # Add final chunk
        if current_chunk:
            chunk_end = current_start + len(current_chunk)
            chunk = Chunk(
                content=current_chunk,
                start_char=current_start,
                end_char=chunk_end,
                metadata=metadata.copy() if metadata else {}
            )
            chunks.append(chunk)
        
        logger.info(
            "sentence_aware_chunking_completed",
            original_length=len(text),
            sentences_found=len(sentences),
            chunks_created=len(chunks)
        )
        
        return chunks


class DocumentChunkingService:
    """Main service for document chunking operations."""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        
        # Default to recursive character splitting
        self.default_chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Sentence-aware chunker for structured documents
        self.sentence_chunker = SentenceAwareTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def chunk_document(
        self,
        text: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = "recursive"
    ) -> List[Chunk]:
        """
        Chunk a document using specified strategy.
        
        Args:
            text: Document text to chunk
            document_metadata: Metadata to attach to each chunk
            chunking_strategy: Strategy to use ("recursive" or "sentence")
            
        Returns:
            List of document chunks
        """
        if not text.strip():
            logger.warning("empty_text_provided_for_chunking")
            return []
        
        # Choose chunker based on strategy
        if chunking_strategy == "sentence":
            chunker = self.sentence_chunker
        else:
            chunker = self.default_chunker
        
        # Add document-level metadata
        chunk_metadata = document_metadata.copy() if document_metadata else {}
        chunk_metadata.update({
            "chunking_strategy": chunking_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        })
        
        chunks = chunker.chunk_text(text, chunk_metadata)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata = chunk.metadata or {}
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        logger.info(
            "document_chunking_completed",
            text_length=len(text),
            chunks_created=len(chunks),
            strategy=chunking_strategy,
            avg_chunk_size=sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0
        )
        
        return chunks
    
    def chunk_with_page_info(
        self,
        text: str,
        page_breaks: List[int],
        document_metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = "recursive"
    ) -> List[Chunk]:
        """
        Chunk document while preserving page information.
        
        Args:
            text: Document text
            page_breaks: List of character positions where pages break
            document_metadata: Document metadata
            chunking_strategy: Chunking strategy
            
        Returns:
            List of chunks with page information
        """
        chunks = self.chunk_document(text, document_metadata, chunking_strategy)
        
        # Add page information to chunks
        for chunk in chunks:
            start_page = self._find_page_for_position(chunk.start_char, page_breaks)
            end_page = self._find_page_for_position(chunk.end_char, page_breaks)
            
            chunk.start_page = start_page
            chunk.end_page = end_page
            
            if chunk.metadata:
                chunk.metadata["start_page"] = start_page
                chunk.metadata["end_page"] = end_page
        
        return chunks
    
    def _find_page_for_position(self, position: int, page_breaks: List[int]) -> int:
        """Find which page a character position belongs to."""
        page = 1
        for break_pos in sorted(page_breaks):
            if position >= break_pos:
                page += 1
            else:
                break
        return page
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunks."""
        if not chunks:
            return {"total_chunks": 0}
        
        chunk_sizes = [len(chunk.content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "avg_chunk_size": sum(chunk_sizes) / len(chunks),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunk_size_target": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }


# Global chunking service instance
chunking_service = DocumentChunkingService()