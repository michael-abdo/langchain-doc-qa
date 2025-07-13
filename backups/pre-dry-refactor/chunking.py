"""
Document chunking service.
Handles intelligent splitting of documents into chunks for vector storage.
"""
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
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
    quality_score: Optional[float] = None
    content_hash: Optional[str] = None
    
    @property
    def length(self) -> int:
        """Get chunk content length."""
        return len(self.content)
    
    def calculate_hash(self) -> str:
        """Calculate content hash for deduplication."""
        if not self.content_hash:
            # Normalize content for hashing (lowercase, strip whitespace)
            normalized = self.content.lower().strip()
            normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
            self.content_hash = hashlib.sha256(normalized.encode()).hexdigest()
        return self.content_hash


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
    
    def calculate_chunk_quality_score(self, chunk: Chunk) -> float:
        """
        Calculate quality score for a chunk based on multiple factors.
        
        Returns a score between 0.0 and 1.0 where higher is better.
        """
        score = 0.0
        weights = {
            "length": 0.3,
            "alphanumeric_ratio": 0.2,
            "word_count": 0.2,
            "punctuation": 0.1,
            "capitalization": 0.1,
            "whitespace": 0.1
        }
        
        content = chunk.content
        
        # 1. Length score (prefer chunks near target size)
        length = len(content)
        if length >= self.chunk_size * 0.5 and length <= self.chunk_size * 1.2:
            length_score = 1.0
        elif length < 100:
            length_score = 0.2
        else:
            # Gradual decrease for too short or too long chunks
            if length < self.chunk_size * 0.5:
                length_score = 0.5 + (length / (self.chunk_size * 0.5)) * 0.5
            else:
                length_score = max(0.3, 1.0 - (length - self.chunk_size) / (self.chunk_size * 2))
        score += length_score * weights["length"]
        
        # 2. Alphanumeric ratio (prefer content with actual text)
        alphanumeric = sum(c.isalnum() for c in content)
        alphanumeric_ratio = alphanumeric / length if length > 0 else 0
        alphanumeric_score = min(1.0, alphanumeric_ratio * 1.5)  # Scale up slightly
        score += alphanumeric_score * weights["alphanumeric_ratio"]
        
        # 3. Word count (prefer chunks with reasonable word count)
        words = content.split()
        word_count = len(words)
        if word_count >= 20:
            word_score = 1.0
        elif word_count < 5:
            word_score = 0.2
        else:
            word_score = 0.2 + (word_count / 20) * 0.8
        score += word_score * weights["word_count"]
        
        # 4. Punctuation (some punctuation is good, too much is bad)
        punctuation_count = sum(c in '.!?,;:' for c in content)
        punctuation_ratio = punctuation_count / length if length > 0 else 0
        if 0.01 <= punctuation_ratio <= 0.05:
            punctuation_score = 1.0
        elif punctuation_ratio > 0.1:
            punctuation_score = 0.3
        else:
            punctuation_score = 0.7
        score += punctuation_score * weights["punctuation"]
        
        # 5. Capitalization (proper capitalization indicates quality)
        capital_count = sum(c.isupper() for c in content)
        capital_ratio = capital_count / length if length > 0 else 0
        if 0.02 <= capital_ratio <= 0.1:
            capital_score = 1.0
        elif capital_ratio > 0.3:
            capital_score = 0.3  # Too many capitals (might be code or headers)
        else:
            capital_score = 0.7
        score += capital_score * weights["capitalization"]
        
        # 6. Whitespace ratio (reasonable whitespace is good)
        whitespace_count = sum(c.isspace() for c in content)
        whitespace_ratio = whitespace_count / length if length > 0 else 0
        if 0.1 <= whitespace_ratio <= 0.2:
            whitespace_score = 1.0
        elif whitespace_ratio > 0.4:
            whitespace_score = 0.3  # Too much whitespace
        else:
            whitespace_score = 0.7
        score += whitespace_score * weights["whitespace"]
        
        # Ensure score is between 0 and 1
        chunk.quality_score = max(0.0, min(1.0, score))
        
        return chunk.quality_score
    
    def deduplicate_chunks(
        self,
        chunks: List[Chunk],
        similarity_threshold: float = 0.95
    ) -> Tuple[List[Chunk], List[Chunk]]:
        """
        Remove duplicate chunks based on content hash and similarity.
        
        Args:
            chunks: List of chunks to deduplicate
            similarity_threshold: Threshold for fuzzy matching (not implemented yet)
            
        Returns:
            Tuple of (unique_chunks, duplicate_chunks)
        """
        seen_hashes: Set[str] = set()
        unique_chunks: List[Chunk] = []
        duplicate_chunks: List[Chunk] = []
        
        for chunk in chunks:
            # Calculate hash for deduplication
            chunk_hash = chunk.calculate_hash()
            
            if chunk_hash in seen_hashes:
                duplicate_chunks.append(chunk)
                logger.debug(
                    "duplicate_chunk_found",
                    hash=chunk_hash[:8],
                    content_preview=chunk.content[:50]
                )
            else:
                seen_hashes.add(chunk_hash)
                unique_chunks.append(chunk)
        
        if duplicate_chunks:
            logger.info(
                "chunks_deduplicated",
                total_chunks=len(chunks),
                unique_chunks=len(unique_chunks),
                duplicates_removed=len(duplicate_chunks)
            )
        
        return unique_chunks, duplicate_chunks
    
    def process_chunks_with_quality(
        self,
        chunks: List[Chunk],
        min_quality_score: float = 0.3,
        deduplicate: bool = True
    ) -> List[Chunk]:
        """
        Process chunks with quality scoring and optional deduplication.
        
        Args:
            chunks: Raw chunks to process
            min_quality_score: Minimum quality score to keep chunk
            deduplicate: Whether to remove duplicates
            
        Returns:
            Processed chunks that meet quality criteria
        """
        # Calculate quality scores
        for chunk in chunks:
            self.calculate_chunk_quality_score(chunk)
        
        # Filter by quality score
        quality_chunks = [
            chunk for chunk in chunks
            if chunk.quality_score >= min_quality_score
        ]
        
        if len(quality_chunks) < len(chunks):
            logger.info(
                "low_quality_chunks_filtered",
                original_count=len(chunks),
                kept_count=len(quality_chunks),
                filtered_count=len(chunks) - len(quality_chunks),
                min_score=min_quality_score
            )
        
        # Deduplicate if requested
        if deduplicate:
            quality_chunks, duplicates = self.deduplicate_chunks(quality_chunks)
        
        # Sort by quality score (descending) for better retrieval
        quality_chunks.sort(key=lambda c: c.quality_score, reverse=True)
        
        return quality_chunks
    
    def chunk_document_with_quality(
        self,
        text: str,
        document_metadata: Optional[Dict[str, Any]] = None,
        chunking_strategy: str = "recursive",
        min_quality_score: float = 0.3,
        deduplicate: bool = True
    ) -> List[Chunk]:
        """
        Chunk document with quality scoring and deduplication.
        
        Args:
            text: Document text to chunk
            document_metadata: Metadata to attach to each chunk
            chunking_strategy: Strategy to use ("recursive" or "sentence")
            min_quality_score: Minimum quality score threshold
            deduplicate: Whether to remove duplicate chunks
            
        Returns:
            List of high-quality, unique chunks
        """
        # Get initial chunks
        chunks = self.chunk_document(text, document_metadata, chunking_strategy)
        
        # Process with quality scoring and deduplication
        processed_chunks = self.process_chunks_with_quality(
            chunks,
            min_quality_score=min_quality_score,
            deduplicate=deduplicate
        )
        
        # Re-index chunks after filtering
        for i, chunk in enumerate(processed_chunks):
            chunk.metadata = chunk.metadata or {}
            chunk.metadata["final_chunk_index"] = i
            chunk.metadata["final_total_chunks"] = len(processed_chunks)
            chunk.metadata["quality_score"] = chunk.quality_score
            chunk.metadata["content_hash"] = chunk.content_hash[:8]  # Short hash for logging
        
        return processed_chunks


# Global chunking service instance
chunking_service = DocumentChunkingService()