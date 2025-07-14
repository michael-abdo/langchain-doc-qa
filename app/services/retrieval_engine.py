"""
Retrieval Engine for RAG Pipeline

Provides high-performance document retrieval with quality controls, including
vector search, hybrid search, relevance filtering, and performance monitoring.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings

from app.core.common import (
    BaseService,
    get_service_logger,
    config,
    with_error_handling,
    normalize_text
)
from app.core.config import ConfigAccessor
from app.core.exceptions import (
    VectorStoreError,
    ProcessingError,
    ValidationError
)
from app.services.query_processor import ProcessedQuery
from app.services.vector_store import VectorStoreManager


@dataclass
class SearchResult:
    """Individual search result with metadata."""
    document: Document
    score: float
    relevance_score: float
    chunk_id: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class SearchResults:
    """Collection of search results with metadata."""
    query: ProcessedQuery
    results: List[SearchResult]
    total_found: int
    search_time: float
    search_method: str
    filters_applied: List[str]
    performance_metrics: Dict[str, Any]


@dataclass
class ValidatedResults:
    """Validated search results ready for answer generation."""
    results: List[SearchResult]
    validation_metrics: Dict[str, float]
    quality_score: float
    warnings: List[str]


@dataclass
class FilteredResults:
    """Results after relevance filtering."""
    results: List[SearchResult]
    removed_count: int
    average_relevance: float
    filter_threshold: float


@dataclass
class PerformanceMetrics:
    """Search performance metrics."""
    search_latency: float
    embedding_time: float
    vector_search_time: float
    reranking_time: float
    total_documents_scanned: int
    cache_hit: bool
    memory_usage_mb: float


class ResultValidator:
    """Validates search results for quality."""
    
    MIN_CONTENT_LENGTH = 50
    MAX_CONTENT_LENGTH = 2000
    MIN_RELEVANCE_SCORE = 0.3
    
    def __init__(self, logger):
        self.logger = logger
    
    def validate_results(self, results: List[SearchResult]) -> ValidatedResults:
        """Validate search results for quality and relevance."""
        validated_results = []
        validation_metrics = {
            'total_results': len(results),
            'passed_validation': 0,
            'failed_length': 0,
            'failed_relevance': 0,
            'duplicate_content': 0
        }
        warnings = []
        
        seen_content = set()
        
        for result in results:
            content = result.document.page_content
            
            # Length validation
            if len(content) < self.MIN_CONTENT_LENGTH:
                validation_metrics['failed_length'] += 1
                warnings.append(f"Result from {result.source} too short ({len(content)} chars)")
                continue
            
            if len(content) > self.MAX_CONTENT_LENGTH:
                # Truncate but include
                result.document.page_content = content[:self.MAX_CONTENT_LENGTH] + "..."
                warnings.append(f"Result from {result.source} truncated")
            
            # Relevance validation
            if result.relevance_score < self.MIN_RELEVANCE_SCORE:
                validation_metrics['failed_relevance'] += 1
                continue
            
            # Duplicate detection
            content_hash = hash(content.lower().strip())
            if content_hash in seen_content:
                validation_metrics['duplicate_content'] += 1
                continue
            seen_content.add(content_hash)
            
            validated_results.append(result)
            validation_metrics['passed_validation'] += 1
        
        # Calculate quality score
        if validation_metrics['total_results'] > 0:
            quality_score = validation_metrics['passed_validation'] / validation_metrics['total_results']
            avg_relevance = np.mean([r.relevance_score for r in validated_results]) if validated_results else 0
            quality_score = (quality_score + avg_relevance) / 2
        else:
            quality_score = 0
        
        return ValidatedResults(
            results=validated_results,
            validation_metrics=validation_metrics,
            quality_score=quality_score,
            warnings=warnings
        )


class RelevanceFilter:
    """Applies relevance filtering to search results."""
    
    def __init__(self, logger, threshold: float = 0.6):
        self.logger = logger
        self.threshold = threshold
    
    def apply_filter(self, results: List[SearchResult], custom_threshold: Optional[float] = None) -> FilteredResults:
        """Filter results by relevance threshold."""
        threshold = custom_threshold or self.threshold
        
        filtered_results = []
        removed_count = 0
        
        for result in results:
            if result.relevance_score >= threshold:
                filtered_results.append(result)
            else:
                removed_count += 1
                self.logger.debug(
                    f"Filtered out result with relevance {result.relevance_score:.3f} < {threshold}"
                )
        
        avg_relevance = np.mean([r.relevance_score for r in filtered_results]) if filtered_results else 0
        
        return FilteredResults(
            results=filtered_results,
            removed_count=removed_count,
            average_relevance=avg_relevance,
            filter_threshold=threshold
        )


class PerformanceMonitor:
    """Monitors and tracks search performance."""
    
    def __init__(self, logger):
        self.logger = logger
        self.metrics_history = []
        self.max_history = 1000
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics
        })
        
        # Keep only recent history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
        
        # Log if performance is degrading
        if metrics.search_latency > 10:
            self.logger.warning(
                f"High search latency detected: {metrics.search_latency:.2f}s",
                extra={'metrics': metrics.__dict__}
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = [m['metrics'] for m in self.metrics_history[-100:]]
        
        return {
            'avg_latency': np.mean([m.search_latency for m in recent_metrics]),
            'p95_latency': np.percentile([m.search_latency for m in recent_metrics], 95),
            'avg_documents_scanned': np.mean([m.total_documents_scanned for m in recent_metrics]),
            'cache_hit_rate': sum(1 for m in recent_metrics if m.cache_hit) / len(recent_metrics),
            'total_searches': len(self.metrics_history)
        }


class RetrievalEngine(BaseService):
    """Main retrieval engine with performance optimization."""
    
    SEARCH_TIMEOUT = 30  # seconds
    MAX_RESULTS = 10
    
    def __init__(self):
        super().__init__("RetrievalEngine")
        self.config_accessor = config  # config is already a ConfigAccessor instance
        self.validator = ResultValidator(self.logger)
        self.relevance_filter = RelevanceFilter(self.logger)
        self.performance_monitor = PerformanceMonitor(self.logger)
        self.vector_store_manager = VectorStoreManager()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._cache = {}  # Simple query cache
        self._cache_ttl = 3600  # 1 hour
        self._initialize()
    
    @with_error_handling("initialization")
    def _initialize(self):
        """Initialize the retrieval engine."""
        self.logger.info("Initializing Retrieval Engine")
        self._initialized = True
    
    @with_error_handling("search", raise_as=VectorStoreError)
    def search_documents(self, query: ProcessedQuery, filters: Optional[Dict[str, Any]] = None) -> SearchResults:
        """
        Main search method with timeout and performance tracking.
        
        Args:
            query: Processed query from QueryProcessor
            filters: Optional filters to apply
            
        Returns:
            SearchResults with documents and metrics
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(query, filters)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            self.logger.info(f"Cache hit for query: {query.normalized_query[:50]}...")
            return cached_result
        
        try:
            # Run search with timeout
            future = self._executor.submit(self._perform_search, query, filters)
            results = future.result(timeout=self.SEARCH_TIMEOUT)
            
            # Cache successful results
            self._cache_result(cache_key, results)
            
            return results
            
        except FutureTimeoutError:
            elapsed = time.time() - start_time
            self.logger.error(f"Search timeout after {elapsed:.2f}s for query: {query.normalized_query}")
            raise VectorStoreError(f"Search timeout exceeded ({self.SEARCH_TIMEOUT}s)")
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise
    
    def _perform_search(self, query: ProcessedQuery, filters: Optional[Dict[str, Any]]) -> SearchResults:
        """Perform the actual search operation."""
        metrics = {
            'start_time': time.time(),
            'embedding_time': 0,
            'vector_search_time': 0,
            'reranking_time': 0,
            'total_documents_scanned': 0,
            'cache_hit': False
        }
        
        # Get embeddings
        embed_start = time.time()
        # Use the normalized query for better retrieval
        query_text = query.normalized_query
        metrics['embedding_time'] = time.time() - embed_start
        
        # Perform vector search
        search_start = time.time()
        raw_results = self._vector_search(query_text, filters)
        metrics['vector_search_time'] = time.time() - search_start
        metrics['total_documents_scanned'] = len(raw_results)
        
        # Convert to SearchResult objects
        search_results = []
        for doc, score in raw_results:
            # Calculate relevance score (normalize similarity score)
            relevance_score = self._calculate_relevance_score(score, doc, query)
            
            search_result = SearchResult(
                document=doc,
                score=score,
                relevance_score=relevance_score,
                chunk_id=doc.metadata.get('chunk_id', 'unknown'),
                source=doc.metadata.get('source', 'unknown'),
                metadata=doc.metadata
            )
            search_results.append(search_result)
        
        # Sort by relevance score
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit results
        search_results = search_results[:self.MAX_RESULTS]
        
        # Apply validation
        validated = self.validator.validate_results(search_results)
        
        # Apply relevance filtering
        filtered = self.relevance_filter.apply_filter(validated.results)
        
        # Calculate total search time
        total_time = time.time() - metrics['start_time']
        
        # Record performance metrics
        perf_metrics = PerformanceMetrics(
            search_latency=total_time,
            embedding_time=metrics['embedding_time'],
            vector_search_time=metrics['vector_search_time'],
            reranking_time=metrics['reranking_time'],
            total_documents_scanned=metrics['total_documents_scanned'],
            cache_hit=False,
            memory_usage_mb=self._get_memory_usage()
        )
        self.performance_monitor.record_metrics(perf_metrics)
        
        return SearchResults(
            query=query,
            results=filtered.results,
            total_found=len(raw_results),
            search_time=total_time,
            search_method="vector_search",
            filters_applied=list(filters.keys()) if filters else [],
            performance_metrics={
                'search_latency': total_time,
                'documents_scanned': metrics['total_documents_scanned'],
                'results_returned': len(filtered.results),
                'results_filtered': filtered.removed_count,
                'average_relevance': filtered.average_relevance,
                'quality_score': validated.quality_score,
                'validation_warnings': len(validated.warnings)
            }
        )
    
    def _vector_search(self, query: str, filters: Optional[Dict[str, Any]]) -> List[Tuple[Document, float]]:
        """Perform vector similarity search."""
        try:
            # For now, we'll use the VectorStoreManager's search method directly
            # This is a temporary solution until we implement proper async/sync coordination
            
            # The VectorStoreManager has a search method that we can use
            # We'll need to generate embeddings first
            self.logger.warning("Vector search not fully implemented - returning empty results")
            return []
            
            # TODO: Implement proper vector search integration:
            # 1. Generate embedding for query
            # 2. Call VectorStoreManager.search() method
            # 3. Convert results to expected format
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {str(e)}")
            raise VectorStoreError(f"Vector search failed: {str(e)}")
    
    def _calculate_relevance_score(self, similarity_score: float, document: Document, query: ProcessedQuery) -> float:
        """
        Calculate comprehensive relevance score.
        
        Combines:
        - Vector similarity score
        - Keyword overlap
        - Query intent matching
        - Metadata factors
        """
        # Start with similarity score (usually cosine similarity 0-1)
        relevance = similarity_score
        
        # Keyword overlap bonus
        query_terms = set(query.normalized_query.lower().split())
        doc_terms = set(document.page_content.lower().split())
        overlap = len(query_terms.intersection(doc_terms)) / max(len(query_terms), 1)
        relevance += overlap * 0.2  # 20% weight for keyword overlap
        
        # Intent matching bonus
        if query.detected_intent == "question" and "?" in document.page_content:
            relevance += 0.1
        elif query.detected_intent == "explanation" and any(word in document.page_content.lower() for word in ["because", "therefore", "thus"]):
            relevance += 0.1
        
        # Recency bonus (if timestamp available)
        if 'timestamp' in document.metadata:
            try:
                doc_age_days = (datetime.now() - datetime.fromisoformat(document.metadata['timestamp'])).days
                if doc_age_days < 30:
                    relevance += 0.1
                elif doc_age_days < 90:
                    relevance += 0.05
            except:
                pass
        
        # Normalize to 0-1 range
        return min(relevance, 1.0)
    
    def _get_cache_key(self, query: ProcessedQuery, filters: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for query."""
        filter_str = str(sorted(filters.items())) if filters else ""
        return f"{query.normalized_query}:{filter_str}"
    
    def _check_cache(self, cache_key: str) -> Optional[SearchResults]:
        """Check if results are in cache."""
        if cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if time.time() - cached_item['timestamp'] < self._cache_ttl:
                # Update performance metrics for cache hit
                perf_metrics = PerformanceMetrics(
                    search_latency=0.001,  # Cache hit is fast
                    embedding_time=0,
                    vector_search_time=0,
                    reranking_time=0,
                    total_documents_scanned=0,
                    cache_hit=True,
                    memory_usage_mb=self._get_memory_usage()
                )
                self.performance_monitor.record_metrics(perf_metrics)
                return cached_item['results']
            else:
                # Remove expired cache entry
                del self._cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, results: SearchResults):
        """Cache search results."""
        self._cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }
        
        # Limit cache size
        if len(self._cache) > 100:
            # Remove oldest entries
            sorted_keys = sorted(self._cache.items(), key=lambda x: x[1]['timestamp'])
            for key, _ in sorted_keys[:20]:
                del self._cache[key]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    @with_error_handling("hybrid_search", raise_as=VectorStoreError)
    def hybrid_search(self, query: ProcessedQuery, filters: Optional[Dict[str, Any]] = None) -> SearchResults:
        """
        Perform hybrid search combining vector and keyword search.
        
        This is a placeholder for future enhancement.
        Currently delegates to vector search.
        """
        # For now, use vector search
        # In future, combine with BM25 or other keyword search
        results = self.search_documents(query, filters)
        results.search_method = "hybrid_search"
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get retrieval engine performance statistics."""
        stats = self.performance_monitor.get_performance_stats()
        stats.update({
            'cache_size': len(self._cache),
            'max_results': self.MAX_RESULTS,
            'search_timeout': self.SEARCH_TIMEOUT,
            'relevance_threshold': self.relevance_filter.threshold
        })
        return stats
    
    def clear_cache(self):
        """Clear the search cache."""
        self._cache.clear()
        self.logger.info("Search cache cleared")
    
    def shutdown(self):
        """Shutdown the retrieval engine."""
        self._executor.shutdown(wait=True)
        self.logger.info("Retrieval engine shutdown complete")