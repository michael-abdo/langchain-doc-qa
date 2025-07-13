"""
Vector store service.
Provides abstraction layer for different vector storage backends (FAISS, Chroma, pgvector).
"""
import os
import pickle
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import asyncio
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import VectorStoreError, ExternalServiceError

logger = get_logger(__name__)


class VectorStoreInterface(ABC):
    """Abstract interface for vector store operations."""
    
    @abstractmethod
    async def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to vector store."""
        pass
    
    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from vector store."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check vector store health."""
        pass


class FAISSVectorStore(VectorStoreInterface):
    """FAISS-based vector store implementation."""
    
    def __init__(self, storage_path: str = "/tmp/faiss_index"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_path / "index.faiss"
        self.metadata_file = self.storage_path / "metadata.pkl"
        self._index = None
        self._metadata = {}
        self._dimension = None
        
    async def _ensure_index_loaded(self):
        """Ensure FAISS index is loaded."""
        if self._index is None:
            await self._load_index()
    
    async def _load_index(self):
        """Load existing FAISS index or create new one."""
        try:
            import faiss
            
            if self.index_file.exists():
                # Load existing index
                self._index = faiss.read_index(str(self.index_file))
                self._dimension = self._index.d
                
                # Load metadata
                if self.metadata_file.exists():
                    with open(self.metadata_file, 'rb') as f:
                        self._metadata = pickle.load(f)
                
                logger.info(
                    "faiss_index_loaded",
                    dimension=self._dimension,
                    num_vectors=self._index.ntotal
                )
            else:
                # Will create index when first documents are added
                logger.info("faiss_index_will_be_created_on_first_add")
                
        except Exception as e:
            raise VectorStoreError(f"Failed to load FAISS index: {str(e)}", "load")
    
    async def _save_index(self):
        """Save FAISS index to disk."""
        try:
            if self._index is not None:
                import faiss
                faiss.write_index(self._index, str(self.index_file))
                
                # Save metadata
                with open(self.metadata_file, 'wb') as f:
                    pickle.dump(self._metadata, f)
                
                logger.info("faiss_index_saved", num_vectors=self._index.ntotal)
                
        except Exception as e:
            raise VectorStoreError(f"Failed to save FAISS index: {str(e)}", "save")
    
    async def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to FAISS index."""
        try:
            import faiss
            import uuid
            
            if not embeddings:
                return []
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            dimension = embeddings_array.shape[1]
            
            # Create index if it doesn't exist
            if self._index is None:
                self._index = faiss.IndexFlatL2(dimension)
                self._dimension = dimension
                logger.info("faiss_index_created", dimension=dimension)
            
            # Validate dimension consistency
            if self._dimension != dimension:
                raise VectorStoreError(
                    f"Embedding dimension mismatch. Expected {self._dimension}, got {dimension}",
                    "add_documents"
                )
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
            # Add to index
            start_idx = self._index.ntotal
            self._index.add(embeddings_array)
            
            # Store metadata
            for i, (text, doc_id) in enumerate(zip(texts, ids)):
                idx = start_idx + i
                self._metadata[idx] = {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadatas[i] if metadatas else {}
                }
            
            # Save index
            await self._save_index()
            
            logger.info(
                "documents_added_to_faiss",
                count=len(texts),
                total_vectors=self._index.ntotal
            )
            
            return ids
            
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents: {str(e)}", "add_documents")
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents in FAISS index."""
        try:
            await self._ensure_index_loaded()
            
            if self._index is None or self._index.ntotal == 0:
                return []
            
            # Convert query to numpy array
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # Search
            k = min(k, self._index.ntotal)
            distances, indices = self._index.search(query_array, k)
            
            # Build results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx in self._metadata:  # -1 indicates no result
                    metadata = self._metadata[idx]
                    
                    # Apply filters if provided
                    if filter_dict:
                        doc_metadata = metadata.get("metadata", {})
                        if not all(
                            doc_metadata.get(key) == value
                            for key, value in filter_dict.items()
                        ):
                            continue
                    
                    results.append({
                        "id": metadata["id"],
                        "text": metadata["text"],
                        "score": float(distance),
                        "metadata": metadata.get("metadata", {})
                    })
            
            logger.info(
                "faiss_similarity_search_completed",
                query_dimension=len(query_embedding),
                k=k,
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            raise VectorStoreError(f"Failed to search documents: {str(e)}", "similarity_search")
    
    async def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents from FAISS index (not directly supported, requires rebuild)."""
        # FAISS doesn't support direct deletion, would need to rebuild index
        # For now, just mark as deleted in metadata
        try:
            deleted_count = 0
            for idx, metadata in self._metadata.items():
                if metadata["id"] in ids:
                    metadata["deleted"] = True
                    deleted_count += 1
            
            await self._save_index()
            
            logger.info("faiss_documents_marked_deleted", count=deleted_count)
            return deleted_count > 0
            
        except Exception as e:
            raise VectorStoreError(f"Failed to delete documents: {str(e)}", "delete_documents")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check FAISS vector store health."""
        try:
            await self._ensure_index_loaded()
            
            return {
                "status": "healthy",
                "type": "faiss",
                "message": "FAISS vector store is operational",
                "details": {
                    "index_exists": self._index is not None,
                    "dimension": self._dimension,
                    "num_vectors": self._index.ntotal if self._index else 0,
                    "storage_path": str(self.storage_path),
                    "index_file_exists": self.index_file.exists()
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "type": "faiss",
                "message": f"FAISS vector store health check failed: {str(e)}",
                "details": {"error": str(e)}
            }


class EmbeddingService:
    """Service for generating embeddings."""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        self._client = None
    
    async def _get_client(self):
        """Get OpenAI client for embeddings."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            except Exception as e:
                raise ExternalServiceError(
                    service_name="OpenAI",
                    message=f"Failed to initialize OpenAI client: {str(e)}"
                )
        return self._client
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts."""
        try:
            client = await self._get_client()
            
            # OpenAI has limits on batch size and text length
            # Split into smaller batches if needed
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            logger.info(
                "embeddings_generated",
                model=self.model_name,
                count=len(texts),
                dimension=len(all_embeddings[0]) if all_embeddings else 0
            )
            
            return all_embeddings
            
        except Exception as e:
            raise ExternalServiceError(
                service_name="OpenAI",
                message=f"Failed to generate embeddings: {str(e)}"
            )
    
    async def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        embeddings = await self.generate_embeddings([text])
        return embeddings[0]


class VectorStoreManager:
    """Manager for vector store operations."""
    
    def __init__(self):
        self.store_type = settings.VECTOR_STORE_TYPE
        self._store = None
        self.embedding_service = EmbeddingService()
    
    async def _get_store(self) -> VectorStoreInterface:
        """Get vector store instance."""
        if self._store is None:
            if self.store_type == "faiss":
                self._store = FAISSVectorStore()
            else:
                raise VectorStoreError(
                    f"Unsupported vector store type: {self.store_type}",
                    "initialization"
                )
        return self._store
    
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to vector store with embeddings."""
        store = await self._get_store()
        embeddings = await self.embedding_service.generate_embeddings(texts)
        return await store.add_documents(texts, embeddings, metadatas, ids)
    
    async def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar texts."""
        store = await self._get_store()
        query_embedding = await self.embedding_service.generate_single_embedding(query)
        return await store.similarity_search(query_embedding, k, filter_dict)
    
    async def delete_texts(self, ids: List[str]) -> bool:
        """Delete texts from vector store."""
        store = await self._get_store()
        return await store.delete_documents(ids)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check vector store health."""
        try:
            store = await self._get_store()
            return await store.health_check()
        except Exception as e:
            return {
                "status": "unhealthy",
                "type": self.store_type,
                "message": f"Vector store manager health check failed: {str(e)}",
                "details": {"error": str(e)}
            }


# Global vector store manager instance
vector_store_manager = VectorStoreManager()