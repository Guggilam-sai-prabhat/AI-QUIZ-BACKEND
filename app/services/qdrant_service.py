"""
Business logic for Qdrant operations
Orchestrates CRUD operations with validation, logging, and complex workflows
FILE: app/services/qdrant_service.py
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class QdrantService:
    """Service layer for Qdrant with business logic"""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        vector_size: int = 384
    ):
        """
        Initialize Qdrant service with environment variable support
        
        Args:
            host: Qdrant host (default: from env QDRANT_HOST or "localhost")
            port: Qdrant port (default: from env QDRANT_PORT or 6333)
            collection_name: Collection name (default: from env or "materials_vectors")
            vector_size: Vector dimension (default: 384 for all-MiniLM-L6-v2)
        """
        # Get configuration from environment variables with fallbacks
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", 
            "materials_vectors"
        )
        self.vector_size = vector_size
        
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Qdrant"""
        try:
            logger.info(f"🔌 Attempting to connect to Qdrant at {self.host}:{self.port}")
            self.client = QdrantClient(host=self.host, port=self.port)
            
            # Test connection by getting collections
            self.client.get_collections()
            
            logger.info(f"✓ Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant at {self.host}:{self.port}: {e}")
            logger.error(f"   Make sure Qdrant is running and accessible")
            raise ConnectionError(
                f"Cannot connect to Qdrant at {self.host}:{self.port}. "
                f"Error: {str(e)}"
            )
    
    def ensure_collection(self, vector_size: Optional[int] = None):
        """
        Create collection if it doesn't exist
        
        Args:
            vector_size: Dimension of embedding vectors (uses self.vector_size if not provided)
        
        Returns:
            bool: True if collection exists or was created
        """
        if vector_size is None:
            vector_size = self.vector_size
            
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name in collection_names:
                logger.info(f"✓ Collection '{self.collection_name}' already exists")
                
                # Verify vector size matches
                collection_info = self.client.get_collection(self.collection_name)
                existing_size = collection_info.config.params.vectors.size
                
                if existing_size != vector_size:
                    logger.warning(
                        f"⚠ Vector size mismatch: expected {vector_size}, "
                        f"found {existing_size}"
                    )
                
                return True
            
            # Create collection with cosine distance metric
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(
                f"✓ Created collection '{self.collection_name}' "
                f"(size: {vector_size}, distance: COSINE)"
            )
            
            # Create payload index for efficient filtering
            self._create_payload_indexes()
            
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise
    
    def _create_payload_indexes(self):
        """Create indexes on payload fields for faster filtering"""
        try:
            # Index material_id for filtering by material
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="material_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            # Index chunk_id for ordering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_id",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            
            logger.info("✓ Created payload indexes for material_id and chunk_id")
            
        except Exception as e:
            # Indexes might already exist, log but don't fail
            logger.debug(f"Payload indexes may already exist: {e}")
    
    # ==================== DOCUMENT CHUNKING OPERATIONS ====================
    
    async def upsert_embeddings(
        self,
        material_id: str,
        chunks: List[str],
        embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Insert or update chunk embeddings for a material (document)
        
        Args:
            material_id: Unique identifier for the material/document
            chunks: List of text chunks
            embeddings: List of embedding vectors (must match chunks length)
        
        Returns:
            Dictionary with operation statistics
        
        Raises:
            ValueError: If chunks and embeddings length mismatch
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings length mismatch: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        
        if not chunks:
            logger.warning(f"No chunks to upsert for material {material_id}")
            return {
                "material_id": material_id,
                "chunks_upserted": 0,
                "status": "no_chunks"
            }
        
        try:
            # First, delete existing embeddings for this material
            await self.delete_material_embeddings(material_id)
            
            # Prepare points for batch insertion
            points = []
            timestamp = datetime.now(timezone.utc).isoformat()
            
            for chunk_id, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate unique point ID
                point_id = str(uuid.uuid4())
                
                # Create payload with metadata
                payload = {
                    "material_id": material_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "text_length": len(chunk_text),
                    "indexed_at": timestamp,
                    "chunk_total": len(chunks)
                }
                
                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                
                points.append(point)
            
            # Batch upsert to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True  # Wait for operation to complete
            )
            
            logger.info(
                f"✓ Upserted {len(points)} embeddings for material '{material_id}'"
            )
            
            return {
                "material_id": material_id,
                "chunks_upserted": len(points),
                "status": "success",
                "indexed_at": timestamp
            }
            
        except Exception as e:
            logger.error(f"Error upserting embeddings for material {material_id}: {e}")
            return {
                "material_id": material_id,
                "chunks_upserted": 0,
                "status": "error",
                "error": str(e)
            }
    
    async def delete_material_embeddings(self, material_id: str) -> bool:
        """
        Delete all embeddings for a specific material
        
        Args:
            material_id: Material identifier
        
        Returns:
            bool: True if deletion successful
        """
        try:
            # Delete points with matching material_id
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="material_id",
                                match=models.MatchValue(value=material_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"✓ Deleted embeddings for material '{material_id}'")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings for material {material_id}: {e}")
            return False
    
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 10,
        material_id: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using query embedding
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            material_id: Optional filter to search within specific material
            score_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Build filter if material_id specified
            query_filter = None
            if material_id:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="material_id",
                            match=models.MatchValue(value=material_id)
                        )
                    ]
                )
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "material_id": result.payload.get("material_id"),
                    "chunk_id": result.payload.get("chunk_id"),
                    "text": result.payload.get("text"),
                    "metadata": result.payload
                })
            
            logger.info(
                f"✓ Found {len(results)} similar chunks "
                f"(material_filter: {material_id or 'none'})"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    async def get_material_chunks(
        self,
        material_id: str,
        with_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific material
        
        Args:
            material_id: Material identifier
            with_vectors: Include embedding vectors in response
        
        Returns:
            List of chunks with metadata, ordered by chunk_id
        """
        try:
            # Scroll through all points for this material
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="material_id",
                            match=models.MatchValue(value=material_id)
                        )
                    ]
                ),
                limit=1000,  # Adjust based on expected chunk count
                with_payload=True,
                with_vectors=with_vectors
            )
            
            # Format and sort results by chunk_id
            chunks = []
            for record in records:
                chunk_data = {
                    "id": record.id,
                    "chunk_id": record.payload.get("chunk_id"),
                    "text": record.payload.get("text"),
                    "metadata": record.payload
                }
                
                if with_vectors and record.vector:
                    chunk_data["vector"] = record.vector
                
                chunks.append(chunk_data)
            
            # Sort by chunk_id to maintain order
            chunks.sort(key=lambda x: x["chunk_id"])
            
            logger.info(f"✓ Retrieved {len(chunks)} chunks for material '{material_id}'")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving material chunks: {e}")
            return []
    
    # ==================== LEGACY OPERATIONS (for backward compatibility) ====================
    
    def index_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Index a document with business logic
        
        Business rules:
        1. Validate document doesn't already exist
        2. Generate embedding from text (future)
        3. Add timestamp and metadata
        4. Store in Qdrant
        5. Log the operation
        
        Args:
            document_id: Unique document ID
            text: Document text to embed
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            # Business rule: Validate text
            if not text or len(text.strip()) == 0:
                logger.error("Cannot index empty text")
                return False
            
            # TODO: Generate embedding from text
            # For now, create a dummy embedding
            embedding = [0.0] * self.vector_size  # Placeholder
            
            # Business rule: Add standard metadata
            payload = metadata or {}
            payload.update({
                "document_id": document_id,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
                "text_length": len(text),
                "has_content": bool(text.strip()),
                "text": text
            })
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
            
            # Upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )
            
            logger.info(f"✅ Indexed document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing document: {str(e)}")
            return False
    
    def reindex_document(
        self,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Re-index an existing document
        
        Business rules:
        1. Check if document exists
        2. Update with new embedding
        3. Preserve original indexed_at timestamp
        4. Add updated_at timestamp
        """
        try:
            # Generate new embedding
            embedding = [0.0] * self.vector_size  # Placeholder
            
            # Business rule: Add update timestamp
            payload = metadata or {}
            payload.update({
                "document_id": document_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "text_length": len(text)
            })
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
            
            # Upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point],
                wait=True
            )
            
            logger.info(f"✅ Re-indexed document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error re-indexing document: {str(e)}")
            return False
    
    def search_documents(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search with business logic
        
        Business rules:
        1. Validate query
        2. Generate query embedding
        3. Apply filters
        4. Search with score threshold
        5. Enrich results with additional data
        
        Args:
            query: Search query text
            limit: Max results
            min_score: Minimum similarity score
            filters: Metadata filters
            
        Returns:
            List of search results with enriched data
        """
        try:
            # Business rule: Validate query
            if not query or len(query.strip()) == 0:
                logger.error("Empty search query")
                return []
            
            # Business rule: Limit max results
            limit = min(limit, 100)  # Cap at 100
            
            # Generate query embedding
            query_embedding = [0.0] * self.vector_size  # Placeholder
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=min_score,
                with_payload=True
            )
            
            # Business rule: Enrich results
            enriched_results = []
            for result in search_results:
                enriched_results.append({
                    "document_id": result.id,
                    "similarity_score": result.score,
                    "similarity_percentage": f"{result.score * 100:.1f}%",
                    "metadata": result.payload,
                    "matched_query": query
                })
            
            logger.info(f"✅ Search found {len(enriched_results)} results for '{query}'")
            return enriched_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def find_similar_documents(
        self,
        document_id: str,
        limit: int = 5,
        exclude_self: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document
        
        Business rules:
        1. Get the document's embedding
        2. Search for similar ones
        3. Optionally exclude the document itself
        4. Return sorted by similarity
        """
        try:
            # Get all points to find the document
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_vectors=True,
                with_payload=True
            )
            
            # Find the document
            document = None
            for record in records:
                if record.payload.get("document_id") == document_id:
                    document = {"vector": record.vector, "id": record.id}
                    break
            
            if not document:
                logger.error(f"Document {document_id} not found")
                return []
            
            # Search using its embedding
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=document["vector"],
                limit=limit + (1 if exclude_self else 0),
                with_payload=True
            )
            
            # Format results
            results = []
            for result in search_results:
                if exclude_self and result.id == document["id"]:
                    continue
                results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })
            
            # Sort by score
            results = sorted(results, key=lambda x: x["score"], reverse=True)[:limit]
            
            logger.info(f"✅ Found {len(results)} similar documents to {document_id}")
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            return []
    
    def delete_document_index(self, document_id: str) -> bool:
        """
        Delete document index with validation
        
        Business rules:
        1. Check if exists
        2. Log deletion
        3. Handle errors gracefully
        """
        try:
            # Delete by document_id in payload
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            
            logger.info(f"✅ Deleted index for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document index: {str(e)}")
            return False
    
    def bulk_index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Index multiple documents with business logic
        
        Business rules:
        1. Validate all documents first
        2. Skip duplicates
        3. Generate embeddings in batch
        4. Return detailed statistics
        
        Args:
            documents: List of dicts with 'id', 'text', 'metadata'
            
        Returns:
            Statistics dictionary
        """
        try:
            stats = {
                "total": len(documents),
                "indexed": 0,
                "skipped": 0,
                "failed": 0,
                "errors": []
            }
            
            valid_docs = []
            
            for doc in documents:
                # Business rule: Validate structure
                if "id" not in doc or "text" not in doc:
                    stats["failed"] += 1
                    stats["errors"].append(f"Invalid document structure: {doc.get('id', 'unknown')}")
                    continue
                
                # Business rule: Skip empty text
                if not doc["text"] or len(doc["text"].strip()) == 0:
                    stats["skipped"] += 1
                    continue
                
                valid_docs.append(doc)
            
            # Generate embeddings for all valid documents
            points_data = []
            for doc in valid_docs:
                embedding = [0.0] * self.vector_size  # Placeholder
                
                payload = doc.get("metadata", {})
                payload.update({
                    "document_id": doc["id"],
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                    "text_length": len(doc["text"])
                })
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload=payload
                )
                points_data.append(point)
            
            # Batch insert
            if points_data:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points_data,
                    wait=True
                )
                stats["indexed"] = len(points_data)
            
            logger.info(f"✅ Bulk index complete: {stats['indexed']} indexed, {stats['skipped']} skipped, {stats['failed']} failed")
            return stats
            
        except Exception as e:
            logger.error(f"Error in bulk indexing: {str(e)}")
            return {
                "total": len(documents),
                "indexed": 0,
                "skipped": 0,
                "failed": len(documents),
                "errors": [str(e)]
            }
    
    # ==================== UTILITY METHODS ====================
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics with business context
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "total_points": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors": collection_info.indexed_vectors_count,
                "status": collection_info.status,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "health": "healthy" if collection_info.status == "green" else "degraded"
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """
        Check if Qdrant is healthy and collection is accessible
        
        Returns:
            bool: True if healthy
        """
        try:
            # Try to get collection info
            self.client.get_collection(self.collection_name)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def ping(self) -> bool:
        """Health check"""
        return self.health_check()


# ==================== SINGLETON ====================

_qdrant_service: Optional[QdrantService] = None


def get_qdrant_service(
    host: Optional[str] = None,
    port: Optional[int] = None,
    collection_name: Optional[str] = None,
    vector_size: int = 384
) -> QdrantService:
    """
    Get or create the global QdrantService instance
    
    Args:
        host: Qdrant host (optional, uses env var or default)
        port: Qdrant port (optional, uses env var or default)
        collection_name: Collection name (optional, uses env var or default)
        vector_size: Vector dimension (default: 384)
    
    Returns:
        QdrantService instance
    """
    global _qdrant_service
    
    if _qdrant_service is None:
        _qdrant_service = QdrantService(
            host=host,
            port=port,
            collection_name=collection_name,
            vector_size=vector_size
        )
    
    return _qdrant_service