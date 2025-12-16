"""
Business logic for Qdrant operations
Orchestrates CRUD operations with validation, logging, and complex workflows
UPDATED: Added fallback mechanisms for missing chunks
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging
import os
import uuid
import random

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from qdrant_client.http import models

# Import CRUD operations
from app.crud import qdrant

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
            
            # Index difficulty for filtering by difficulty level
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="difficulty",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            
            logger.info("✓ Created payload indexes for material_id, chunk_id, and difficulty")
            
        except Exception as e:
            # Indexes might already exist, log but don't fail
            logger.debug(f"Payload indexes may already exist: {e}")
    
    # ==================== DOCUMENT CHUNKING OPERATIONS ====================
    
    async def upsert_embeddings(
        self,
        material_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        difficulties: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Insert or update chunk embeddings for a material (document)
        Business logic: validates, deletes old, creates new, tracks metadata
        
        Args:
            material_id: Unique identifier for the material/document
            chunks: List of text chunks
            embeddings: List of embedding vectors (must match chunks length)
            difficulties: Optional list of difficulty ratings ("easy", "medium", "hard")
        
        Returns:
            Dictionary with operation statistics
        
        Raises:
            ValueError: If chunks and embeddings length mismatch
        """
        # Business rule: Validate input
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks and embeddings length mismatch: "
                f"{len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        
        if difficulties and len(difficulties) != len(chunks):
            raise ValueError(
                f"Difficulties length mismatch: "
                f"{len(difficulties)} difficulties vs {len(chunks)} chunks"
            )
        
        if not chunks:
            logger.warning(f"No chunks to upsert for material {material_id}")
            return {
                "material_id": material_id,
                "chunks_upserted": 0,
                "status": "no_chunks"
            }
        
        try:
            # Business rule: Delete existing embeddings first
            await self.delete_material_embeddings(material_id)
            
            # Business rule: Prepare points with metadata
            points = []
            timestamp = datetime.now(timezone.utc).isoformat()
            
            for chunk_id, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate unique point ID
                point_id = str(uuid.uuid4())
                
                # Business rule: Add standard metadata
                payload = {
                    "material_id": material_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "text_length": len(chunk_text),
                    "indexed_at": timestamp,
                    "chunk_total": len(chunks)
                }
                
                # Add difficulty rating if provided
                if difficulties:
                    payload["difficulty"] = difficulties[chunk_id]
                
                # Create point
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                
                points.append(point)
            
            # Use CRUD layer for batch insert
            success = qdrant.batch_create_points(
                client=self.client,
                collection_name=self.collection_name,
                points_data=points
            )
            
            if not success:
                raise Exception("Failed to upsert points to Qdrant")
            
            logger.info(
                f"✓ Upserted {len(points)} embeddings for material '{material_id}'"
            )
            
            # Calculate difficulty distribution if provided
            difficulty_stats = {}
            if difficulties:
                for diff in ["easy", "medium", "hard"]:
                    count = difficulties.count(diff)
                    difficulty_stats[diff] = count
            
            return {
                "material_id": material_id,
                "chunks_upserted": len(points),
                "status": "success",
                "indexed_at": timestamp,
                "difficulty_stats": difficulty_stats if difficulty_stats else None
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
        Business logic: uses filter-based deletion for efficiency
        
        Args:
            material_id: Material identifier
        
        Returns:
            bool: True if deletion successful
        """
        try:
            # Business rule: Build filter for material_id
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="material_id",
                        match=models.MatchValue(value=material_id)
                    )
                ]
            )
            
            # Use CRUD layer for deletion
            success = qdrant.delete_points_by_filter(
                client=self.client,
                collection_name=self.collection_name,
                filter_condition=filter_condition
            )
            
            if success:
                logger.info(f"✓ Deleted embeddings for material '{material_id}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting embeddings for material {material_id}: {e}")
            return False
    
    async def search_similar_chunks(
        self,
        query_embedding: List[float],
        limit: int = 10,
        material_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using query embedding
        Business logic: applies filters, enriches results
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            material_id: Optional filter to search within specific material
            difficulty: Optional filter by difficulty level ("easy", "medium", "hard")
            score_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Business rule: Build filter conditions
            filter_conditions = []
            
            if material_id:
                filter_conditions.append(
                    models.FieldCondition(
                        key="material_id",
                        match=models.MatchValue(value=material_id)
                    )
                )
            
            if difficulty:
                filter_conditions.append(
                    models.FieldCondition(
                        key="difficulty",
                        match=models.MatchValue(value=difficulty)
                    )
                )
            
            query_filter = None
            if filter_conditions:
                query_filter = models.Filter(must=filter_conditions)
            
            # Business rule: Cap limit at reasonable max
            limit = min(limit, 100)
            
            # Use CRUD layer for search
            search_results = qdrant.search_similar(
                client=self.client,
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True
            )
            
            # Business rule: Format and enrich results
            results = []
            for result in search_results:
                results.append({
                    "id": result["id"],
                    "score": result["score"],
                    "similarity_percentage": f"{result['score'] * 100:.1f}%",
                    "material_id": result["payload"].get("material_id"),
                    "chunk_id": result["payload"].get("chunk_id"),
                    "text": result["payload"].get("text"),
                    "difficulty": result["payload"].get("difficulty", "unknown"),
                    "metadata": result["payload"]
                })
            
            logger.info(
                f"✓ Found {len(results)} similar chunks "
                f"(material_filter: {material_id or 'none'}, "
                f"difficulty_filter: {difficulty or 'none'})"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    async def get_material_chunks(
        self,
        material_id: str,
        with_vectors: bool = False,
        difficulty: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks for a specific material
        Business logic: filters by material, sorts by chunk_id
        
        Args:
            material_id: Material identifier
            with_vectors: Include embedding vectors in response
            difficulty: Optional filter by difficulty level
        
        Returns:
            List of chunks with metadata, ordered by chunk_id
        """
        try:
            # Business rule: Build filter for material
            filter_conditions = [
                models.FieldCondition(
                    key="material_id",
                    match=models.MatchValue(value=material_id)
                )
            ]
            
            if difficulty:
                filter_conditions.append(
                    models.FieldCondition(
                        key="difficulty",
                        match=models.MatchValue(value=difficulty)
                    )
                )
            
            filter_condition = models.Filter(must=filter_conditions)
            
            # Use CRUD layer for scrolling
            records, _ = qdrant.scroll_points(
                client=self.client,
                collection_name=self.collection_name,
                limit=1000,  # Adjust based on expected chunk count
                with_vectors=with_vectors,
                filter_condition=filter_condition
            )
            
            # Business rule: Format and sort results by chunk_id
            chunks = []
            for record in records:
                chunk_data = {
                    "id": record["id"],
                    "chunk_id": record["payload"].get("chunk_id"),
                    "text": record["payload"].get("text"),
                    "difficulty": record["payload"].get("difficulty", "unknown"),
                    "metadata": record["payload"]
                }
                
                if with_vectors and record["vector"]:
                    chunk_data["vector"] = record["vector"]
                
                chunks.append(chunk_data)
            
            # Sort by chunk_id to maintain order
            chunks.sort(key=lambda x: x["chunk_id"])
            
            logger.info(f"✓ Retrieved {len(chunks)} chunks for material '{material_id}'")
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving material chunks: {e}")
            return []
    
    async def get_chunk_by_difficulty(
        self,
        material_id: str,
        difficulty: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a random chunk filtered by material and difficulty
        Business logic: filters by both material_id and difficulty, returns random selection
        
        Args:
            material_id: Material identifier
            difficulty: Difficulty level ("easy", "medium", "hard")
        
        Returns:
            Dictionary with chunk text and payload, or None if no matching chunks found
        """
        try:
            # Business rule: Validate difficulty level
            valid_difficulties = ["easy", "medium", "hard"]
            if difficulty not in valid_difficulties:
                logger.warning(
                    f"Invalid difficulty '{difficulty}'. Must be one of: {valid_difficulties}"
                )
                return None
            
            # Business rule: Build filter for material_id and difficulty
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="material_id",
                        match=models.MatchValue(value=material_id)
                    ),
                    models.FieldCondition(
                        key="difficulty",
                        match=models.MatchValue(value=difficulty)
                    )
                ]
            )
            
            # Use CRUD layer to retrieve matching chunks
            records, _ = qdrant.scroll_points(
                client=self.client,
                collection_name=self.collection_name,
                limit=1000,  # Get all matching chunks
                with_vectors=False,
                filter_condition=filter_condition
            )
            
            if not records:
                logger.info(
                    f"No chunks found for material '{material_id}' "
                    f"with difficulty '{difficulty}'"
                )
                return None
            
            # Business rule: Select random chunk from matching results
            selected_record = random.choice(records)
            
            # Format result
            result = {
                "id": selected_record["id"],
                "text": selected_record["payload"].get("text"),
                "payload": selected_record["payload"]
            }
            
            logger.info(
                f"✓ Selected random chunk (id: {result['id']}) from "
                f"{len(records)} matching chunks for material '{material_id}' "
                f"with difficulty '{difficulty}'"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Error retrieving chunk by difficulty for material {material_id}: {e}"
            )
            return None
    
    async def get_any_chunk_for_material(
        self,
        material_id: str,
        limit: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve any chunk for a material without difficulty filtering.
        Used as last resort fallback when no chunks match specific difficulty.
        
        Args:
            material_id: Material ID to retrieve chunks for
            limit: Number of chunks to retrieve (default: 1)
        
        Returns:
            Dictionary with chunk data or None if no chunks exist
        """
        try:
            # Build filter for only material_id
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="material_id",
                        match=models.MatchValue(value=material_id)
                    )
                ]
            )
            
            # Scroll through points with only material_id filter
            records, _ = qdrant.scroll_points(
                client=self.client,
                collection_name=self.collection_name,
                limit=limit,
                with_vectors=False,
                filter_condition=filter_condition
            )
            
            if not records:
                logger.warning(f"No chunks found for material '{material_id}' at all")
                return None
            
            # Pick a random chunk if multiple available
            selected_record = random.choice(records) if len(records) > 1 else records[0]
            
            result = {
                "id": selected_record["id"],
                "text": selected_record["payload"].get("text"),
                "payload": selected_record["payload"]
            }
            
            logger.info(
                f"✓ Retrieved any chunk (id: {result['id']}) from "
                f"{len(records)} total chunks for material '{material_id}'"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error retrieving any chunk for material {material_id}: {e}")
            return None
    
    # ==================== UTILITY METHODS ====================
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics with business context
        Business logic: adds health status interpretation
        """
        try:
            # Use CRUD layer for collection info
            stats = qdrant.get_collection_info(
                client=self.client,
                collection_name=self.collection_name
            )
            
            if not stats:
                return {"error": "Failed to get collection info"}
            
            # Business rule: Add health interpretation
            stats["health"] = "healthy" if stats["status"] == "green" else "degraded"
            
            return stats
            
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
            stats = qdrant.get_collection_info(
                client=self.client,
                collection_name=self.collection_name
            )
            return stats is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def ping(self) -> bool:
        """Health check alias"""
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