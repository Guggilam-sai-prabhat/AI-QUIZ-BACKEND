"""
Vector Service for Materials Management
Handles Qdrant operations for storing and retrieving material embeddings.
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timezone
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


class VectorService:
    """Service for managing material embeddings in Qdrant"""
    
    def __init__(
        self,
        host: str = "qdrant",
        port: int = 6333,
        collection_name: str = "materials_vectors"
    ):
        """
        Initialize VectorService with Qdrant connection.
        
        Args:
            host: Qdrant host (default: "qdrant" for Docker)
            port: Qdrant port (default: 6333)
            collection_name: Collection name (default: "materials_vectors")
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Qdrant"""
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"✓ Connected to Qdrant at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def ensure_collection(self, vector_size: int = 384):
        """
        Create collection if it doesn't exist.
        
        Args:
            vector_size: Dimension of embedding vectors (default: 384 for all-MiniLM-L6-v2)
        
        Returns:
            bool: True if collection exists or was created
        """
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
    
    async def upsert_embeddings(
        self,
        material_id: str,
        chunks: List[str],
        embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """
        Insert or update chunk embeddings for a material.
        
        Args:
            material_id: Unique identifier for the material
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
        Delete all embeddings for a specific material.
        
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
        Search for similar chunks using query embedding.
        
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
        Retrieve all chunks for a specific material.
        
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
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
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
                "distance_metric": collection_info.config.params.vectors.distance.name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """
        Check if Qdrant is healthy and collection is accessible.
        
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


# ==================== SINGLETON INSTANCE ====================

_vector_service: Optional[VectorService] = None


def get_vector_service(
    host: str = "qdrant",
    port: int = 6333,
    collection_name: str = "materials_vectors"
) -> VectorService:
    """
    Get or create the global VectorService instance.
    
    Args:
        host: Qdrant host
        port: Qdrant port
        collection_name: Collection name
    
    Returns:
        VectorService instance
    """
    global _vector_service
    
    if _vector_service is None:
        _vector_service = VectorService(
            host=host,
            port=port,
            collection_name=collection_name
        )
    
    return _vector_service


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    import asyncio
    
    async def test_vector_service():
        """Test the vector service functionality"""
        print("=" * 70)
        print("VECTOR SERVICE TEST")
        print("=" * 70)
        
        # Initialize service
        service = get_vector_service(host="localhost")  # Change to localhost for local testing
        
        # Ensure collection exists
        print("\n1. Ensuring collection exists...")
        service.ensure_collection(vector_size=384)
        
        # Get stats
        print("\n2. Collection statistics:")
        stats = service.get_collection_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test data
        material_id = "test_material_001"
        chunks = [
            "This is the first chunk of text about artificial intelligence.",
            "Machine learning is a subset of AI that focuses on data.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        # Create dummy embeddings (in production, use real embeddings)
        embeddings = [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384
        ]
        
        # Upsert embeddings
        print(f"\n3. Upserting {len(chunks)} chunks for material '{material_id}'...")
        result = await service.upsert_embeddings(material_id, chunks, embeddings)
        print(f"   Result: {result}")
        
        # Retrieve chunks
        print(f"\n4. Retrieving chunks for material '{material_id}'...")
        retrieved_chunks = await service.get_material_chunks(material_id)
        print(f"   Found {len(retrieved_chunks)} chunks")
        for chunk in retrieved_chunks:
            print(f"   - Chunk {chunk['chunk_id']}: {chunk['text'][:50]}...")
        
        # Search similar
        print("\n5. Searching for similar chunks...")
        query_embedding = [0.15] * 384
        results = await service.search_similar_chunks(query_embedding, limit=5)
        print(f"   Found {len(results)} similar chunks")
        for result in results:
            print(f"   - Score: {result['score']:.3f}, Text: {result['text'][:50]}...")
        
        # Health check
        print("\n6. Health check...")
        is_healthy = service.health_check()
        print(f"   Status: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")
        
        # Cleanup
        print(f"\n7. Cleaning up test data...")
        await service.delete_material_embeddings(material_id)
        
        print("\n" + "=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
    
    # Run test
    asyncio.run(test_vector_service())