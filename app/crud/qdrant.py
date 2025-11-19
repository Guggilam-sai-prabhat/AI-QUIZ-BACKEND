"""
CRUD operations for Qdrant
Pure database operations - NO business logic
"""
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Any, Optional
import logging
from app.db.qdrant import get_qdrant_client
from app.core.config import settings

logger = logging.getLogger(__name__)


# ==================== CREATE ====================

def create_embedding(
    document_id: str,
    embedding: List[float],
    payload: Dict[str, Any]
) -> bool:
    """
    Create a new embedding point in Qdrant
    
    Args:
        document_id: Unique ID
        embedding: Vector (384 dimensions)
        payload: Metadata dictionary
        
    Returns:
        bool: Success status
    """
    try:
        client = get_qdrant_client()
        if not client:
            return False
        
        point = PointStruct(
            id=document_id,
            vector=embedding,
            payload=payload
        )
        
        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=[point]
        )
        
        logger.debug(f"Created embedding: {document_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create embedding: {str(e)}")
        return False


def batch_create_embeddings(points_data: List[Dict[str, Any]]) -> bool:
    """
    Create multiple embeddings at once
    
    Args:
        points_data: List of dicts with 'id', 'vector', 'payload'
        
    Returns:
        bool: Success status
    """
    try:
        client = get_qdrant_client()
        if not client:
            return False
        
        points = [
            PointStruct(
                id=data["id"],
                vector=data["vector"],
                payload=data["payload"]
            )
            for data in points_data
        ]
        
        client.upsert(
            collection_name=settings.qdrant_collection_name,
            points=points
        )
        
        logger.debug(f"Batch created {len(points)} embeddings")
        return True
        
    except Exception as e:
        logger.error(f"Failed batch create: {str(e)}")
        return False


# ==================== READ ====================

def get_embedding(document_id: str, with_vector: bool = True) -> Optional[Dict[str, Any]]:
    """
    Get a single embedding by ID
    
    Args:
        document_id: Document ID
        with_vector: Include vector in response
        
    Returns:
        dict or None
    """
    try:
        client = get_qdrant_client()
        if not client:
            return None
        
        points = client.retrieve(
            collection_name=settings.qdrant_collection_name,
            ids=[document_id],
            with_vectors=with_vector,
            with_payload=True
        )
        
        if not points:
            return None
        
        point = points[0]
        return {
            "id": point.id,
            "vector": point.vector if with_vector else None,
            "payload": point.payload
        }
        
    except Exception as e:
        logger.error(f"Failed to get embedding: {str(e)}")
        return None


def list_embeddings(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    List embeddings with pagination
    
    Args:
        limit: Max results
        offset: Skip results
        
    Returns:
        List of embeddings
    """
    try:
        client = get_qdrant_client()
        if not client:
            return []
        
        points, _ = client.scroll(
            collection_name=settings.qdrant_collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            {
                "id": point.id,
                "payload": point.payload
            }
            for point in points
        ]
        
    except Exception as e:
        logger.error(f"Failed to list embeddings: {str(e)}")
        return []


def search_embeddings(
    query_vector: List[float],
    limit: int = 10,
    score_threshold: Optional[float] = None,
    filter_dict: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for similar embeddings
    
    Args:
        query_vector: Query embedding
        limit: Max results
        score_threshold: Minimum similarity score
        filter_dict: Metadata filters
        
    Returns:
        List of similar documents
    """
    try:
        client = get_qdrant_client()
        if not client:
            return []
        
        # Build filter if provided
        query_filter = None
        if filter_dict:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_dict.items()
            ]
            query_filter = Filter(must=conditions)
        
        results = client.search(
            collection_name=settings.qdrant_collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload
            }
            for result in results
        ]
        
    except Exception as e:
        logger.error(f"Failed to search embeddings: {str(e)}")
        return []


# ==================== UPDATE ====================

def update_embedding(
    document_id: str,
    embedding: Optional[List[float]] = None,
    payload: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Update embedding vector and/or payload
    
    Args:
        document_id: Document ID
        embedding: New vector (optional)
        payload: New payload (optional)
        
    Returns:
        bool: Success status
    """
    try:
        client = get_qdrant_client()
        if not client:
            return False
        
        if embedding:
            # Update entire point (vector + payload)
            point = PointStruct(
                id=document_id,
                vector=embedding,
                payload=payload or {}
            )
            client.upsert(
                collection_name=settings.qdrant_collection_name,
                points=[point]
            )
        elif payload:
            # Update only payload
            client.set_payload(
                collection_name=settings.qdrant_collection_name,
                payload=payload,
                points=[document_id]
            )
        
        logger.debug(f"Updated embedding: {document_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update embedding: {str(e)}")
        return False


# ==================== DELETE ====================

def delete_embedding(document_id: str) -> bool:
    """
    Delete a single embedding
    
    Args:
        document_id: Document ID
        
    Returns:
        bool: Success status
    """
    try:
        client = get_qdrant_client()
        if not client:
            return False
        
        client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=[document_id]
        )
        
        logger.debug(f"Deleted embedding: {document_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete embedding: {str(e)}")
        return False


def batch_delete_embeddings(document_ids: List[str]) -> bool:
    """
    Delete multiple embeddings
    
    Args:
        document_ids: List of document IDs
        
    Returns:
        bool: Success status
    """
    try:
        client = get_qdrant_client()
        if not client:
            return False
        
        client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=document_ids
        )
        
        logger.debug(f"Batch deleted {len(document_ids)} embeddings")
        return True
        
    except Exception as e:
        logger.error(f"Failed batch delete: {str(e)}")
        return False


# ==================== UTILITY ====================

def count_embeddings() -> int:
    """Get total count of embeddings"""
    try:
        client = get_qdrant_client()
        if not client:
            return 0
        
        info = client.get_collection(settings.qdrant_collection_name)
        return info.points_count
        
    except Exception as e:
        logger.error(f"Failed to count embeddings: {str(e)}")
        return 0


def embedding_exists(document_id: str) -> bool:
    """Check if embedding exists"""
    result = get_embedding(document_id, with_vector=False)
    return result is not None