"""
CRUD operations for Qdrant
Pure database operations - NO business logic
Handles direct interactions with Qdrant vector database
"""
from qdrant_client.models import (
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue,
    FilterSelector
)
from qdrant_client.http import models
from typing import List, Dict, Any, Optional
import logging
import uuid

logger = logging.getLogger(__name__)


# ==================== CREATE ====================

def create_point(
    client,
    collection_name: str,
    point_id: str,
    vector: List[float],
    payload: Dict[str, Any]
) -> bool:
    """
    Create a single point in Qdrant
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        point_id: Unique point ID
        vector: Embedding vector
        payload: Metadata dictionary
        
    Returns:
        bool: Success status
    """
    try:
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
        
        client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True
        )
        
        logger.debug(f"Created point: {point_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create point: {str(e)}")
        return False


def batch_create_points(
    client,
    collection_name: str,
    points_data: List[PointStruct]
) -> bool:
    """
    Create multiple points at once
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        points_data: List of PointStruct objects
        
    Returns:
        bool: Success status
    """
    try:
        client.upsert(
            collection_name=collection_name,
            points=points_data,
            wait=True
        )
        
        logger.debug(f"Batch created {len(points_data)} points")
        return True
        
    except Exception as e:
        logger.error(f"Failed batch create: {str(e)}")
        return False


# ==================== READ ====================

def get_point_by_id(
    client,
    collection_name: str,
    point_id: str,
    with_vector: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Get a single point by ID
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        point_id: Point ID
        with_vector: Include vector in response
        
    Returns:
        dict or None
    """
    try:
        points = client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
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
        logger.error(f"Failed to get point: {str(e)}")
        return None


def scroll_points(
    client,
    collection_name: str,
    limit: int = 100,
    offset: Optional[str] = None,
    with_vectors: bool = False,
    filter_condition: Optional[Filter] = None
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Scroll through points with optional filtering
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        limit: Max results
        offset: Pagination offset
        with_vectors: Include vectors in response
        filter_condition: Optional filter
        
    Returns:
        Tuple of (points list, next_offset)
    """
    try:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
            scroll_filter=filter_condition
        )
        
        points = [
            {
                "id": record.id,
                "payload": record.payload,
                "vector": record.vector if with_vectors else None
            }
            for record in records
        ]
        
        return points, next_offset
        
    except Exception as e:
        logger.error(f"Failed to scroll points: {str(e)}")
        return [], None


def search_similar(
    client,
    collection_name: str,
    query_vector: List[float],
    limit: int = 10,
    score_threshold: Optional[float] = None,
    query_filter: Optional[Filter] = None,
    with_payload: bool = True,
    with_vectors: bool = False
) -> List[Dict[str, Any]]:
    """
    Search for similar vectors
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        query_vector: Query embedding
        limit: Max results
        score_threshold: Minimum similarity score
        query_filter: Optional filter
        with_payload: Include payload in results
        with_vectors: Include vectors in results
        
    Returns:
        List of similar points with scores
    """
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            query_filter=query_filter,
            with_payload=with_payload,
            with_vectors=with_vectors
        )
        
        return [
            {
                "id": result.id,
                "score": result.score,
                "payload": result.payload if with_payload else None,
                "vector": result.vector if with_vectors else None
            }
            for result in results
        ]
        
    except Exception as e:
        logger.error(f"Failed to search: {str(e)}")
        return []


# ==================== UPDATE ====================

def update_point_vector(
    client,
    collection_name: str,
    point_id: str,
    vector: List[float],
    payload: Dict[str, Any]
) -> bool:
    """
    Update point vector and payload
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        point_id: Point ID
        vector: New vector
        payload: New payload
        
    Returns:
        bool: Success status
    """
    try:
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload
        )
        
        client.upsert(
            collection_name=collection_name,
            points=[point],
            wait=True
        )
        
        logger.debug(f"Updated point vector: {point_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update point: {str(e)}")
        return False


def update_point_payload(
    client,
    collection_name: str,
    point_id: str,
    payload: Dict[str, Any]
) -> bool:
    """
    Update only the payload (keeps vector unchanged)
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        point_id: Point ID
        payload: New payload
        
    Returns:
        bool: Success status
    """
    try:
        client.set_payload(
            collection_name=collection_name,
            payload=payload,
            points=[point_id],
            wait=True
        )
        
        logger.debug(f"Updated point payload: {point_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update payload: {str(e)}")
        return False


# ==================== DELETE ====================

def delete_point(
    client,
    collection_name: str,
    point_id: str
) -> bool:
    """
    Delete a single point
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        point_id: Point ID
        
    Returns:
        bool: Success status
    """
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=[point_id],
            wait=True
        )
        
        logger.debug(f"Deleted point: {point_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete point: {str(e)}")
        return False


def delete_points_by_filter(
    client,
    collection_name: str,
    filter_condition: Filter
) -> bool:
    """
    Delete points matching a filter
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        filter_condition: Filter to match points
        
    Returns:
        bool: Success status
    """
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=filter_condition),
            wait=True
        )
        
        logger.debug("Deleted points by filter")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete by filter: {str(e)}")
        return False


def batch_delete_points(
    client,
    collection_name: str,
    point_ids: List[str]
) -> bool:
    """
    Delete multiple points by IDs
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        point_ids: List of point IDs
        
    Returns:
        bool: Success status
    """
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=point_ids,
            wait=True
        )
        
        logger.debug(f"Batch deleted {len(point_ids)} points")
        return True
        
    except Exception as e:
        logger.error(f"Failed batch delete: {str(e)}")
        return False


# ==================== COLLECTION OPERATIONS ====================

def get_collection_info(client, collection_name: str) -> Optional[Dict[str, Any]]:
    """
    Get collection information
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        
    Returns:
        Collection info dict or None
    """
    try:
        info = client.get_collection(collection_name)
        
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
            "vector_size": info.config.params.vectors.size,
            "distance": info.config.params.vectors.distance.name
        }
        
    except Exception as e:
        logger.error(f"Failed to get collection info: {str(e)}")
        return None


def count_points(client, collection_name: str) -> int:
    """
    Get total count of points in collection
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        
    Returns:
        Point count
    """
    try:
        info = client.get_collection(collection_name)
        return info.points_count
        
    except Exception as e:
        logger.error(f"Failed to count points: {str(e)}")
        return 0


def point_exists(
    client,
    collection_name: str,
    point_id: str
) -> bool:
    """
    Check if a point exists
    
    Args:
        client: Qdrant client instance
        collection_name: Collection name
        point_id: Point ID
        
    Returns:
        bool: True if exists
    """
    result = get_point_by_id(client, collection_name, point_id, with_vector=False)
    return result is not None