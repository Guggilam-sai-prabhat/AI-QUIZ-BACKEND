"""
Search API Routes
Semantic search endpoints using vector embeddings
FILE: app/api/search.py

COMPLETE IMPLEMENTATION - Copy this entire file
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
import logging
import time

from app.models.search import (
    SearchContextRequest,
    SearchContextResponse,
    SearchResult
)
from app.db.qdrant import get_qdrant_service, check_qdrant_health
from app.services.embedding_service import get_embeddings
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

router = APIRouter()


# ==================== DEPENDENCY INJECTION ====================

def get_qdrant_dependency() -> QdrantService:
    """
    Dependency for getting Qdrant service instance
    
    Returns:
        QdrantService: Configured Qdrant service
        
    Raises:
        HTTPException: If service is unavailable
    """
    try:
        return get_qdrant_service()
    except Exception as e:
        logger.error(f"Failed to get Qdrant service: {e}")
        raise HTTPException(
            status_code=503,
            detail="Vector database service unavailable"
        )


# ==================== SEARCH ENDPOINT ====================

@router.post(
    "/search-context",
    response_model=SearchContextResponse,
    summary="Search for similar content",
    description="""
    Perform semantic search to find relevant chunks of text.
    
    ## How it works:
    1. Embeds the query text using the same model as the indexed documents
    2. Performs vector similarity search in Qdrant
    3. Returns the most similar chunks with relevance scores
    
    ## Parameters:
    - **query**: The search text (required, 1-1000 characters)
    - **top_k**: Number of results to return (default: 5, max: 50)
    - **material_id**: Optional filter to search within a specific document
    - **score_threshold**: Optional minimum similarity score (0.0-1.0)
    
    ## Returns:
    - List of relevant text chunks with similarity scores
    - Metadata including material_id, chunk_id, and text content
    - Execution time in milliseconds
    
    ## Example Request:
    ```json
    {
      "query": "What are the safety protocols?",
      "top_k": 5,
      "score_threshold": 0.7
    }
    ```
    
    ## Example Response:
    ```json
    {
      "query": "What are the safety protocols?",
      "results": [
        {
          "material_id": "doc_123",
          "chunk_id": 5,
          "score": 0.89,
          "similarity_percentage": "89.0%",
          "text": "Safety protocols require...",
          "text_length": 150
        }
      ],
      "total_results": 1,
      "top_k_requested": 5,
      "execution_time_ms": 125.5
    }
    ```
    """,
    responses={
        200: {
            "description": "Search completed successfully",
            "model": SearchContextResponse
        },
        400: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["body", "query"],
                                "msg": "Query cannot be empty",
                                "type": "value_error"
                            }
                        ]
                    }
                }
            }
        },
        500: {
            "description": "Server error during search",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Failed to generate query embedding"
                    }
                }
            }
        },
        503: {
            "description": "Service unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Vector database service unavailable"
                    }
                }
            }
        }
    }
)
async def search_context(
    request: SearchContextRequest,
    qdrant_service: QdrantService = Depends(get_qdrant_dependency)
):
    """
    Search for semantically similar content using vector embeddings
    
    Args:
        request: Search request with query and parameters
        qdrant_service: Injected Qdrant service instance
        
    Returns:
        SearchContextResponse with results and metadata
        
    Raises:
        HTTPException: If search fails or service unavailable
    """
    start_time = time.time()
    
    try:
        logger.info(
            f"🔍 Search request: query='{request.query[:50]}...', "
            f"top_k={request.top_k}, material_id={request.material_id or 'all'}"
        )
        
        # ==================== STEP 1: Generate Query Embedding ====================
        try:
            query_embeddings = get_embeddings(
                chunks=[request.query],
                model="sentence-transformers",
                model_name="all-MiniLM-L6-v2"
            )
            
            if not query_embeddings or len(query_embeddings) == 0:
                raise ValueError("Failed to generate query embedding")
            
            query_vector = query_embeddings[0]
            logger.debug(f"✓ Generated query embedding (dim: {len(query_vector)})")
            
        except Exception as e:
            logger.error(f"❌ Embedding generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate query embedding: {str(e)}"
            )
        
        # ==================== STEP 2: Search Qdrant ====================
        try:
            search_results = await qdrant_service.search_similar_chunks(
                query_embedding=query_vector,
                limit=request.top_k,
                material_id=request.material_id,
                score_threshold=request.score_threshold
            )
            
            logger.info(f"✓ Search returned {len(search_results)} results")
            
        except Exception as e:
            logger.error(f"❌ Vector search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Vector search failed: {str(e)}"
            )
        
        # ==================== STEP 3: Format Results ====================
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                SearchResult(
                    material_id=result["material_id"],
                    chunk_id=result["chunk_id"],
                    score=result["score"],
                    similarity_percentage=result["similarity_percentage"],
                    text=result["text"],
                    text_length=len(result["text"])
                )
            )
        
        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Build response
        response = SearchContextResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results),
            top_k_requested=request.top_k,
            execution_time_ms=round(execution_time, 2)
        )
        
        logger.info(
            f"✓ Search completed: {len(formatted_results)} results in {execution_time:.2f}ms"
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in search endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during search"
        )


# ==================== HEALTH CHECK ====================

@router.get(
    "/search-health",
    summary="Check search service health",
    description="Verify that the search service and vector database are operational",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "message": "Search service is operational",
                        "qdrant": {
                            "status": "healthy",
                            "connected": True,
                            "host": "localhost",
                            "port": 6333
                        },
                        "collection": {
                            "name": "materials_vectors",
                            "points_count": 1523,
                            "status": "green"
                        }
                    }
                }
            }
        }
    }
)
async def search_health():
    """
    Health check endpoint for search service
    
    Returns:
        Health status and collection information
    """
    try:
        health_info = await check_qdrant_health()
        
        if health_info["status"] == "healthy":
            # Get additional stats
            try:
                qdrant_service = get_qdrant_service()
                stats = qdrant_service.get_collection_stats()
                
                return {
                    "status": "healthy",
                    "message": "Search service is operational",
                    "qdrant": health_info,
                    "collection": {
                        "name": stats.get("name"),
                        "points_count": stats.get("points_count"),
                        "vectors_count": stats.get("vectors_count"),
                        "status": stats.get("status")
                    }
                }
            except Exception as e:
                logger.warning(f"Could not get collection stats: {e}")
                return {
                    "status": "healthy",
                    "message": "Search service is operational",
                    "qdrant": health_info
                }
        else:
            return {
                "status": "unhealthy",
                "message": "Vector database connection failed",
                "qdrant": health_info
            }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": str(e)
        }


# ==================== COLLECTION INFO ====================

@router.get(
    "/collection-info",
    summary="Get collection information",
    description="Get detailed information about the vector database collection",
    responses={
        200: {
            "description": "Collection information retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "collection": {
                            "name": "materials_vectors",
                            "points_count": 1523,
                            "vectors_count": 1523,
                            "indexed_vectors_count": 1523,
                            "status": "green",
                            "health": "healthy",
                            "vector_size": 384,
                            "distance": "COSINE"
                        },
                        "message": "Collection information retrieved successfully"
                    }
                }
            }
        },
        500: {
            "description": "Failed to retrieve collection information"
        }
    }
)
async def get_collection_info(
    qdrant_service: QdrantService = Depends(get_qdrant_dependency)
):
    """
    Get information about the Qdrant collection
    
    Returns:
        Collection statistics and configuration
    """
    try:
        stats = qdrant_service.get_collection_stats()
        
        return {
            "collection": stats,
            "message": "Collection information retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve collection information: {str(e)}"
        )


# ==================== BATCH SEARCH (OPTIONAL) ====================

@router.post(
    "/batch-search",
    summary="Batch search multiple queries",
    description="Search multiple queries at once for efficiency"
)
async def batch_search(
    queries: List[str],
    top_k: int = 5,
    qdrant_service: QdrantService = Depends(get_qdrant_dependency)
):
    """
    Perform batch search for multiple queries
    
    Args:
        queries: List of query strings
        top_k: Number of results per query
        
    Returns:
        List of search results for each query
    """
    if len(queries) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 queries allowed per batch request"
        )
    
    results = []
    
    for query in queries:
        try:
            # Generate embedding
            embeddings = get_embeddings([query])
            
            # Search
            search_results = await qdrant_service.search_similar_chunks(
                query_embedding=embeddings[0],
                limit=top_k
            )
            
            results.append({
                "query": query,
                "results": search_results,
                "total": len(search_results)
            })
            
        except Exception as e:
            logger.error(f"Batch search failed for query '{query}': {e}")
            results.append({
                "query": query,
                "results": [],
                "total": 0,
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_queries": len(queries),
        "successful": sum(1 for r in results if "error" not in r)
    }
