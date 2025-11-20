"""
Search Models
Pydantic models for search request and response
FILE: app/models/search.py
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional


class SearchContextRequest(BaseModel):
    """Request model for semantic search"""
    
    query: str = Field(
        ..., 
        min_length=1,
        max_length=1000,
        description="Search query text",
        example="What are the safety protocols?"
    )
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return (1-50)",
        example=5
    )
    
    material_id: Optional[str] = Field(
        None,
        description="Optional filter to search within specific material/document",
        example="mat_123"
    )
    
    score_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0-1.0)",
        example=0.7
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Ensure query is not just whitespace"""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the safety protocols?",
                "top_k": 5,
                "material_id": "mat_123",
                "score_threshold": 0.7
            }
        }


class SearchResult(BaseModel):
    """Single search result with metadata"""
    
    material_id: str = Field(
        ...,
        description="Material/document identifier",
        example="mat_123"
    )
    
    chunk_id: int = Field(
        ...,
        description="Chunk index within the material",
        example=12
    )
    
    score: float = Field(
        ...,
        description="Similarity score (0-1, higher is better)",
        example=0.89
    )
    
    similarity_percentage: str = Field(
        ...,
        description="Formatted similarity percentage",
        example="89.0%"
    )
    
    text: str = Field(
        ...,
        description="Chunk text content",
        example="Safety protocols require proper protective equipment..."
    )
    
    text_length: int = Field(
        ...,
        description="Length of text in characters",
        example=250
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "material_id": "mat_123",
                "chunk_id": 12,
                "score": 0.89,
                "similarity_percentage": "89.0%",
                "text": "Safety protocols require proper protective equipment and regular inspections.",
                "text_length": 76
            }
        }


class SearchContextResponse(BaseModel):
    """Response model for search results"""
    
    query: str = Field(
        ...,
        description="Original search query",
        example="What are the safety protocols?"
    )
    
    results: List[SearchResult] = Field(
        ...,
        description="List of search results ordered by relevance",
        example=[]
    )
    
    total_results: int = Field(
        ...,
        description="Number of results returned",
        example=5
    )
    
    top_k_requested: int = Field(
        ...,
        description="Number of results requested",
        example=5
    )
    
    execution_time_ms: float = Field(
        ...,
        description="Query execution time in milliseconds",
        example=125.5
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the safety protocols?",
                "results": [
                    {
                        "material_id": "mat_123",
                        "chunk_id": 12,
                        "score": 0.89,
                        "similarity_percentage": "89.0%",
                        "text": "Safety protocols require proper protective equipment...",
                        "text_length": 76
                    }
                ],
                "total_results": 1,
                "top_k_requested": 5,
                "execution_time_ms": 125.5
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str = Field(
        ...,
        description="Error message",
        example="Search failed"
    )
    
    detail: Optional[str] = Field(
        None,
        description="Detailed error information",
        example="Connection to vector database failed"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Search failed",
                "detail": "Connection to vector database failed"
            }
        }