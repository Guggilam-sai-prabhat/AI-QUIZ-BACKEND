"""
Pydantic models for document management
FILE: app/models/document.py
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic v2"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string"}


class ChunkMetadata(BaseModel):
    """
    Metadata for individual text chunks
    """
    chunk_id: int
    text: str
    text_length: int
    difficulty: str  # "easy", "medium", "hard"
    indexed_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentMetadata(BaseModel):
    """
    Metadata for uploaded documents (stored in MongoDB)
    Includes all fields including extracted_text which is not returned in API responses
    """
    
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    filename: str
    file_path: str
    file_size: int
    content_type: str
    file_hash: str
    
    # PDF extraction data (stored but not returned in list responses)
    extracted_text: Optional[str] = None
    page_count: Optional[int] = None
    
    # Embedding pipeline metadata
    chunk_count: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_model_name: Optional[str] = None
    embedding_dimension: Optional[int] = None
    embedding_status: Optional[str] = None  # "embedded", "failed", "pending"
    embedding_error: Optional[str] = None
    indexed_at: Optional[datetime] = None
    embedding_attempted_at: Optional[datetime] = None
    
    # Difficulty rating metadata
    difficulty_distribution: Optional[Dict[str, int]] = None  # {"easy": 5, "medium": 8, "hard": 3}
    difficulty_provider: Optional[str] = None  # "openai", "anthropic", "grok", "rule-based"
    chunks: Optional[List[Dict[str, Any]]] = None  # Array of chunk metadata
    
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


class DocumentResponse(BaseModel):
    """
    Response model for document list queries
    Excludes extracted_text to keep responses lightweight
    """
    
    id: str
    filename: str
    file_size: int
    content_type: str
    file_hash: str
    uploaded_at: datetime
    
    # Optional embedding metadata
    chunk_count: Optional[int] = None
    embedding_status: Optional[str] = None
    
    # Difficulty metadata
    difficulty_distribution: Optional[Dict[str, int]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DocumentDetailResponse(BaseModel):
    """
    Detailed response model for single document queries
    Includes extracted_text and page_count
    """
    
    id: str
    filename: str
    file_size: int
    content_type: str
    file_hash: str
    uploaded_at: datetime
    
    # PDF content (included in detail view)
    extracted_text: Optional[str] = None
    page_count: Optional[int] = None
    
    # Embedding metadata
    chunk_count: Optional[int] = None
    embedding_model: Optional[str] = None
    embedding_model_name: Optional[str] = None
    embedding_dimension: Optional[int] = None
    embedding_status: Optional[str] = None
    indexed_at: Optional[datetime] = None
    
    # Difficulty metadata
    difficulty_distribution: Optional[Dict[str, int]] = None
    difficulty_provider: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EmbeddingResponse(BaseModel):
    """
    Response model for document upload with embedding pipeline results
    Returned after successful upload and embedding processing
    """
    
    # Document metadata
    id: str
    filename: str
    file_size: int
    content_type: str
    file_hash: str
    uploaded_at: datetime
    
    # Embedding pipeline results
    material_id: str
    chunk_count: int
    status: str  # "embedded", "no_content", "error"
    embedding_dimension: Optional[int] = None
    embedding_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Difficulty rating results
    difficulty_distribution: Optional[Dict[str, int]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SearchQuery(BaseModel):
    """
    Request model for semantic search
    """
    
    query: str = Field(..., description="Search query text", min_length=1)
    limit: int = Field(
        default=10, 
        ge=1, 
        le=100, 
        description="Maximum number of results to return"
    )
    material_id: Optional[str] = Field(
        default=None, 
        description="Optional: limit search to specific material"
    )
    difficulty: Optional[str] = Field(
        default=None,
        description="Optional: filter by difficulty level (easy, medium, hard)",
        pattern="^(easy|medium|hard)$"
    )
    score_threshold: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity score (0.0-1.0)"
    )


class SearchResult(BaseModel):
    """
    Single search result with relevance score
    """
    
    id: str
    score: float
    material_id: str
    chunk_id: int
    text: str
    difficulty: str  # "easy", "medium", "hard"
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """
    Response for semantic search queries
    """
    
    query: str
    results: list[SearchResult]
    total_results: int
    search_time_ms: Optional[float] = None
    filters_applied: Optional[Dict[str, Any]] = None


class ChunkResponse(BaseModel):
    """
    Response model for individual chunk data
    """
    
    id: str
    chunk_id: int
    text: str
    difficulty: str
    text_length: int
    material_id: str
    metadata: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ChunksListResponse(BaseModel):
    """
    Response model for listing chunks from a material
    """
    
    material_id: str
    chunk_count: int
    difficulty_filter: Optional[str] = None
    chunks: List[Dict[str, Any]]


class ReindexRequest(BaseModel):
    """
    Request to reindex a document's embeddings
    """
    
    chunk_size: Optional[int] = Field(default=500, ge=100, le=2000)
    chunk_overlap: Optional[int] = Field(default=50, ge=0, le=500)
    force: bool = Field(
        default=False, 
        description="Force reindex even if already indexed"
    )
    recompute_difficulty: bool = Field(
        default=False,
        description="Recompute difficulty ratings during reindex"
    )


class ReindexResponse(BaseModel):
    """
    Response for reindexing operation
    """
    
    material_id: str
    status: str
    chunk_count: int
    message: str
    previous_chunk_count: Optional[int] = None
    difficulty_distribution: Optional[Dict[str, int]] = None


class BatchDeleteRequest(BaseModel):
    """
    Request to delete multiple documents
    """
    
    document_ids: list[str] = Field(..., min_items=1, max_items=100)


class BatchDeleteResponse(BaseModel):
    """
    Response for batch delete operation
    """
    
    total_requested: int
    deleted_count: int
    failed_count: int
    failed_ids: list[str] = []
    errors: Optional[Dict[str, str]] = None


class HealthCheckResponse(BaseModel):
    """
    Health check response
    """
    
    status: str  # "ok", "degraded", "down"
    timestamp: datetime
    checks: Dict[str, str]
    version: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CollectionStatsResponse(BaseModel):
    """
    Statistics about document collection
    """
    
    total_documents: int
    total_file_size: int
    total_chunks: int
    avg_chunks_per_document: float
    embedding_status_breakdown: Dict[str, int]
    difficulty_breakdown: Optional[Dict[str, int]] = None  # Total across all documents
    oldest_document: Optional[datetime] = None
    newest_document: Optional[datetime] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class VectorStatsResponse(BaseModel):
    """
    Statistics about vector collection in Qdrant
    """
    
    collection_name: str
    total_points: int
    vectors_count: int
    indexed_vectors: int
    status: str
    vector_size: int
    distance_metric: str
    difficulty_indexed: bool = True  # Indicates if difficulty field is indexed