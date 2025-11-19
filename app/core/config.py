"""
Application configuration settings
FILE: app/core/config.py
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # MongoDB Configuration
    mongodb_url: str = "mongodb://localhost:27017"
    database_name: str = "quiz_app"
    
    # Also support the new names for materials system
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "materials_db"
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "quiz_embeddings"
    qdrant_vector_size: int = 384
    
    # Additional Qdrant fields for materials system
    qdrant_collection: str = "materials_vectors"
    
    # Upload Configuration
    upload_dir: str = "uploads"
    max_file_size: int = 10485760
    
    # Embedding Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    
    class Config:
        env_file = ".env"
        case_sensitive = False  # This allows case-insensitive matching
        extra = "allow"  # This allows extra fields


settings = Settings()