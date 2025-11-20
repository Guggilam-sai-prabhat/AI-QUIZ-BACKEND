"""
Qdrant Database Connection Manager
UPDATED VERSION - Replace your entire app/db/qdrant.py with this
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse
import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class QdrantDB:
    client: Optional[QdrantClient] = None
    is_connected: bool = False

qdrant_db = QdrantDB()

def connect_to_qdrant():
    """
    Connect to Qdrant and ensure collection exists
    """
    try:
        # Create Qdrant client
        url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
        qdrant_db.client = QdrantClient(url=url)
        
        # Test connection
        collections = qdrant_db.client.get_collections()
        logger.info(f"✅ Connected to Qdrant at {url}")
        logger.info(f"📊 Found {len(collections.collections)} existing collections")
        
        qdrant_db.is_connected = True
        
        # Ensure materials collection exists
        _ensure_collection_exists()
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to Qdrant: {str(e)}")
        logger.warning("⚠️ Application will continue without Qdrant")
        qdrant_db.is_connected = False

def _ensure_collection_exists():
    """
    Check if collection exists, create it if not
    """
    try:
        collection_name = settings.qdrant_collection_name
        
        # Get existing collections
        collections = qdrant_db.client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name in collection_names:
            logger.info(f"✅ Collection '{collection_name}' already exists")
            
            # Get and log collection info
            info = qdrant_db.client.get_collection(collection_name)
            logger.info(f"📊 Collection has {info.points_count} points")
        else:
            logger.info(f"📦 Creating collection '{collection_name}'...")
            
            # Create collection with dummy vector field
            qdrant_db.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.qdrant_vector_size,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"✅ Collection '{collection_name}' created successfully")
            logger.info(f"   Vector size: {settings.qdrant_vector_size}")
            logger.info(f"   Distance metric: COSINE")
            
    except UnexpectedResponse as e:
        logger.error(f"❌ Qdrant API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to ensure collection exists: {str(e)}")
        raise

def ping_qdrant() -> bool:
    """
    Ping Qdrant to confirm connection
    
    Returns:
        bool: True if connected and healthy, False otherwise
    """
    try:
        if not qdrant_db.client:
            logger.warning("⚠️ Qdrant client not initialized")
            return False
        
        # Try to get collections as health check
        collections = qdrant_db.client.get_collections()
        
        logger.info(f"✅ Qdrant ping successful - {len(collections.collections)} collections")
        qdrant_db.is_connected = True
        return True
        
    except Exception as e:
        logger.error(f"❌ Qdrant ping failed: {str(e)}")
        qdrant_db.is_connected = False
        return False

def get_qdrant_client() -> Optional[QdrantClient]:
    """
    Get the Qdrant client instance
    
    Returns:
        QdrantClient or None if not connected
    """
    return qdrant_db.client

def get_collection_info() -> Optional[dict]:
    """
    Get information about the collection
    
    Returns:
        dict with collection info or None if error
    """
    try:
        if not qdrant_db.client or not qdrant_db.is_connected:
            return None
        
        collection_name = settings.qdrant_collection_name
        info = qdrant_db.client.get_collection(collection_name)
        
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status.value if hasattr(info.status, 'value') else str(info.status)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get collection info: {str(e)}")
        return None

def close_qdrant_connection():
    """Close Qdrant connection"""
    if qdrant_db.client:
        qdrant_db.client.close()
        logger.info("👋 Qdrant connection closed")
        qdrant_db.is_connected = False


# ==================== NEW FUNCTIONS FOR SEARCH API ====================

def get_qdrant_service():
    """
    Get QdrantService instance with business logic
    Lazy import to avoid circular dependencies
    
    Returns:
        QdrantService: Service instance
    """
    from app.services.qdrant_service import QdrantService
    
    if not qdrant_db.client or not qdrant_db.is_connected:
        logger.warning("⚠️ Qdrant not connected, attempting to connect...")
        connect_to_qdrant()
    
    return QdrantService(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.qdrant_vector_size
    )


async def check_qdrant_health() -> dict:
    """
    Async health check for Qdrant
    
    Returns:
        dict: Health status information
    """
    try:
        if ping_qdrant():
            info = get_collection_info()
            
            return {
                "status": "healthy",
                "connected": True,
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "collection": info
            }
        else:
            return {
                "status": "unhealthy",
                "connected": False,
                "host": settings.qdrant_host,
                "port": settings.qdrant_port,
                "error": "Connection failed"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "connected": False,
            "host": settings.qdrant_host,
            "port": settings.qdrant_port,
            "error": str(e)
        }


def is_qdrant_connected() -> bool:
    """Check if Qdrant is currently connected"""
    return qdrant_db.is_connected


def get_collection_name() -> str:
    """Get the configured collection name"""
    return settings.qdrant_collection_name