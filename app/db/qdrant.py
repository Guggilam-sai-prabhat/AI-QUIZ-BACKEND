from qdrant_client import QdrantClient
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class QdrantDB:
    client: QdrantClient = None

qdrant_db = QdrantDB()

def connect_to_qdrant():
    """Connect to Qdrant and test the connection"""
    try:
        qdrant_db.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        # Test connection by getting collections
        collections = qdrant_db.client.get_collections()
        logger.info(f"✓ Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
    except Exception as e:
        logger.error(f"✗ Failed to connect to Qdrant: {e}")
        # Don't raise - Qdrant is optional for now
    
def get_qdrant_client():
    """Get the Qdrant client instance"""
    return qdrant_db.client