from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class MongoDB:
    client: AsyncIOMotorClient = None
    
mongodb = MongoDB()

async def connect_to_mongo():
    """Connect to MongoDB and test the connection"""
    try:
        mongodb.client = AsyncIOMotorClient(settings.mongodb_url)
        # Test connection
        await mongodb.client.admin.command('ping')
        logger.info(f"✓ Connected to MongoDB at {settings.mongodb_url}")
    except Exception as e:
        logger.error(f"✗ Failed to connect to MongoDB: {e}")
        raise
    
async def close_mongo_connection():
    """Close MongoDB connection"""
    if mongodb.client:
        mongodb.client.close()
        logger.info("✓ Closed MongoDB connection")

def get_database():
    """Get the database instance"""
    return mongodb.client[settings.database_name]

def get_collection(collection_name: str):
    """Get a specific collection"""
    db = get_database()
    return db[collection_name]