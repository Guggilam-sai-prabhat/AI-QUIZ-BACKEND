"""
Start Session Service
Business logic for creating quiz sessions
"""
import logging
from typing import Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.services.quiz_session_service import QuizSessionService
from app.models.quiz_sessions import QuizSession

logger = logging.getLogger(__name__)


class StartSessionServiceError(Exception):
    """Base exception for start session service errors"""
    pass


class StartSessionService:
    """
    Service for handling quiz session creation
    
    This service handles:
    1. Creating a new quiz session in MongoDB
    2. Setting default difficulty level
    3. Returning session information
    """
    
    DEFAULT_DIFFICULTY = "medium"
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize start session service
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.session_service = QuizSessionService(db)
    
    async def start_session(
        self,
        material_id: str,
        initial_difficulty: str = None
    ) -> Dict[str, Any]:
        """
        Create a new quiz session with default settings
        
        Args:
            material_id: ID of the study material
            initial_difficulty: Optional initial difficulty (defaults to "medium")
        
        Returns:
            Dictionary with session ID, material ID, and difficulty
        
        Raises:
            StartSessionServiceError: If session creation fails
        
        Example:
            result = await service.start_session("material_123")
            # Returns: {"sessionId": "...", "materialId": "material_123", "difficulty": "medium"}
        """
        # Use default difficulty if not specified
        difficulty = initial_difficulty or self.DEFAULT_DIFFICULTY
        
        logger.info(
            f"🎬 Starting new quiz session - "
            f"Material: {material_id}, Initial difficulty: {difficulty}"
        )
        
        try:
            # Create session in MongoDB
            session = await self._create_session(material_id)
            
            # Build response
            response = {
                "sessionId": session.sessionId,
                "materialId": session.materialId,
                "difficulty": difficulty
            }
            
            logger.info(
                f"✅ Successfully started quiz session - "
                f"Session ID: {session.sessionId}, Difficulty: {difficulty}"
            )
            
            return response
        
        except Exception as e:
            logger.error(f"❌ Failed to start session for material {material_id}: {e}")
            raise StartSessionServiceError(
                f"Failed to start quiz session: {str(e)}"
            )
    
    async def _create_session(self, material_id: str) -> QuizSession:
        """
        Create a new quiz session in MongoDB
        
        Args:
            material_id: ID of the study material
        
        Returns:
            Created QuizSession object
        
        Raises:
            Exception: If database operation fails
        """
        try:
            session = await self.session_service.create_session(material_id)
            
            logger.debug(
                f"✓ Created session in MongoDB - "
                f"Session ID: {session.sessionId}"
            )
            
            return session
        
        except Exception as e:
            logger.error(f"❌ Error creating session in MongoDB: {e}")
            raise
    
    async def validate_material_exists(self, material_id: str) -> bool:
        """
        Optional: Validate that material exists before creating session
        
        This is a placeholder - implement based on your material storage
        
        Args:
            material_id: ID of the study material
        
        Returns:
            True if material exists
        """
        # TODO: Add validation logic if you have a materials collection
        # For now, we assume material exists
        logger.debug(f"Skipping material validation for: {material_id}")
        return True
