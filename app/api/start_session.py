"""
Start Session API Routes
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.start_session import (
    StartSessionRequest,
    StartSessionResponse
)
from app.services.start_session_service import (
    StartSessionService,
    StartSessionServiceError
)

logger = logging.getLogger(__name__)

# Create router (can be merged with quiz router or kept separate)
router = APIRouter(prefix="/api/quiz")


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_db() -> AsyncIOMotorDatabase:
    """Dependency to get MongoDB database instance"""
    from app.db.mongodb import get_database
    return get_database()


def get_start_session_service(
    db: AsyncIOMotorDatabase = Depends(get_db)
) -> StartSessionService:
    """Dependency to get StartSessionService instance"""
    return StartSessionService(db=db)


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/start-session",
    response_model=StartSessionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {
            "description": "Quiz session successfully created",
            "model": StartSessionResponse
        },
        400: {
            "description": "Invalid request"
        },
        500: {
            "description": "Internal server error"
        }
    },
    summary="Start a new quiz session",
    description="""
    Create a new quiz session for a study material.
    
    **Workflow:**
    1. Creates a new session document in MongoDB
    2. Sets the initial difficulty to "medium" (default)
    3. Returns the session ID and initial difficulty
    
    **Default Settings:**
    - Initial difficulty: medium
    - Status: ongoing
    - Questions: empty array
    
    **Next Steps:**
    After starting a session, use the `/next-question` endpoint to generate questions.
    """
)
async def start_quiz_session(
    request: StartSessionRequest,
    service: StartSessionService = Depends(get_start_session_service)
) -> StartSessionResponse:
    """
    Start a new quiz session
    
    Args:
        request: StartSessionRequest with material ID
        service: StartSessionService instance (injected)
    
    Returns:
        StartSessionResponse with session ID and initial difficulty
    
    Raises:
        HTTPException: On validation or creation errors
    
    Example:
        POST /api/quiz/start-session
        {
            "materialId": "material_python_basics"
        }
        
        Response:
        {
            "sessionId": "550e8400-e29b-41d4-a716-446655440000",
            "materialId": "material_python_basics",
            "difficulty": "medium"
        }
    """
    try:
        # Validate material ID is not empty
        if not request.materialId or not request.materialId.strip():
            logger.error("Empty material ID provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Material ID cannot be empty"
            )
        
        # Start the session
        result = await service.start_session(
            material_id=request.materialId.strip()
        )
        
        return StartSessionResponse(**result)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    
    except StartSessionServiceError as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error starting session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while starting the session"
        )


@router.post(
    "/start-session/custom",
    response_model=StartSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start session with custom difficulty",
    description="Start a quiz session with a custom initial difficulty level"
)
async def start_quiz_session_custom(
    material_id: str,
    initial_difficulty: str = "medium",
    service: StartSessionService = Depends(get_start_session_service)
) -> StartSessionResponse:
    """
    Start a new quiz session with custom initial difficulty
    
    This is an alternative endpoint that allows specifying the initial difficulty.
    Most users should use the standard /start-session endpoint.
    
    Args:
        material_id: ID of the study material
        initial_difficulty: Initial difficulty level (default: "medium")
        service: StartSessionService instance (injected)
    
    Returns:
        StartSessionResponse with session ID and specified difficulty
    """
    try:
        # Validate difficulty
        valid_difficulties = ["easy", "medium", "hard"]
        if initial_difficulty not in valid_difficulties:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid difficulty. Must be one of: {valid_difficulties}"
            )
        
        result = await service.start_session(
            material_id=material_id,
            initial_difficulty=initial_difficulty
        )
        
        return StartSessionResponse(**result)
    
    except HTTPException:
        raise
    
    except StartSessionServiceError as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )
