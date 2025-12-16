"""
MCP Server for Quiz System
Provides tools for adaptive quiz generation with OpenAI integration
UPDATED: Supports first question with wasCorrect=null and get_session_state tool
"""
import asyncio
import logging
import os
from typing import Literal

from mcp.server import Server
from mcp.server.stdio import stdio_server
from motor.motor_asyncio import AsyncIOMotorClient

from app.services.quiz_session_service import QuizSessionService
from app.services.qdrant_service import QdrantService
from app.models.quiz_sessions import QuestionRecord

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp_server = Server("quiz-mcp-server")

# Global database and service instances
db = None
session_service = None
qdrant_service = None


# ============================================================================
# INITIALIZATION
# ============================================================================

async def initialize_services():
    """Initialize database connections and services"""
    global db, session_service, qdrant_service
    
    try:
        # MongoDB connection
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        mongo_db_name = os.getenv("MONGODB_DB_NAME", "quiz_db")
        
        client = AsyncIOMotorClient(mongo_uri)
        db = client[mongo_db_name]
        
        logger.info(f"✅ Connected to MongoDB: {mongo_db_name}")
        
        # Initialize services
        session_service = QuizSessionService(db)
        qdrant_service = QdrantService(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333"))
        )
        
        logger.info("✅ Services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise


# ============================================================================
# MCP TOOLS
# ============================================================================

@mcp_server.tool()
async def compute_next_difficulty(
    current_difficulty: Literal["easy", "medium", "hard"],
    was_correct: bool
) -> dict:
    """
    Compute the next difficulty level based on user's previous answer.
    
    Adaptive Logic:
    - Correct answer: Increase difficulty (if not already at max)
    - Incorrect answer: Decrease difficulty (if not already at min)
    - Stay at same level if at boundaries
    
    Args:
        current_difficulty: Current difficulty level
        was_correct: Whether the previous answer was correct
    
    Returns:
        Dictionary with next difficulty level and change explanation
    
    Example:
        compute_next_difficulty("medium", True) -> {"difficulty": "hard", "changed": True}
    """
    try:
        logger.info(
            f"🎯 Computing next difficulty - "
            f"Current: {current_difficulty}, WasCorrect: {was_correct}"
        )
        
        # Difficulty progression map
        difficulty_map = {
            "easy": {"correct": "medium", "incorrect": "easy"},
            "medium": {"correct": "hard", "incorrect": "easy"},
            "hard": {"correct": "hard", "incorrect": "medium"}
        }
        
        # Determine next difficulty
        next_difficulty = (
            difficulty_map[current_difficulty]["correct"] 
            if was_correct 
            else difficulty_map[current_difficulty]["incorrect"]
        )
        
        changed = next_difficulty != current_difficulty
        
        result = {
            "difficulty": next_difficulty,
            "previousDifficulty": current_difficulty,
            "changed": changed,
            "reason": (
                f"{'Increased' if changed and was_correct else 'Decreased' if changed else 'Maintained'} "
                f"difficulty based on {'correct' if was_correct else 'incorrect'} answer"
            )
        }
        
        logger.info(
            f"✅ Next difficulty computed - "
            f"Previous: {current_difficulty} → Next: {next_difficulty} "
            f"({'changed' if changed else 'unchanged'})"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"❌ Error computing next difficulty: {e}", exc_info=True)
        return {
            "error": f"Failed to compute difficulty: {str(e)}",
            "difficulty": current_difficulty,  # Fallback to current
            "changed": False
        }


@mcp_server.tool()
async def get_chunk_by_difficulty(
    material_id: str,
    difficulty: Literal["easy", "medium", "hard"]
) -> dict:
    """
    Retrieve a content chunk from Qdrant based on material and difficulty.
    
    This tool provides the source content for question generation.
    Questions MUST be generated from the returned chunk text.
    
    Args:
        material_id: Study material identifier
        difficulty: Desired difficulty level
    
    Returns:
        Dictionary containing:
        - chunk_id: Unique chunk identifier
        - content: Text content for question generation
        - difficulty: Confirmed difficulty level
        - metadata: Additional chunk information
    
    Example:
        get_chunk_by_difficulty("mat_123", "medium") -> {
            "chunk_id": "chunk_456",
            "content": "Mitochondria are the powerhouse...",
            "difficulty": "medium"
        }
    """
    try:
        logger.info(
            f"📚 Retrieving chunk - "
            f"Material: {material_id}, Difficulty: {difficulty}"
        )
        
        # Retrieve chunk from Qdrant
        chunk = await qdrant_service.get_chunk_by_difficulty(
            material_id=material_id,
            difficulty=difficulty
        )
        
        if not chunk:
            logger.warning(
                f"⚠️ No chunk found - "
                f"Material: {material_id}, Difficulty: {difficulty}"
            )
            return {
                "success": False,
                "error": f"No content available for material {material_id} at {difficulty} difficulty",
                "chunk_id": None,
                "content": None
            }
        
        result = {
            "success": True,
            "chunk_id": chunk.get("id"),
            "content": chunk.get("content"),
            "difficulty": difficulty,
            "metadata": chunk.get("metadata", {})
        }
        
        logger.info(
            f"✅ Chunk retrieved - "
            f"ChunkId: {chunk.get('id')}, "
            f"ContentLength: {len(chunk.get('content', ''))} chars"
        )
        
        return result
    
    except Exception as e:
        logger.error(
            f"❌ Error retrieving chunk for material {material_id}: {e}",
            exc_info=True
        )
        return {
            "success": False,
            "error": f"Failed to retrieve content: {str(e)}",
            "chunk_id": None,
            "content": None
        }


@mcp_server.tool()
async def record_question_event(
    session_id: str,
    question_id: str,
    difficulty: Literal["easy", "medium", "hard"],
    was_correct: bool | None  # UPDATED: Allow None for first question
) -> dict:
    """
    Record a question event in the quiz session.
    
    UPDATED: Supports wasCorrect=null for first question in session.
    
    This tool stores the question attempt in the session history for:
    - Progress tracking
    - Performance analytics
    - Difficulty adaptation on next question
    
    Args:
        session_id: The quiz session ID
        question_id: The unique question identifier
        difficulty: The difficulty level (easy/medium/hard)
        was_correct: Whether the answer was correct (None for first question)
    
    Returns:
        Success confirmation or error details
    
    Usage:
        - First question: Call with was_correct=null
        - Subsequent questions: Call with was_correct=true/false
    
    Examples:
        # First question
        record_question_event("session_123", "q_001", "medium", null)
        
        # Subsequent questions
        record_question_event("session_123", "q_002", "medium", true)
    """
    try:
        logger.info(
            f"📝 Recording question event - "
            f"Session: {session_id}, Question: {question_id}, "
            f"Difficulty: {difficulty}, WasCorrect: {was_correct}"
        )
        
        # Validate difficulty
        if difficulty not in ["easy", "medium", "hard"]:
            error_msg = f"Invalid difficulty: {difficulty}. Must be easy, medium, or hard."
            logger.error(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "recorded": False
            }
        
        # Create question record
        # UPDATED: wasCorrect can be None for first question
        question_record = QuestionRecord(
            questionId=question_id,
            difficulty=difficulty,
            wasCorrect=was_correct  # Can be None
        )
        
        # Store in session
        success = await session_service.append_question(
            session_id=session_id,
            record=question_record
        )
        
        if success:
            correctness_str = (
                "N/A (first question)" if was_correct is None 
                else f"{'✓ Correct' if was_correct else '✗ Incorrect'}"
            )
            logger.info(
                f"✅ Question event recorded - "
                f"Session: {session_id}, Question: {question_id}, "
                f"Result: {correctness_str}"
            )
            return {
                "success": True,
                "message": "Question event recorded successfully",
                "recorded": True,
                "sessionId": session_id,
                "questionId": question_id,
                "difficulty": difficulty,
                "wasCorrect": was_correct,
                "isFirstQuestion": was_correct is None
            }
        else:
            logger.warning(
                f"⚠️ Failed to record question event - "
                f"Session: {session_id}, Question: {question_id}"
            )
            return {
                "success": False,
                "error": "Failed to record question event - session may not exist",
                "recorded": False
            }
    
    except ValueError as e:
        logger.error(f"❌ Validation error while recording question event: {e}")
        return {
            "success": False,
            "error": str(e),
            "recorded": False
        }
    
    except Exception as e:
        logger.error(
            f"❌ Error recording question event for session {session_id}: {e}",
            exc_info=True
        )
        return {
            "success": False,
            "error": f"Failed to record question event: {str(e)}",
            "recorded": False
        }


@mcp_server.tool()
async def get_session_info(session_id: str) -> dict:
    """
    Get information about a quiz session.
    
    Retrieves session metadata including:
    - Total questions asked
    - Correct answers count
    - Session status
    - Material ID
    
    Args:
        session_id: The quiz session ID
    
    Returns:
        Session information or error details
    """
    try:
        logger.info(f"📊 Retrieving session info - Session: {session_id}")
        
        session = await session_service.get_session(session_id)
        
        if not session:
            logger.warning(f"⚠️ Session not found: {session_id}")
            return {
                "success": False,
                "error": f"Session not found: {session_id}",
                "exists": False
            }
        
        # Calculate statistics
        total_questions = len(session.questions)
        # Only count answered questions (wasCorrect != None)
        answered_questions = [q for q in session.questions if q.wasCorrect is not None]
        correct_count = sum(1 for q in answered_questions if q.wasCorrect)
        
        result = {
            "success": True,
            "exists": True,
            "sessionId": session.sessionId,
            "materialId": session.materialId,
            "status": session.status,
            "totalQuestions": total_questions,
            "answeredQuestions": len(answered_questions),
            "correctAnswers": correct_count,
            "accuracy": (
                round(correct_count / len(answered_questions) * 100, 1)
                if answered_questions else 0
            ),
            "createdAt": session.createdAt.isoformat(),
            "updatedAt": session.updatedAt.isoformat()
        }
        
        logger.info(
            f"✅ Session info retrieved - "
            f"Session: {session_id}, "
            f"Questions: {total_questions}, "
            f"Correct: {correct_count}/{len(answered_questions)}"
        )
        
        return result
    
    except Exception as e:
        logger.error(
            f"❌ Error retrieving session info for {session_id}: {e}",
            exc_info=True
        )
        return {
            "success": False,
            "error": f"Failed to retrieve session info: {str(e)}",
            "exists": False
        }


@mcp_server.tool()
async def validate_session_material(
    session_id: str,
    material_id: str
) -> dict:
    """
    Validate that a session belongs to a specific material.
    
    This prevents cross-material question generation attempts.
    
    Args:
        session_id: The quiz session ID
        material_id: The material ID to validate against
    
    Returns:
        Validation result with match status
    """
    try:
        logger.info(
            f"🔍 Validating session material - "
            f"Session: {session_id}, Material: {material_id}"
        )
        
        session = await session_service.get_session(session_id)
        
        if not session:
            logger.warning(f"⚠️ Session not found: {session_id}")
            return {
                "valid": False,
                "error": f"Session not found: {session_id}",
                "matches": False
            }
        
        matches = session.materialId == material_id
        
        result = {
            "valid": True,
            "matches": matches,
            "sessionMaterialId": session.materialId,
            "requestedMaterialId": material_id
        }
        
        if matches:
            logger.info(
                f"✅ Material validation passed - "
                f"Session: {session_id}, Material: {material_id}"
            )
        else:
            logger.warning(
                f"⚠️ Material mismatch - "
                f"Session: {session_id}, "
                f"Expected: {material_id}, Got: {session.materialId}"
            )
        
        return result
    
    except Exception as e:
        logger.error(
            f"❌ Error validating session material for {session_id}: {e}",
            exc_info=True
        )
        return {
            "valid": False,
            "error": f"Failed to validate session: {str(e)}",
            "matches": False
        }


@mcp_server.tool()
async def get_session_state(session_id: str) -> dict:
    """
    Get the current state of a quiz session including question history.
    
    Retrieves the full question history with difficulty levels to enable
    context-aware question generation and difficulty adaptation.
    
    Args:
        session_id: The quiz session ID
    
    Returns:
        Dictionary containing:
        - questions: List of question records with IDs, difficulty, and correctness
        - lastDifficulty: The difficulty of the most recent question (or null if none)
        - questionCount: Total number of questions in the session
        - success: Whether the operation succeeded
    
    Example Response:
        {
            "success": true,
            "questions": [
                {"questionId": "q_001", "difficulty": "medium", "wasCorrect": null},
                {"questionId": "q_002", "difficulty": "medium", "wasCorrect": true},
                {"questionId": "q_003", "difficulty": "hard", "wasCorrect": false}
            ],
            "lastDifficulty": "hard",
            "questionCount": 3
        }
    """
    try:
        logger.info(f"📋 Retrieving session state - Session: {session_id}")
        
        session = await session_service.get_session(session_id)
        
        if not session:
            logger.warning(f"⚠️ Session not found: {session_id}")
            return {
                "success": False,
                "error": f"Session not found: {session_id}",
                "questions": [],
                "lastDifficulty": None,
                "questionCount": 0
            }
        
        # Build question list
        questions = [
            {
                "questionId": q.questionId,
                "difficulty": q.difficulty,
                "wasCorrect": q.wasCorrect,
                "timestamp": q.timestamp.isoformat()
            }
            for q in session.questions
        ]
        
        # Get last difficulty
        last_difficulty = session.questions[-1].difficulty if session.questions else None
        
        result = {
            "success": True,
            "questions": questions,
            "lastDifficulty": last_difficulty,
            "questionCount": len(questions),
            "sessionId": session_id,
            "materialId": session.materialId,
            "status": session.status
        }
        
        logger.info(
            f"✅ Session state retrieved - "
            f"Session: {session_id}, "
            f"Questions: {len(questions)}, "
            f"Last Difficulty: {last_difficulty}"
        )
        
        return result
    
    except Exception as e:
        logger.error(
            f"❌ Error retrieving session state for {session_id}: {e}",
            exc_info=True
        )
        return {
            "success": False,
            "error": f"Failed to retrieve session state: {str(e)}",
            "questions": [],
            "lastDifficulty": None,
            "questionCount": 0
        }


# ============================================================================
# SERVER LIFECYCLE
# ============================================================================

async def main():
    """Main entry point for MCP server"""
    try:
        logger.info("🚀 Starting MCP Quiz Server...")
        
        # Initialize services
        await initialize_services()
        
        logger.info("✅ MCP Quiz Server ready")
        logger.info("📋 Available tools:")
        logger.info("   - compute_next_difficulty")
        logger.info("   - get_chunk_by_difficulty")
        logger.info("   - record_question_event (supports wasCorrect=null)")
        logger.info("   - get_session_info")
        logger.info("   - validate_session_material")
        logger.info("   - get_session_state")
        
        # Run server
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options()
            )
    
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())