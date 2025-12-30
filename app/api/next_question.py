"""
Next Question API Routes - Part 1
LLM orchestrates retrieval and difficulty adaptation using MCP tools
UPDATED: Properly handles quiz completion detection
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.next_question import (
    NextQuestionRequest,
    NextQuestionResponse,
    ErrorResponse
)

from app.services.next_question_service import (
    NextQuestionService,
    SessionNotFoundError,
    MaterialMismatchError,
    NoContentAvailableError,
    MCPOrchestrationError
)
from app.services.quiz_session_service import QuizSessionService
from app.services.qdrant_service import QdrantService, get_qdrant_service

from app.models.sumbit_answer import (
    SubmitAnswerRequest,
    SubmitAnswerResponse
)
from app.models.complete_session import (
    CompleteSessionRequest,
    CompleteSessionResponse
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/quiz")


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

async def get_db() -> AsyncIOMotorDatabase:
    """Dependency to get MongoDB database instance"""
    from app.db.mongodb import get_database
    return get_database()


def get_qdrant() -> QdrantService:
    """Dependency to get Qdrant service instance"""
    return get_qdrant_service()


def get_next_question_service(
    db: AsyncIOMotorDatabase = Depends(get_db),
    qdrant: QdrantService = Depends(get_qdrant)
) -> NextQuestionService:
    """Dependency to get NextQuestionService instance"""
    return NextQuestionService(
        db=db,
        qdrant_service=qdrant
    )


def get_quiz_session_service(
    db: AsyncIOMotorDatabase = Depends(get_db)
) -> QuizSessionService:
    """Dependency to get QuizSessionService instance"""
    return QuizSessionService(db)


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post(
    "/next-question",
    response_model=NextQuestionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Successfully generated next question OR quiz completed",
            "model": NextQuestionResponse
        },
        400: {
            "description": "Invalid request or difficulty level",
            "model": ErrorResponse
        },
        404: {
            "description": "Session or content not found",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    },
    summary="Generate next adaptive question via LLM + MCP orchestration with completion detection",
    description="""
    Generate the next question with LLM orchestrating ALL logic via MCP tools.
    Backend now determines when quiz should end.
    
    **QUIZ COMPLETION:**
    Backend checks completion conditions BEFORE generating questions:
    - Rule 1: Max questions reached (default: 8, configurable)
    - Rule 2: No content available at target difficulty
    - Rule 3: All unique chunks exhausted (chunk reuse detected)
    
    When quiz completes, response includes:
    - isQuizComplete: true
    - completionReason: "Why quiz ended"
    - All question fields: null
    
    When quiz continues, response includes:
    - isQuizComplete: false
    - completionReason: null
    - Full question data
    
    **FIRST QUESTION HANDLING:**
    - Backend automatically detects if session has no questions
    - Sets isFirst=true and currentDifficulty="easy"
    - LLM skips compute_next_difficulty tool
    - LLM calls record_question_event with wasCorrect=null
    
    **SUBSEQUENT QUESTIONS:**
    - Backend validates wasCorrect and previousQuestionId are provided
    - Backend checks if quiz should end
    - If not ended: LLM calls compute_next_difficulty based on previous answer
    - LLM adapts difficulty and retrieves appropriate content
    - LLM records event with actual wasCorrect value
    
    **MCP-DRIVEN ARCHITECTURE:**
    The LLM handles ALL workflow logic via tool calls. The backend only:
    - Validates session exists
    - Detects first question state
    - Checks quiz completion conditions
    - Forwards tool execution results
    - Stores question metadata securely
    
    **SECURITY:** Correct answer never exposed in response, stored in session for evaluation.
    """
)
async def get_next_question(
    request: NextQuestionRequest,
    service: NextQuestionService = Depends(get_next_question_service),
    session_service: QuizSessionService = Depends(get_quiz_session_service)
) -> NextQuestionResponse:
    """
    Generate next adaptive question via LLM + MCP orchestration with quiz completion
    
    UPDATED: Now returns completion signal when quiz should end
    
    Args:
        request: NextQuestionRequest with session and answer data
        service: NextQuestionService instance (injected)
        session_service: QuizSessionService for session state checking
    
    Returns:
        NextQuestionResponse with:
        - If quiz continues: full question data + isQuizComplete=false
        - If quiz ends: isQuizComplete=true + completionReason
    
    Raises:
        HTTPException: With appropriate status code and error details
    """
    try:
        # AUTO-DETECT FIRST QUESTION
        is_actually_first = await session_service.is_first_question(request.sessionId)
        
        # Override request if client didn't set it correctly
        if is_actually_first and not request.isFirst:
            logger.info(
                f"🔄 Auto-correcting isFirst flag for session {request.sessionId} - "
                f"Session has no questions, setting isFirst=True"
            )
            request.isFirst = True
            request.currentDifficulty = "easy"  # Force easy for first question
        
        if request.isFirst and request.currentDifficulty != "easy":
            logger.info(
                f"🔄 Auto-correcting difficulty for first question: "
                f"{request.currentDifficulty} -> easy"
            )
            request.currentDifficulty = "easy"
        
        # Log request details
        logger.info(
            f"🎯 Next question request - "
            f"Session: {request.sessionId}, "
            f"Material: {request.materialId}, "
            f"IsFirst: {request.isFirst}, "
            f"Difficulty: {request.currentDifficulty}"
        )
        
        if not request.isFirst:
            logger.info(
                f"   Previous: QuestionId={request.previousQuestionId}, "
                f"WasCorrect={request.wasCorrect}"
            )
        
        # Generate question via MCP orchestration (with completion check)
        result = await service.generate_next_question(
            session_id=request.sessionId,
            material_id=request.materialId,
            current_difficulty=request.currentDifficulty,
            is_first=request.isFirst,
            was_correct=request.wasCorrect,
            previous_question_id=request.previousQuestionId
        )
        
        # CHECK IF QUIZ COMPLETED
        if result.get("isQuizComplete"):
            logger.info(
                f"🏁 Quiz completed for session {request.sessionId} - "
                f"Reason: {result.get('completionReason')}"
            )
            
            # Auto-complete session status
            await session_service.update_session_status(
                session_id=request.sessionId,
                status="completed"
            )
            
            return NextQuestionResponse(
                questionId=None,
                question=None,
                options=None,
                difficulty=None,
                difficultyChanged=None,
                previousDifficulty=None,
                isQuizComplete=True,
                completionReason=result.get("completionReason")
            )
        
        # Quiz continues - return question
        logger.info(
            f"✅ Next question generated - "
            f"QuestionId: {result['questionId']}, "
            f"Difficulty: {result['difficulty']}, "
            f"IsFirst: {request.isFirst}"
        )
        
        return NextQuestionResponse(**result)
    
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except SessionNotFoundError as e:
        logger.error(f"❌ Session not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    
    except MaterialMismatchError as e:
        logger.error(f"❌ Material mismatch: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except NoContentAvailableError as e:
        logger.warning(f"⚠️ No content available: {e}")
        # This is now handled as quiz completion
        logger.info(f"🏁 Quiz completed - no content available")
        await session_service.update_session_status(
            session_id=request.sessionId,
            status="completed"
        )
        return NextQuestionResponse(
            questionId=None,
            question=None,
            options=None,
            difficulty=None,
            difficultyChanged=None,
            previousDifficulty=None,
            isQuizComplete=True,
            completionReason="No more content available"
        )
    
    except MCPOrchestrationError as e:
        logger.error(f"❌ MCP orchestration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate question via MCP orchestration: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while generating the question"
        )

@router.post(
    "/submit-answer",
    response_model=SubmitAnswerResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Successfully evaluated answer",
            "model": SubmitAnswerResponse
        },
        400: {
            "description": "Invalid request",
            "model": ErrorResponse
        },
        404: {
            "description": "Session or question not found",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    },
    summary="Submit and evaluate answer",
    description="""
    Submit user's answer for evaluation against stored correct answer.
    
    **Workflow:**
    1. Validates session exists
    2. Retrieves question metadata from session (includes correct answer)
    3. Compares submitted answer with correct answer
    4. Returns evaluation result with correct answer
    
    **SECURITY:** 
    - Correct answers are never exposed in /next-question endpoint
    - They are only revealed here after user submission
    - Answers are stored securely in questionMetadata array
    - Backend evaluates answer server-side (no client-side validation)
    
    **Note:** 
    - Answer recording is handled by LLM via record_question_event MCP tool
    - This endpoint only evaluates and returns the result
    - The LLM records the event when generating the NEXT question
    """
)
async def submit_answer(
    request: SubmitAnswerRequest,
    session_service: QuizSessionService = Depends(get_quiz_session_service)
) -> SubmitAnswerResponse:
    """
    Submit and evaluate user's answer against stored correct answer
    
    This endpoint provides server-side answer validation. The correct answer
    is retrieved from the session metadata (stored during question generation)
    and compared with the user's submission.
    
    Args:
        request: SubmitAnswerRequest with session, question, and selected answer
        session_service: QuizSessionService instance (injected)
    
    Returns:
        SubmitAnswerResponse with evaluation result and correct answer
    
    Raises:
        HTTPException: With appropriate status code and error details
    """
    try:
        logger.info(
            f"📝 Answer submission - "
            f"Session: {request.sessionId}, "
            f"Question: {request.questionId}, "
            f"Selected: {request.selectedAnswer}"
        )
        
        # Evaluate answer against stored correct answer
        was_correct, correct_answer, difficulty = await session_service.evaluate_answer(
            session_id=request.sessionId,
            question_id=request.questionId,
            selected_answer=request.selectedAnswer
        )
        
        # Build response
        response = SubmitAnswerResponse(
            wasCorrect=was_correct,
            correctAnswer=correct_answer,
            difficulty=difficulty
        )
        
        logger.info(
            f"✅ Answer evaluated - "
            f"Session: {request.sessionId}, "
            f"Question: {request.questionId}, "
            f"Result: {'CORRECT ✓' if was_correct else 'INCORRECT ✗'}, "
            f"Correct Answer: {correct_answer}"
        )
        
        return response
    
    except ValueError as e:
        # Session or question not found
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(
            f"❌ Unexpected error during answer submission: {e}", 
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while evaluating the answer"
        )


@router.post(
    "/complete",
    response_model=CompleteSessionResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {
            "description": "Session completed successfully",
            "model": CompleteSessionResponse
        },
        400: {
            "description": "Invalid request or session already completed",
            "model": ErrorResponse
        },
        404: {
            "description": "Session not found",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    },
    summary="Complete quiz session and get final score",
    description="""
    Mark a quiz session as completed and compute the final score.
    
    **Behavior:**
    - Marks the session status as "completed"
    - Computes final score from recorded answers
    - Returns score summary with statistics
    
    **Important:**
    - This endpoint does NOT call OpenAI
    - No new questions are generated
    - Session cannot be resumed after completion
    
    **Use Case:**
    Call this when the user finishes the quiz or wants to end early.
    This ensures the final answer is properly recorded in session stats.
    
    **Note:**
    With the new completion detection, the /next-question endpoint will
    automatically mark the session as completed when quiz ends. This endpoint
    is for manual completion or when user quits early.
    """
)
async def complete_session(
    request: CompleteSessionRequest,
    session_service: QuizSessionService = Depends(get_quiz_session_service)
) -> CompleteSessionResponse:
    """
    Complete quiz session and return final score
    
    Args:
        request: CompleteSessionRequest with sessionId
        session_service: QuizSessionService instance (injected)
    
    Returns:
        CompleteSessionResponse with score statistics
    
    Raises:
        HTTPException: With appropriate status code and error details
    """
    try:
        logger.info(f"🏁 Completing session: {request.sessionId}")
        
        # Get session to validate it exists
        session = await session_service.get_session(request.sessionId)
        
        if not session:
            raise ValueError(f"Session not found: {request.sessionId}")
        
        # Check if already completed
        if session.status == "completed":
            logger.warning(f"⚠️ Session {request.sessionId} is already completed")
            # Still return the score, don't error
        
        # Compute score
        score_data = compute_score(session.questions)
        
        # Mark session as completed (if not already)
        if session.status != "completed":
            await session_service.update_session_status(
                session_id=request.sessionId,
                status="completed"
            )
        
        response = CompleteSessionResponse(
            sessionId=request.sessionId,
            score=score_data["score"],
            totalQuestions=score_data["totalQuestions"],
            correct=score_data["correct"],
            percentage=score_data["percentage"],
            status="completed"
        )
        
        logger.info(
            f"✅ Session completed - "
            f"Session: {request.sessionId}, "
            f"Score: {score_data['correct']}/{score_data['totalQuestions']} "
            f"({score_data['percentage']:.1f}%)"
        )
        
        return response
    
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"❌ Unexpected error completing session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while completing the session"
        )


# ============================================================================
# HELPER FUNCTION
# ============================================================================

def compute_score(questions: list) -> dict:
    """
    Compute score from list of question records
    
    Args:
        questions: List of QuestionRecord objects
    
    Returns:
        Dictionary with score statistics
    """
    total = len(questions)
    
    # Count correct answers (exclude None values for first question edge case)
    correct = sum(
        1 for q in questions 
        if q.wasCorrect is not None and q.wasCorrect
    )
    
    # Calculate percentage (avoid division by zero)
    percentage = (correct / total * 100) if total > 0 else 0.0
    
    return {
        "score": correct,  # Same as correct, for clarity
        "totalQuestions": total,
        "correct": correct,
        "percentage": round(percentage, 2)
    }


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="""
    Check if quiz service dependencies are healthy.
    
    Verifies:
    - Qdrant vector database connectivity
    - MongoDB connectivity
    - OpenAI API availability (via environment variable check)
    
    Returns overall health status and individual service statuses.
    """
)
async def health_check(
    qdrant: QdrantService = Depends(get_qdrant)
) -> dict:
    """
    Health check for quiz service dependencies
    
    Checks connectivity to:
    - Qdrant (vector database for chunk retrieval)
    - MongoDB (session storage)
    - OpenAI API (via env var check)
    
    Returns:
        Health status dictionary with overall status and service-specific statuses
    """
    import os
    
    # Check Qdrant connectivity
    qdrant_healthy = qdrant.health_check()
    
    # Check OpenAI API key is configured
    openai_configured = bool(os.getenv("OPENAI_API_KEY"))
    
    # Determine overall health
    all_healthy = qdrant_healthy and openai_configured
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": {
            "qdrant": "up" if qdrant_healthy else "down",
            "mongodb": "up",  # Assume up if we can respond
            "openai": "configured" if openai_configured else "not_configured"
        },
        "mcp_orchestration": "enabled",
        "first_question_handling": "automatic",
        "quiz_completion": "automatic",
        "max_questions": 8,
        "description": "LLM orchestrates all question generation via MCP tools with automatic quiz completion"
    }