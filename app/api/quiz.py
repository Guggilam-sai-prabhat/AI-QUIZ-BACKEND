"""
Quiz API Routes
FastAPI endpoints for quiz generation, submission, and management
"""
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from app.services.quiz_service import (
    get_quiz_service,
    QuizGenerationError,
    MaterialNotFoundError,
    ChunkRetrievalError,
    QuizNotFoundError,
    QuizValidationError
)
from app.models.quiz import (
    QuizSubmission,
    QuizAttemptResponse,
    QuizAttemptSummary
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz", tags=["Quiz"])


# ==================== GENERATION REQUEST/RESPONSE MODELS ====================

class GenerateQuizRequest(BaseModel):
    """Request model for quiz generation"""
    material_id: str = Field(..., description="MongoDB document ID of the material")
    num_questions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of questions to generate (1-20)"
    )
    top_k_chunks: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of chunks to use for context"
    )
    llm_provider: str = Field(
        default="grok",
        description="LLM provider: grok, openai, or anthropic"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "material_id": "507f1f77bcf86cd799439011",
                "num_questions": 5,
                "top_k_chunks": 10,
                "llm_provider": "grok"
            }
        }


class GenerateQuizResponse(BaseModel):
    """Response model for quiz generation"""
    quiz_id: str
    question_count: int
    status: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "quiz_id": "550e8400-e29b-41d4-a716-446655440000",
                "question_count": 5,
                "status": "success"
            }
        }


class QuestionModel(BaseModel):
    """Model for a single quiz question"""
    question: str
    options: List[str]
    answer: str


class QuizDetailResponse(BaseModel):
    """Response model for quiz details"""
    quiz_id: str
    material_id: str
    questions: List[QuestionModel]
    question_count: int
    llm_provider: str
    created_at: str


class QuizListItem(BaseModel):
    """Model for quiz list items"""
    quiz_id: str
    material_id: str
    question_count: int
    created_at: str


# ==================== QUIZ GENERATION ENDPOINTS ====================

@router.post(
    "/generate",
    response_model=GenerateQuizResponse,
    summary="Generate Quiz",
    description="Generate a quiz from material content using AI"
)
async def generate_quiz(request: GenerateQuizRequest):
    """
    Generate a quiz from material content
    
    Workflow:
    1. Retrieve top-K chunks from the material
    2. Format chunks into context
    3. Build quiz generation prompt
    4. Call LLM to generate questions
    5. Parse and validate the response
    6. Save quiz to database
    
    Returns:
        Quiz ID, question count, and status
    """
    try:
        quiz_service = get_quiz_service()
        
        result = await quiz_service.generate_quiz(
            material_id=request.material_id,
            num_questions=request.num_questions,
            top_k_chunks=request.top_k_chunks,
            llm_provider=request.llm_provider
        )
        
        return GenerateQuizResponse(**result)
        
    except MaterialNotFoundError as e:
        logger.warning(f"⚠️ Material not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except ChunkRetrievalError as e:
        logger.error(f"❌ Chunk retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except QuizGenerationError as e:
        logger.error(f"❌ Quiz generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate quiz: {str(e)}"
        )


@router.get(
    "/all",
    response_model=List[QuizListItem],
    summary="Get All Quizzes",
    description="Retrieve all quizzes across all materials with pagination"
)
async def get_all_quizzes(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of quizzes to return")
):
    """
    Get all quizzes in the system
    
    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of quizzes to return
    
    Returns:
        List of all quizzes sorted by creation date (newest first)
        
    Example response:
        [
            {
                "quiz_id": "550e8400-e29b-41d4-a716-446655440000",
                "material_id": "507f1f77bcf86cd799439011",
                "question_count": 5,
                "created_at": "2025-12-08T10:30:00Z"
            },
            ...
        ]
    """
    try:
        quiz_service = get_quiz_service()
        quizzes = await quiz_service.get_all_quizzes(skip, limit)
        
        return [
            QuizListItem(
                quiz_id=q["quiz_id"],
                material_id=q["material_id"],
                question_count=q["question_count"],
                created_at=q["created_at"].isoformat()
            )
            for q in quizzes
        ]
        
    except Exception as e:
        logger.error(f"❌ Failed to get all quizzes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve quizzes: {str(e)}"
        )


@router.get(
    "/{quiz_id}",
    response_model=QuizDetailResponse,
    summary="Get Quiz",
    description="Retrieve a quiz by its ID"
)
async def get_quiz(quiz_id: str):
    """
    Get quiz details by ID
    
    Args:
        quiz_id: Quiz UUID
    
    Returns:
        Full quiz details including questions
    """
    try:
        quiz_service = get_quiz_service()
        quiz = await quiz_service.get_quiz(quiz_id)
        
        if not quiz:
            raise HTTPException(
                status_code=404,
                detail=f"Quiz not found: {quiz_id}"
            )
        
        return QuizDetailResponse(
            quiz_id=quiz["quiz_id"],
            material_id=quiz["material_id"],
            questions=quiz["questions"],
            question_count=quiz["question_count"],
            llm_provider=quiz.get("llm_provider", "unknown"),
            created_at=quiz["created_at"].isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get quiz: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve quiz: {str(e)}"
        )


@router.get(
    "/material/{material_id}",
    response_model=List[QuizListItem],
    summary="Get Quizzes by Material",
    description="Get all quizzes generated from a specific material"
)
async def get_quizzes_by_material(
    material_id: str,
    limit: int = Query(default=10, ge=1, le=50)
):
    """
    Get all quizzes for a material
    
    Args:
        material_id: MongoDB document ID
        limit: Maximum number of quizzes to return
    
    Returns:
        List of quizzes
    """
    try:
        quiz_service = get_quiz_service()
        quizzes = await quiz_service.get_quizzes_by_material(material_id, limit)
        
        return [
            QuizListItem(
                quiz_id=q["quiz_id"],
                material_id=q["material_id"],
                question_count=q["question_count"],
                created_at=q["created_at"].isoformat()
            )
            for q in quizzes
        ]
        
    except Exception as e:
        logger.error(f"❌ Failed to get quizzes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve quizzes: {str(e)}"
        )


@router.get(
    "/all",
    response_model=List[QuizListItem],
    summary="Get All Quizzes",
    description="Retrieve all quizzes across all materials with pagination"
)
async def get_all_quizzes(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of quizzes to return")
):
    """
    Get all quizzes in the system
    
    Args:
        skip: Number of records to skip (for pagination)
        limit: Maximum number of quizzes to return
    
    Returns:
        List of all quizzes sorted by creation date (newest first)
        
    Example response:
        [
            {
                "quiz_id": "550e8400-e29b-41d4-a716-446655440000",
                "material_id": "507f1f77bcf86cd799439011",
                "question_count": 5,
                "created_at": "2025-12-08T10:30:00Z"
            },
            ...
        ]
    """
    try:
        quiz_service = get_quiz_service()
        quizzes = await quiz_service.get_all_quizzes(skip, limit)
        
        return [
            QuizListItem(
                quiz_id=q["quiz_id"],
                material_id=q["material_id"],
                question_count=q["question_count"],
                created_at=q["created_at"].isoformat()
            )
            for q in quizzes
        ]
        
    except Exception as e:
        logger.error(f"❌ Failed to get all quizzes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve quizzes: {str(e)}"
        )


@router.delete(
    "/{quiz_id}",
    summary="Delete Quiz",
    description="Delete a quiz by its ID"
)
async def delete_quiz(quiz_id: str):
    """
    Delete a quiz by ID
    
    Args:
        quiz_id: Quiz UUID
    
    Returns:
        Deletion confirmation
    """
    try:
        quiz_service = get_quiz_service()
        deleted = await quiz_service.delete_quiz(quiz_id)
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Quiz not found: {quiz_id}"
            )
        
        return {
            "quiz_id": quiz_id,
            "status": "deleted"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete quiz: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete quiz: {str(e)}"
        )


# ==================== QUIZ SUBMISSION ENDPOINTS ====================

@router.post(
    "/submit-quiz",
    response_model=QuizAttemptResponse,
    summary="Submit Quiz Answers",
    description="""
    Submit answers for a quiz and receive immediate grading results.
    
    ## Workflow:
    1. Validates quiz exists in the system
    2. Compares user answers with correct answers
    3. Calculates score and percentage
    4. Saves attempt to MongoDB
    5. Returns detailed results
    
    ## Response includes:
    - Score (number correct)
    - Total questions
    - Percentage
    - Your answers vs correct answers
    - Submission timestamp
    """
)
async def submit_quiz(submission: QuizSubmission):
    """
    Submit quiz answers for grading
    
    Args:
        submission: Quiz submission with quiz_id and answers
    
    Returns:
        QuizAttemptResponse with grading results
    """
    try:
        logger.info(
            f"📝 Quiz submission received: quiz_id={submission.quiz_id}, "
            f"answers_count={len(submission.answers)}"
        )
        
        quiz_service = get_quiz_service()
        result = await quiz_service.submit_quiz_attempt(
            quiz_id=submission.quiz_id,
            user_answers=submission.answers
        )
        
        logger.info(
            f"✅ Quiz graded: attempt_id={result['id']}, "
            f"score={result['score']}/{result['total']} ({result['percentage']}%)"
        )
        
        return QuizAttemptResponse(
            _id=result["id"],
            quiz_id=result["quiz_id"],
            user_answers=result["user_answers"],
            correct_answers=result["correct_answers"],
            score=result["score"],
            total=result["total"],
            percentage=result["percentage"],
            submitted_at=result["submitted_at"]
        )
        
    except QuizNotFoundError as e:
        logger.warning(f"⚠️ Quiz not found: {submission.quiz_id}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except QuizValidationError as e:
        logger.warning(f"⚠️ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"❌ Quiz submission failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process quiz submission. Please try again."
        )


@router.get(
    "/attempts/{attempt_id}",
    response_model=QuizAttemptResponse,
    summary="Get Quiz Attempt Details",
    description="Retrieve detailed information about a specific quiz attempt"
)
async def get_quiz_attempt(attempt_id: str):
    """Get details of a specific quiz attempt"""
    try:
        quiz_service = get_quiz_service()
        result = await quiz_service.get_attempt_by_id(attempt_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Quiz attempt '{attempt_id}' not found"
            )
        
        return QuizAttemptResponse(
            _id=result["_id"],
            quiz_id=result["quiz_id"],
            user_answers=result["user_answers"],
            correct_answers=result["correct_answers"],
            score=result["score"],
            total=result["total"],
            percentage=result["percentage"],
            submitted_at=result["submitted_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get quiz attempt: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve quiz attempt"
        )


@router.get(
    "/{quiz_id}/attempts",
    response_model=List[QuizAttemptSummary],
    summary="Get All Attempts for a Quiz",
    description="Retrieve all submission attempts for a specific quiz"
)
async def get_quiz_attempts(
    quiz_id: str,
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum records to return")
):
    """Get all attempts for a specific quiz"""
    try:
        quiz_service = get_quiz_service()
        attempts = await quiz_service.get_attempts_for_quiz(quiz_id, skip, limit)
        
        return [
            QuizAttemptSummary(
                _id=attempt["_id"],
                quiz_id=attempt["quiz_id"],
                score=attempt["score"],
                total=attempt["total"],
                percentage=attempt["percentage"],
                submitted_at=attempt["submitted_at"]
            )
            for attempt in attempts
        ]
        
    except Exception as e:
        logger.error(f"❌ Failed to get quiz attempts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve quiz attempts"
        )


@router.get(
    "/{quiz_id}/statistics",
    summary="Get Quiz Statistics",
    description="Get aggregated statistics for a quiz"
)
async def get_quiz_statistics(quiz_id: str):
    """Get statistical summary for a quiz"""
    try:
        quiz_service = get_quiz_service()
        stats = await quiz_service.get_quiz_stats(quiz_id)
        
        return {
            "quiz_id": quiz_id,
            "statistics": stats,
            "message": "Statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to get quiz statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve quiz statistics"
        )


# ==================== ATTEMPTS LISTING ENDPOINTS ====================

@router.get(
    "/attempts/{quiz_id}",
    summary="Get All Attempts for a Quiz",
    description="Retrieve full list of all submission attempts for a specific quiz"
)
async def get_all_attempts(quiz_id: str):
    """
    Get all attempts for a specific quiz
    
    Args:
        quiz_id: Quiz identifier (UUID)
    
    Returns:
        List of attempts with score, total, and submission timestamp
        
    Example response:
        [
            {"score": 3, "total": 5, "submitted_at": "2025-12-08T10:30:00Z"},
            {"score": 4, "total": 5, "submitted_at": "2025-12-07T15:20:00Z"}
        ]
    """
    try:
        quiz_service = get_quiz_service()
        attempts = await quiz_service.get_all_attempts_for_quiz(quiz_id)
        
        # Format response
        return [
            {
                "score": attempt["score"],
                "total": attempt["total"],
                "submitted_at": attempt["submitted_at"].isoformat()
            }
            for attempt in attempts
        ]
        
    except Exception as e:
        logger.error(f"❌ Failed to get attempts for quiz {quiz_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve quiz attempts"
        )


@router.get(
    "/attempts/recent/{limit}",
    summary="Get Recent Attempts",
    description="Retrieve recent quiz attempts across all quizzes, sorted by submission date (newest first)"
)
async def get_recent_attempts(
    limit: int = Path(..., ge=1, le=100, description="Maximum number of attempts to return (1-100)")
):
    """
    Get recent quiz attempts across all quizzes
    
    Args:
        limit: Maximum number of attempts to return
    
    Returns:
        List of recent attempts sorted by date (newest first)
        
    Example response:
        [
            {"score": 5, "total": 5, "submitted_at": "2025-12-08T10:30:00Z"},
            {"score": 3, "total": 5, "submitted_at": "2025-12-08T09:15:00Z"},
            {"score": 4, "total": 5, "submitted_at": "2025-12-07T15:20:00Z"}
        ]
    """
    try:
        quiz_service = get_quiz_service()
        attempts = await quiz_service.get_recent_attempts(limit)
        
        # Format response
        return [
            {
                "score": attempt["score"],
                "total": attempt["total"],
                "submitted_at": attempt["submitted_at"].isoformat()
            }
            for attempt in attempts
        ]
        
    except Exception as e:
        logger.error(f"❌ Failed to get recent attempts: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve recent attempts"
        )


