"""
Quiz API Routes
FastAPI endpoints for quiz generation and management
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from app.services.quiz_service import (
    get_quiz_service,
    QuizGenerationError,
    MaterialNotFoundError,
    ChunkRetrievalError
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quiz", tags=["Quiz"])


# Request/Response Models

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


# Endpoints

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


@router.post(
    "/generate-quiz",
    response_model=GenerateQuizResponse,
    summary="Generate Quiz (Alternative)",
    description="Alternative endpoint matching the original spec",
    deprecated=True
)
async def generate_quiz_alt(request: GenerateQuizRequest):
    """
    Alternative endpoint for backward compatibility
    Redirects to /quiz/generate
    """
    return await generate_quiz(request)