from pydantic import BaseModel, Field, field_validator
from typing import Literal


class SubmitAnswerRequest(BaseModel):
    """Request model for answer submission"""
    sessionId: str = Field(..., description="Quiz session ID")
    questionId: str = Field(..., description="Question ID being answered")
    selectedAnswer: str = Field(..., description="User's selected answer (A, B, C, or D)")
    
    @field_validator('selectedAnswer')
    @classmethod
    def validate_answer_format(cls, v):
        """Validate that answer is A, B, C, or D"""
        v = v.upper().strip()
        if v not in ['A', 'B', 'C', 'D']:
            raise ValueError("selectedAnswer must be 'A', 'B', 'C', or 'D'")
        return v


class SubmitAnswerResponse(BaseModel):
    """Response model for answer evaluation"""
    wasCorrect: bool = Field(..., description="Whether the answer was correct")
    correctAnswer: str = Field(..., description="The correct answer (A, B, C, or D)")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., 
        description="Difficulty level of the answered question"
    )