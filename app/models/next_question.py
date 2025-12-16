"""
Next Question Request/Response Models
Handles first question and adaptive subsequent questions
"""
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional


class NextQuestionRequest(BaseModel):
    """Request model for next question generation"""
    sessionId: str = Field(..., description="Current quiz session ID")
    materialId: str = Field(..., description="Study material ID")
    currentDifficulty: Literal["easy", "medium", "hard"] = Field(
        ..., 
        description="Current difficulty level"
    )
    isFirst: bool = Field(
        default=False,
        description="Whether this is the first question in the session"
    )
    wasCorrect: Optional[bool] = Field(
        None, 
        description="Whether previous answer was correct (required if isFirst=False)"
    )
    previousQuestionId: Optional[str] = Field(
        None, 
        description="ID of the previous question (required if isFirst=False)"
    )
    
    @model_validator(mode='after')
    def validate_subsequent_fields(self):
        """Validate that wasCorrect and previousQuestionId are provided when isFirst=False"""
        if not self.isFirst:
            if self.wasCorrect is None:
                raise ValueError("wasCorrect is required when isFirst=False")
            if self.previousQuestionId is None:
                raise ValueError("previousQuestionId is required when isFirst=False")
        return self


class NextQuestionResponse(BaseModel):
    """
    Response model for next question
    
    SECURITY: Does NOT include the correct answer
    Answer is stored in session for evaluation during submission
    """
    questionId: str = Field(..., description="Unique question identifier")
    question: str = Field(..., description="Question text")
    options: list[str] = Field(..., description="List of 4 answer options")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., 
        description="Question difficulty level"
    )
    difficultyChanged: bool = Field(
        ..., 
        description="Whether difficulty was adjusted from previous level"
    )
    previousDifficulty: str = Field(
        ..., 
        description="Previous difficulty level"
    )
    
    class Config:
        extra = "ignore"


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: str = Field(..., description="Error code")