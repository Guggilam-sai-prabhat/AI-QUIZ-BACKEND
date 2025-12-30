"""
Next Question Request/Response Models
Handles first question and adaptive subsequent questions with quiz completion
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
    
    NEW: Includes isQuizComplete to signal when backend determines quiz should end
    """
    questionId: Optional[str] = Field(None, description="Unique question identifier (None if quiz complete)")
    question: Optional[str] = Field(None, description="Question text (None if quiz complete)")
    options: Optional[list[str]] = Field(None, description="List of 4 answer options (None if quiz complete)")
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(
        None, 
        description="Question difficulty level (None if quiz complete)"
    )
    difficultyChanged: Optional[bool] = Field(
        None, 
        description="Whether difficulty was adjusted from previous level (None if quiz complete)"
    )
    previousDifficulty: Optional[str] = Field(
        None, 
        description="Previous difficulty level (None if quiz complete)"
    )
    isQuizComplete: bool = Field(
        ..., 
        description="Whether the quiz should end (True = no more questions, False = continue)"
    )
    completionReason: Optional[str] = Field(
        None,
        description="Reason why quiz ended (if isQuizComplete=True)"
    )
    
    class Config:
        extra = "ignore"


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    code: str = Field(..., description="Error code")