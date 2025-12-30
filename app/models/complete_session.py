"""
Complete Session Models
"""
from pydantic import BaseModel, Field
from typing import Literal


class CompleteSessionRequest(BaseModel):
    """Request model for completing a quiz session"""
    
    sessionId: str = Field(
        ...,
        description="Unique identifier of the quiz session to complete",
        examples=["session_abc123"]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sessionId": "session_abc123"
            }
        }


class CompleteSessionResponse(BaseModel):
    """Response model for completed quiz session with final score"""
    
    sessionId: str = Field(
        ...,
        description="The completed session ID"
    )
    
    score: int = Field(
        ...,
        ge=0,
        description="Final score (number of correct answers)"
    )
    
    totalQuestions: int = Field(
        ...,
        ge=0,
        description="Total number of questions answered"
    )
    
    correct: int = Field(
        ...,
        ge=0,
        description="Number of correct answers"
    )
    
    percentage: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage score (0-100)"
    )
    
    status: Literal["completed"] = Field(
        default="completed",
        description="Session status after completion"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sessionId": "session_abc123",
                "score": 7,
                "totalQuestions": 10,
                "correct": 7,
                "percentage": 70.0,
                "status": "completed"
            }
        }