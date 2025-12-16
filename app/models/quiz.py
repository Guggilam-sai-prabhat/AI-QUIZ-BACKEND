"""
Quiz Models
Pydantic models for quiz submissions and attempts
FILE: app/models/quiz.py
"""
from pydantic import BaseModel, Field, field_validator
from typing import List
from datetime import datetime, timezone
from bson import ObjectId


class QuizSubmission(BaseModel):
    """
    Request model for submitting quiz answers
    """
    quiz_id: str = Field(
        ...,
        description="Unique identifier for the quiz",
        min_length=1,
        max_length=100
    )
    answers: List[str] = Field(
        ...,
        description="List of user-selected answer options",
        min_length=1
    )
    
    @field_validator('answers')
    @classmethod
    def validate_answers(cls, v):
        """Ensure answers list is not empty and contains valid strings"""
        if not v:
            raise ValueError("Answers list cannot be empty")
        
        # Remove empty strings
        cleaned = [a.strip() for a in v if a and a.strip()]
        
        if not cleaned:
            raise ValueError("Answers list must contain valid non-empty strings")
        
        return cleaned
    
    @field_validator('quiz_id')
    @classmethod
    def validate_quiz_id(cls, v):
        """Validate quiz_id is not empty"""
        if not v or not v.strip():
            raise ValueError("quiz_id cannot be empty")
        return v.strip()
    
    class Config:
        json_schema_extra = {
            "example": {
                "quiz_id": "material_507f1f77bcf86cd799439011_quiz",
                "answers": ["Option A", "Option C", "Option B"]
            }
        }


class QuizAttemptResponse(BaseModel):
    """
    Response model for quiz attempt results
    """
    id: str = Field(
        ...,
        description="MongoDB ObjectId of the attempt",
        alias="_id"
    )
    quiz_id: str = Field(
        ...,
        description="Quiz identifier"
    )
    user_answers: List[str] = Field(
        ...,
        description="Answers submitted by the user"
    )
    correct_answers: List[str] = Field(
        ...,
        description="Correct answers for the quiz"
    )
    score: int = Field(
        ...,
        description="Number of correct answers",
        ge=0
    )
    total: int = Field(
        ...,
        description="Total number of questions",
        gt=0
    )
    percentage: float = Field(
        ...,
        description="Score as percentage (0-100)",
        ge=0.0,
        le=100.0
    )
    submitted_at: datetime = Field(
        ...,
        description="Timestamp when quiz was submitted"
    )
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "_id": "507f1f77bcf86cd799439011",
                "quiz_id": "material_123_quiz",
                "user_answers": ["Option A", "Option C", "Option B"],
                "correct_answers": ["Option A", "Option B", "Option B"],
                "score": 2,
                "total": 3,
                "percentage": 66.67,
                "submitted_at": "2025-12-08T10:30:00Z"
            }
        }


class QuizAttemptSummary(BaseModel):
    """
    Lightweight summary of a quiz attempt
    """
    id: str = Field(..., alias="_id")
    quiz_id: str
    score: int
    total: int
    percentage: float
    submitted_at: datetime
    
    class Config:
        populate_by_name = True


class QuizAttemptDocument(BaseModel):
    """
    MongoDB document model for quiz attempts
    Internal use - matches database schema exactly
    """
    quiz_id: str
    user_answers: List[str]
    correct_answers: List[str]
    score: int
    total: int
    percentage: float
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        json_schema_extra = {
            "example": {
                "quiz_id": "material_123_quiz",
                "user_answers": ["A", "C", "B"],
                "correct_answers": ["A", "B", "B"],
                "score": 2,
                "total": 3,
                "percentage": 66.67,
                "submitted_at": "2025-12-08T10:30:00.000Z"
            }
        }