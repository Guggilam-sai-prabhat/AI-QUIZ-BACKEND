"""
Quiz Session Models
Updated to store correct answers in question records for later evaluation
"""
from datetime import datetime
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
import uuid


class QuestionMetadata(BaseModel):
    """
    Complete question metadata stored in session
    SECURITY: correctAnswer is stored here but NEVER exposed in /next-question
    """
    questionId: str = Field(..., description="Unique question identifier")
    question: str = Field(..., description="Question text")
    options: List[str] = Field(..., description="List of 4 answer options")
    correctAnswer: str = Field(..., description="Correct answer (A, B, C, or D) - PRIVATE")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Question difficulty")
    
    class Config:
        json_schema_extra = {
            "example": {
                "questionId": "q_12345",
                "question": "What is the time complexity of binary search?",
                "options": ["O(n)", "O(log n)", "O(n^2)", "O(1)"],
                "correctAnswer": "B",
                "difficulty": "medium"
            }
        }


class QuestionRecord(BaseModel):
    """
    Question record with user's answer result and stored correct answer
    This is appended after answer submission
    """
    questionId: str = Field(..., description="Question ID reference")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="Question difficulty")
    correctAnswer: str = Field(..., description="Correct answer (A, B, C, or D) - stored for evaluation")
    wasCorrect: bool = Field(..., description="Whether the answer was correct")
    answeredAt: datetime = Field(default_factory=datetime.utcnow, description="When answered")
    
    class Config:
        json_schema_extra = {
            "example": {
                "questionId": "q_12345",
                "difficulty": "medium",
                "correctAnswer": "B",
                "wasCorrect": True,
                "answeredAt": "2024-01-15T10:30:00Z"
            }
        }


class QuizSession(BaseModel):
    """Quiz session with question metadata and answer records"""
    sessionId: str = Field(
        default_factory=lambda: f"session_{uuid.uuid4().hex[:12]}",
        description="Unique session identifier"
    )
    materialId: str = Field(..., description="Study material ID")
    status: Literal["ongoing", "completed", "abandoned"] = Field(
        default="ongoing",
        description="Session status"
    )
    
    # Store complete question metadata (including correct answers)
    questionMetadata: List[QuestionMetadata] = Field(
        default_factory=list,
        description="Complete question data with correct answers (PRIVATE)"
    )
    
    # Store user's answer records (now includes correctAnswer)
    questions: List[QuestionRecord] = Field(
        default_factory=list,
        description="User's answer records with correct answers stored"
    )
    
    createdAt: datetime = Field(
        default_factory=datetime.utcnow,
        description="Session creation timestamp"
    )
    updatedAt: datetime = Field(
        default_factory=datetime.utcnow,
        description="Session last update timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sessionId": "session_abc123",
                "materialId": "material_456",
                "status": "ongoing",
                "questionMetadata": [
                    {
                        "questionId": "q_1",
                        "question": "What is 2+2?",
                        "options": ["3", "4", "5", "6"],
                        "correctAnswer": "B",
                        "difficulty": "easy"
                    }
                ],
                "questions": [
                    {
                        "questionId": "q_1",
                        "difficulty": "easy",
                        "correctAnswer": "B",
                        "wasCorrect": True,
                        "answeredAt": "2024-01-15T10:30:00Z"
                    }
                ],
                "createdAt": "2024-01-15T10:00:00Z",
                "updatedAt": "2024-01-15T10:30:00Z"
            }
        }


class QuizSessionResponse(BaseModel):
    """Public session statistics (no correct answers exposed)"""
    sessionId: str = Field(..., description="Session ID")
    materialId: str = Field(..., description="Material ID")
    questionsCount: int = Field(..., description="Total questions answered")
    correctCount: int = Field(..., description="Number of correct answers")
    status: Literal["ongoing", "completed", "abandoned"] = Field(..., description="Session status")
    createdAt: datetime = Field(..., description="Creation timestamp")
    updatedAt: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sessionId": "session_abc123",
                "materialId": "material_456",
                "questionsCount": 5,
                "correctCount": 4,
                "status": "ongoing",
                "createdAt": "2024-01-15T10:00:00Z",
                "updatedAt": "2024-01-15T10:30:00Z"
            }
        }