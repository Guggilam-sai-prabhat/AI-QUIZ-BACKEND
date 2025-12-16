"""
Start Session Request/Response Models
"""
from pydantic import BaseModel, Field
from typing import Literal


class StartSessionRequest(BaseModel):
    """Request model for starting a new quiz session"""
    materialId: str = Field(..., description="ID of the study material")
    
    class Config:
        json_schema_extra = {
            "example": {
                "materialId": "material_67890"
            }
        }


class StartSessionResponse(BaseModel):
    """Response model for started quiz session"""
    sessionId: str = Field(..., description="Unique session identifier")
    materialId: str = Field(..., description="ID of the study material")
    difficulty: Literal["easy", "medium", "hard"] = Field(
        default="medium",
        description="Initial difficulty level"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "sessionId": "550e8400-e29b-41d4-a716-446655440000",
                "materialId": "material_67890",
                "difficulty": "medium"
            }
        }
