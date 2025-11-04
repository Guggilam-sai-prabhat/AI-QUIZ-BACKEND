from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string"}

class DocumentMetadata(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    filename: str
    file_path: str
    file_size: int
    content_type: str
    file_hash: str
    extracted_text: Optional[str] = None  # Stored in DB but not returned in API
    page_count: Optional[int] = None  # Stored in DB but not returned in API
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }

class DocumentResponse(BaseModel):
    id: str
    filename: str
    file_size: int
    content_type: str
    file_hash: str
    uploaded_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }