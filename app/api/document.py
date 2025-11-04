from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import shutil
from datetime import datetime
from app.models.document import DocumentMetadata, DocumentResponse
from app.db.mongodb import get_collection

router = APIRouter()

# Temporary upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document
    - Saves file temporarily
    - Stores metadata in MongoDB
    """
    # Validate PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Generate unique filename
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Create metadata
        document = DocumentMetadata(
            filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            content_type=file.content_type or "application/pdf"
        )
        
        # Store in MongoDB
        collection = get_collection("documents")
        result = await collection.insert_one(document.model_dump(by_alias=True, exclude=["id"]))
        
        # Return response
        return DocumentResponse(
            id=str(result.inserted_id),
            filename=document.filename,
            file_size=document.file_size,
            content_type=document.content_type,
            uploaded_at=document.uploaded_at
        )
        
    except Exception as e:
        # Clean up file if something went wrong
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/materials", response_model=List[DocumentResponse])
async def get_materials():
    """
    Get all document materials from MongoDB
    Returns list of document metadata
    """
    try:
        collection = get_collection("documents")
        documents = []
        
        async for doc in collection.find().sort("uploaded_at", -1):
            documents.append(
                DocumentResponse(
                    id=str(doc["_id"]),
                    filename=doc["filename"],
                    file_size=doc["file_size"],
                    content_type=doc["content_type"],
                    uploaded_at=doc["uploaded_at"]
                )
            )
        
        return documents
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch materials: {str(e)}")