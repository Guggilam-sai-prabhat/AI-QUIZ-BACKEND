from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List, Optional
import os
import shutil
import secrets
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import logging
import fitz  # ✅ PyMuPDF
from app.models.document import DocumentMetadata, DocumentResponse
from app.db.mongodb import get_collection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf"}
ALLOWED_MIME_TYPES = {"application/pdf"}

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)


def validate_file_type(contents: bytes, filename: str) -> bool:
    """
    Validate file is actually a PDF by checking magic bytes
    PDF files start with %PDF-
    """
    if not contents.startswith(b'%PDF-'):
        return False
    return True


def generate_safe_filename(original_filename: str) -> str:
    """
    Generate a safe, unique filename
    Prevents path traversal attacks
    """
    # Extract only the base filename (no path components)
    safe_name = os.path.basename(original_filename)
    
    # Get file extension
    _, ext = os.path.splitext(safe_name)
    
    # Generate timestamp and random token
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    random_token = secrets.token_hex(8)
    
    # Construct safe filename
    safe_filename = f"{timestamp}_{random_token}{ext}"
    
    return safe_filename


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file for deduplication"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def extract_text_from_pdf(file_path: str) -> tuple[str, int]:
    """
    Extract text from PDF using PyMuPDF (fitz)
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        Tuple of (extracted_text, page_count)
    
    Raises:
        ValueError: If PDF is empty or invalid
        Exception: For other PDF processing errors
    """
    try:
        doc = fitz.open(file_path)
        
        # Check if PDF has pages
        if len(doc) == 0:
            doc.close()
            raise ValueError("PDF file is empty (no pages)")
        
        page_count = len(doc)
        extracted_text = ""
        
        # Extract text from each page
        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()
            extracted_text += text
        
        doc.close()
        
        # Check if any text was extracted
        if not extracted_text.strip():
            logger.warning(f"No text extracted from PDF: {file_path}")
            # Don't raise error - some PDFs are image-only
        
        logger.info(f"📄 Extracted {len(extracted_text)} characters from {page_count} pages")
        
        return extracted_text.strip(), page_count
        
    except fitz.FileDataError as e:
        logger.error(f"Invalid PDF file structure: {str(e)}")
        raise ValueError(f"Invalid PDF file: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {str(e)}")
        raise Exception(f"PDF processing error: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint
    Checks MongoDB connection and upload directory
    """
    health_status = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {}
    }
    
    # Check MongoDB connection
    try:
        collection = get_collection("documents")
        await collection.find_one()
        health_status["checks"]["mongodb"] = "connected"
        logger.info("MongoDB health check: OK")
    except Exception as e:
        health_status["checks"]["mongodb"] = "disconnected"
        health_status["status"] = "degraded"
        logger.error(f"MongoDB health check failed: {str(e)}")
    
    # Check upload directory
    try:
        is_writable = os.access(UPLOAD_DIR, os.W_OK)
        health_status["checks"]["upload_dir"] = "writable" if is_writable else "not_writable"
        if not is_writable:
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["upload_dir"] = "error"
        health_status["status"] = "degraded"
        logger.error(f"Upload directory check failed: {str(e)}")
    
    return health_status


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document with text extraction and duplicate detection"""
    file_path = None
    
    try:
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"Invalid file extension attempted: {file_ext}")
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Read file contents
        contents = await file.read()
        
        # Validate file size
        if len(contents) > MAX_FILE_SIZE:
            logger.warning(f"File too large: {len(contents)} bytes")
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Validate file content
        if not validate_file_type(contents, file.filename):
            logger.warning(f"Invalid PDF content for file: {file.filename}")
            raise HTTPException(status_code=400, detail="File does not appear to be a valid PDF")
        
        # Generate safe filename
        safe_filename = generate_safe_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        logger.info(f"✅ File saved: {file_path}")
        
        # Calculate file hash
        file_hash = calculate_file_hash(file_path)
        logger.info(f"📊 File hash calculated: {file_hash}")
        
        # Check for duplicate
        collection = get_collection("documents")
        existing_doc = await collection.find_one({"file_hash": file_hash})
        
        if existing_doc:
            logger.info(f"🔴 DUPLICATE DETECTED! Hash: {file_hash}")
            logger.info(f"   Existing document ID: {existing_doc['_id']}")
            logger.info(f"   Existing filename: {existing_doc['filename']}")
            
            # Remove the newly uploaded file
            os.remove(file_path)
            logger.info(f"🗑️  Removed duplicate file: {file_path}")
            
            # Return the existing document
            return DocumentResponse(
                id=str(existing_doc["_id"]),
                filename=existing_doc["filename"],
                file_size=existing_doc["file_size"],
                content_type=existing_doc["content_type"],
                file_hash=existing_doc["file_hash"],
                extracted_text=existing_doc.get("extracted_text"),
                page_count=existing_doc.get("page_count"),
                uploaded_at=existing_doc["uploaded_at"]
            )
        
        logger.info(f"✅ No duplicate found. Proceeding with new upload.")
        
        # ✅ EXTRACT TEXT FROM PDF
        try:
            extracted_text, page_count = extract_text_from_pdf(file_path)
            logger.info(f"✅ Text extraction successful: {page_count} pages, {len(extracted_text)} chars")
        except ValueError as ve:
            # Handle empty or invalid PDFs
            logger.error(f"❌ PDF validation failed: {str(ve)}")
            # Clean up file
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # Handle other PDF processing errors
            logger.error(f"❌ PDF processing failed: {str(e)}")
            # Clean up file
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(
                status_code=422,
                detail=f"Failed to process PDF: {str(e)}"
            )
        
        # Get file size
        file_size = len(contents)
        
        # Create metadata with extracted text
        document = DocumentMetadata(
            filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            content_type=file.content_type or "application/pdf",
            file_hash=file_hash,
            extracted_text=extracted_text,  # ✅ NEW
            page_count=page_count  # ✅ NEW
        )
        
        # Store in MongoDB
        result = await collection.insert_one(
            document.model_dump(by_alias=True, exclude=["id"])
        )
        
        logger.info(f"✅ New document inserted with ID: {result.inserted_id}")
        
        # Return response
        return DocumentResponse(
            id=str(result.inserted_id),
            filename=document.filename,
            file_size=document.file_size,
            content_type=document.content_type,
            file_hash=document.file_hash,
            uploaded_at=document.uploaded_at
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file after error: {file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup file: {cleanup_error}")
        
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during file upload. Please try again."
        )


@router.get("/debug/document/{document_id}")
async def debug_document(document_id: str):
    """Debug endpoint to see raw document data"""
    from bson import ObjectId
    
    try:
        collection = get_collection("documents")
        doc = await collection.find_one({"_id": ObjectId(document_id)})
        
        if not doc:
            return {"error": "Document not found"}
        
        # Convert ObjectId to string for JSON serialization
        doc["_id"] = str(doc["_id"])
        
        return doc
    except Exception as e:
        return {"error": str(e)}


@router.get("/materials", response_model=List[DocumentResponse])
async def get_materials(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of records to return")
):
    """
    Get all document materials from MongoDB with pagination
    
    Args:
        skip: Number of documents to skip (for pagination)
        limit: Maximum number of documents to return (max 100)
    
    Returns:
        List of document metadata sorted by upload date (newest first)
    """
    try:
        collection = get_collection("documents")
        documents = []
        
        # Query with pagination
        cursor = collection.find().sort("uploaded_at", -1).skip(skip).limit(limit)
        
        async for doc in cursor:
            documents.append(
                DocumentResponse(
                    id=str(doc["_id"]),
                    filename=doc["filename"],
                    file_size=doc["file_size"],
                    content_type=doc["content_type"],
                    file_hash=doc.get("file_hash", "unknown"),
                    extracted_text=doc.get("extracted_text"),  # ✅ NEW
                    page_count=doc.get("page_count"),  # ✅ NEW
                    uploaded_at=doc["uploaded_at"]
                )
            )
        
        logger.info(f"Retrieved {len(documents)} documents (skip={skip}, limit={limit})")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to fetch materials: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve documents. Please try again."
        )


@router.get("/materials/{document_id}", response_model=DocumentResponse)
async def get_material(document_id: str):
    """
    Get a specific document by ID
    
    Args:
        document_id: MongoDB ObjectId of the document
    
    Returns:
        Document metadata with extracted text
    """
    try:
        from bson import ObjectId
        
        collection = get_collection("documents")
        doc = await collection.find_one({"_id": ObjectId(document_id)})
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentResponse(
            id=str(doc["_id"]),
            filename=doc["filename"],
            file_size=doc["file_size"],
            content_type=doc["content_type"],
            file_hash=doc.get("file_hash", "unknown"),
            extracted_text=doc.get("extracted_text"),  # ✅ NEW
            page_count=doc.get("page_count"),  # ✅ NEW
            uploaded_at=doc["uploaded_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch document {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve document"
        )


@router.delete("/materials/{document_id}")
async def delete_material(document_id: str):
    """
    Delete a document by ID
    
    Args:
        document_id: MongoDB ObjectId of the document
    
    Returns:
        Success message
    """
    try:
        from bson import ObjectId
        
        collection = get_collection("documents")
        
        # Find document first to get file path
        doc = await collection.find_one({"_id": ObjectId(document_id)})
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file from disk
        file_path = doc.get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete file: {str(e)}")
        
        # Delete from database
        result = await collection.delete_one({"_id": ObjectId(document_id)})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Document deleted: {document_id}")
        
        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document"
        )


@router.get("/materials/count")
async def get_materials_count():
    """
    Get total count of documents
    
    Returns:
        Total number of documents in the collection
    """
    try:
        collection = get_collection("documents")
        count = await collection.count_documents({})
        
        return {"total": count}
        
    except Exception as e:
        logger.error(f"Failed to count documents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to count documents"
        )


@router.post("/admin/migrate-hashes")
async def migrate_file_hashes():
    """
    Migration endpoint: Add file_hash to existing documents that don't have it
    Run this once to update old documents
    """
    collection = get_collection("documents")
    
    # Find documents without file_hash
    cursor = collection.find({"file_hash": {"$exists": False}})
    
    updated_count = 0
    failed_count = 0
    results = []
    
    async for doc in cursor:
        file_path = doc.get("file_path")
        doc_id = str(doc["_id"])
        
        if file_path and os.path.exists(file_path):
            try:
                # Calculate hash
                file_hash = calculate_file_hash(file_path)
                
                # Update document
                await collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"file_hash": file_hash}}
                )
                
                updated_count += 1
                logger.info(f"✅ Added hash to {doc_id}: {file_hash}")
                results.append({
                    "id": doc_id,
                    "status": "success",
                    "hash": file_hash
                })
                
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Failed to hash {file_path}: {str(e)}")
                results.append({
                    "id": doc_id,
                    "status": "failed",
                    "error": str(e)
                })
        else:
            failed_count += 1
            logger.warning(f"⚠️ File not found: {file_path}")
            results.append({
                "id": doc_id,
                "status": "file_not_found",
                "file_path": file_path
            })
    
    return {
        "message": "Migration complete",
        "updated": updated_count,
        "failed": failed_count,
        "details": results
    }