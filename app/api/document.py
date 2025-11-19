from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List, Optional
import os
import shutil
import secrets
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import logging
import fitz  # PyMuPDF
from app.models.document import DocumentMetadata, DocumentResponse, EmbeddingResponse
from app.db.mongodb import get_collection

# Import chunking, embedding, and vector services
from app.services.text_chunker import chunk_text, get_chunk_stats
from app.services.embedding_service import get_embeddings, get_embedding_dimension
from app.services.qdrant_service import get_qdrant_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Configuration
UPLOAD_DIR = "uploads"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {".pdf"}
ALLOWED_MIME_TYPES = {"application/pdf"}

# Vector pipeline configuration
CHUNK_SIZE = 500  # tokens
CHUNK_OVERLAP = 50  # tokens
EMBEDDING_MODEL = "sentence-transformers"  # or "openai"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Default for sentence-transformers

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


async def process_embeddings_pipeline(
    document_id: str,
    extracted_text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> dict:
    """
    Process the complete embeddings pipeline:
    1. Chunk text
    2. Generate embeddings
    3. Store in Qdrant
    4. Update MongoDB with chunk_count
    
    Args:
        document_id: MongoDB document ID
        extracted_text: Text extracted from PDF
        chunk_size: Token size per chunk
        chunk_overlap: Overlapping tokens between chunks
    
    Returns:
        Dictionary with pipeline results
    """
    try:
        logger.info(f"🔄 Starting embedding pipeline for document {document_id}")
        
        # Step 1: Chunk the text
        logger.info(f"📝 Step 1: Chunking text ({len(extracted_text)} chars)")
        chunks = chunk_text(
            text=extracted_text,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )
        
        if not chunks:
            logger.warning(f"⚠️ No chunks generated for document {document_id}")
            return {
                "material_id": document_id,
                "chunk_count": 0,
                "status": "no_content"
            }
        
        # Get chunk statistics
        stats = get_chunk_stats(extracted_text, chunks)
        logger.info(
            f"✅ Created {len(chunks)} chunks "
            f"(avg: {stats['avg_chunk_tokens']:.0f} tokens, "
            f"range: {stats['min_chunk_tokens']}-{stats['max_chunk_tokens']})"
        )
        
        # Step 2: Generate embeddings
        logger.info(f"🧠 Step 2: Generating embeddings ({EMBEDDING_MODEL})")
        embeddings = get_embeddings(
            chunks=chunks,
            model=EMBEDDING_MODEL,
            model_name=EMBEDDING_MODEL_NAME
        )
        
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: {len(embeddings)} embeddings "
                f"for {len(chunks)} chunks"
            )
        
        embedding_dim = len(embeddings[0]) if embeddings else 0
        logger.info(f"✅ Generated {len(embeddings)} embeddings (dim: {embedding_dim})")
        
        # Step 3: Store in Qdrant
        logger.info(f"💾 Step 3: Storing vectors in Qdrant")
        vector_service = get_qdrant_service()
        
        # Ensure collection exists with correct dimension
        vector_service.ensure_collection(vector_size=embedding_dim)
        
        # Upsert embeddings
        upsert_result = await vector_service.upsert_embeddings(
            material_id=document_id,
            chunks=chunks,
            embeddings=embeddings
        )
        
        if upsert_result["status"] != "success":
            raise Exception(f"Vector upsert failed: {upsert_result.get('error')}")
        
        logger.info(
            f"✅ Stored {upsert_result['chunks_upserted']} vectors in Qdrant"
        )
        
        # Step 4: Update MongoDB with chunk_count
        logger.info(f"📊 Step 4: Updating MongoDB document")
        collection = get_collection("documents")
        from bson import ObjectId
        
        update_result = await collection.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "chunk_count": len(chunks),
                    "embedding_model": EMBEDDING_MODEL,
                    "embedding_model_name": EMBEDDING_MODEL_NAME,
                    "embedding_dimension": embedding_dim,
                    "indexed_at": datetime.now(timezone.utc)
                }
            }
        )
        
        if update_result.modified_count == 0:
            logger.warning(f"⚠️ MongoDB document not updated for {document_id}")
        else:
            logger.info(f"✅ MongoDB document updated with chunk_count={len(chunks)}")
        
        # Return success result
        return {
            "material_id": document_id,
            "chunk_count": len(chunks),
            "embedding_dimension": embedding_dim,
            "status": "embedded",
            "stats": {
                "total_chunks": len(chunks),
                "avg_chunk_tokens": stats['avg_chunk_tokens'],
                "total_tokens": stats['total_chunk_tokens']
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Embedding pipeline failed for {document_id}: {str(e)}")
        
        # Update MongoDB with error status
        try:
            collection = get_collection("documents")
            from bson import ObjectId
            await collection.update_one(
                {"_id": ObjectId(document_id)},
                {
                    "$set": {
                        "embedding_status": "failed",
                        "embedding_error": str(e),
                        "embedding_attempted_at": datetime.now(timezone.utc)
                    }
                }
            )
        except Exception as update_error:
            logger.error(f"Failed to update error status: {update_error}")
        
        return {
            "material_id": document_id,
            "chunk_count": 0,
            "status": "error",
            "error": str(e)
        }


@router.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint
    Checks MongoDB, Qdrant, and upload directory
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
    
    # Check Qdrant connection
    try:
        vector_service = get_qdrant_service()
        is_healthy = vector_service.health_check()
        health_status["checks"]["qdrant"] = "connected" if is_healthy else "disconnected"
        if not is_healthy:
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["checks"]["qdrant"] = "error"
        health_status["status"] = "degraded"
        logger.error(f"Qdrant health check failed: {str(e)}")
    
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


@router.post("/upload", response_model=EmbeddingResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF document with complete vector embedding pipeline:
    1. Extract text from PDF
    2. Chunk text into manageable pieces
    3. Generate embeddings for each chunk
    4. Store vectors in Qdrant
    5. Update MongoDB with chunk count
    """
    file_path = None
    document_id = None
    
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
            
            # Return the existing document with embedding info
            return EmbeddingResponse(
                id=str(existing_doc["_id"]),
                filename=existing_doc["filename"],
                file_size=existing_doc["file_size"],
                content_type=existing_doc["content_type"],
                file_hash=existing_doc["file_hash"],
                uploaded_at=existing_doc["uploaded_at"],
                material_id=str(existing_doc["_id"]),
                chunk_count=existing_doc.get("chunk_count", 0),
                status=existing_doc.get("embedding_status", "unknown")
            )
        
        logger.info(f"✅ No duplicate found. Proceeding with new upload.")
        
        # Extract text from PDF
        try:
            extracted_text, page_count = extract_text_from_pdf(file_path)
            logger.info(f"✅ Text extraction successful: {page_count} pages, {len(extracted_text)} chars")
        except ValueError as ve:
            # Handle empty or invalid PDFs
            logger.error(f"❌ PDF validation failed: {str(ve)}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            # Handle other PDF processing errors
            logger.error(f"❌ PDF processing failed: {str(e)}")
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
            extracted_text=extracted_text,
            page_count=page_count
        )
        
        # Store in MongoDB
        result = await collection.insert_one(
            document.model_dump(by_alias=True, exclude=["id"])
        )
        
        document_id = str(result.inserted_id)
        logger.info(f"✅ New document inserted with ID: {document_id}")
        
        # Process embedding pipeline
        logger.info(f"🚀 Starting embedding pipeline for document {document_id}")
        embedding_result = await process_embeddings_pipeline(
            document_id=document_id,
            extracted_text=extracted_text,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Return comprehensive response
        return EmbeddingResponse(
            id=document_id,
            filename=document.filename,
            file_size=document.file_size,
            content_type=document.content_type,
            file_hash=document.file_hash,
            uploaded_at=document.uploaded_at,
            material_id=embedding_result["material_id"],
            chunk_count=embedding_result["chunk_count"],
            status=embedding_result["status"],
            embedding_stats=embedding_result.get("stats")
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        # Cleanup on error
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file after error: {file_path}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup file: {cleanup_error}")
        
        # Try to delete from MongoDB if document was created
        if document_id:
            try:
                collection = get_collection("documents")
                from bson import ObjectId
                await collection.delete_one({"_id": ObjectId(document_id)})
                logger.info(f"Cleaned up MongoDB document: {document_id}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup MongoDB document: {cleanup_error}")
        
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during file upload. Please try again."
        )


@router.get("/debug/document/{document_id}")
async def debug_document(document_id: str):
    """Debug endpoint to see raw document data including embedding info"""
    from bson import ObjectId
    
    try:
        collection = get_collection("documents")
        doc = await collection.find_one({"_id": ObjectId(document_id)})
        
        if not doc:
            return {"error": "Document not found"}
        
        # Convert ObjectId to string for JSON serialization
        doc["_id"] = str(doc["_id"])
        
        # Exclude extracted_text for brevity (can be very long)
        if "extracted_text" in doc:
            doc["extracted_text_length"] = len(doc["extracted_text"])
            doc["extracted_text_preview"] = doc["extracted_text"][:500] + "..."
            del doc["extracted_text"]
        
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
            extracted_text=doc.get("extracted_text"),
            page_count=doc.get("page_count"),
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
    Delete a document by ID (includes cleanup of vectors from Qdrant)
    
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
        
        # Delete vectors from Qdrant
        try:
            vector_service = get_qdrant_service()
            await vector_service.delete_material_embeddings(document_id)
            logger.info(f"✅ Deleted vectors from Qdrant for document {document_id}")
        except Exception as e:
            logger.error(f"⚠️ Failed to delete vectors from Qdrant: {str(e)}")
            # Continue with deletion even if vector cleanup fails
        
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


@router.get("/vectors/stats")
async def get_vector_stats():
    """
    Get statistics about the vector collection in Qdrant
    
    Returns:
        Collection statistics including point count and configuration
    """
    try:
        vector_service = get_qdrant_service()
        stats = vector_service.get_collection_stats()
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get vector stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve vector statistics"
        )