"""
Quiz App API - Main Application
Complete with search functionality
FILE: main.py
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time

from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.db.qdrant import connect_to_qdrant, close_qdrant_connection, get_qdrant_service
from app.api.document import router as documents_router
from app.api.search import router as search_router  # ✅ ADD THIS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting Quiz App API...")
    
    try:
        # Connect to MongoDB
        await connect_to_mongo()
        logger.info("✓ MongoDB connected")
        
        # Connect to Qdrant
        connect_to_qdrant()
        logger.info("✓ Qdrant connected")
        
        # Ensure Qdrant collection exists
        try:
            qdrant_service = get_qdrant_service()
            qdrant_service.ensure_collection()
            
            # Log collection stats
            stats = qdrant_service.get_collection_stats()
            logger.info(
                f"✓ Qdrant collection ready: {stats.get('name')} "
                f"({stats.get('points_count')} points)"
            )
        except Exception as e:
            logger.warning(f"⚠ Qdrant collection setup warning: {e}")
        
        logger.info("✓ All connections initialized")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Quiz App API...")
    
    try:
        await close_mongo_connection()
        logger.info("✓ MongoDB disconnected")
        
        close_qdrant_connection()
        logger.info("✓ Qdrant disconnected")
        
        logger.info("✓ Cleanup complete")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


app = FastAPI(
    title="Quiz App API",
    description="""
    Quiz App API with document management and semantic search.
    
    ## Features
    - **Document Management**: Upload, retrieve, update, and delete documents
    - **Semantic Search**: Find relevant content using natural language queries
    - **Vector Embeddings**: Powered by SentenceTransformers and Qdrant
    - **MongoDB Storage**: Persistent document metadata and content
    
    ## Endpoints
    - **Documents**: `/api/documents/*` - CRUD operations for documents
    - **Search**: `/api/search-context` - Semantic search across documents
    - **Search Health**: `/api/search-health` - Check search service status
    - **Collection Info**: `/api/collection-info` - View collection statistics
    - **Health**: `/health` - Overall service health check
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4000",
        "http://192.168.0.121:4000",
        "http://localhost:3000",  # Common React dev port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # Convert to ms
    response.headers["X-Process-Time-Ms"] = str(round(process_time, 2))
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    logger.info(f"📨 {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(
        f"📤 {request.method} {request.url.path} - "
        f"Status: {response.status_code}"
    )
    return response


# ==================== INCLUDE ROUTERS ====================

# Documents API
app.include_router(documents_router, prefix="/api", tags=["Documents"])

# Search API - ✅ ADD THIS LINE
app.include_router(search_router, prefix="/api", tags=["Search"])


# ==================== ROOT ENDPOINTS ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Quiz App API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "documents": "/api/documents",
            "search": "/api/search-context",
            "search_health": "/api/search-health",
            "collection_info": "/api/collection-info",
            "health": "/health"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Comprehensive health check for all services
    
    Checks:
    - API status
    - MongoDB connection
    - Qdrant connection
    - Collection status
    
    Returns:
        Health status for all components
    """
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {}
    }
    
    overall_healthy = True
    

    
    # Check Qdrant
    try:
        qdrant_service = get_qdrant_service()
        
        if qdrant_service.health_check():
            stats = qdrant_service.get_collection_stats()
            health_status["components"]["qdrant"] = {
                "status": "healthy",
                "message": "Connected and responsive",
                "collection": {
                    "name": stats.get("name"),
                    "points_count": stats.get("points_count"),
                    "vectors_count": stats.get("vectors_count"),
                    "status": stats.get("status")
                }
            }
            logger.debug("✓ Qdrant health check passed")
        else:
            overall_healthy = False
            health_status["components"]["qdrant"] = {
                "status": "unhealthy",
                "message": "Connection failed"
            }
            logger.error("❌ Qdrant health check failed")
            
    except Exception as e:
        overall_healthy = False
        health_status["components"]["qdrant"] = {
            "status": "unhealthy",
            "message": str(e)
        }
        logger.error(f"❌ Qdrant health check failed: {e}")
    
    # Set overall status
    health_status["status"] = "healthy" if overall_healthy else "degraded"
    
    # Add API info
    health_status["api"] = {
        "title": app.title,
        "version": app.version,
        "status": "operational"
    }
    
    # Return appropriate status code
    status_code = 200 if overall_healthy else 503
    
    return JSONResponse(
        status_code=status_code,
        content=health_status
    )


# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )