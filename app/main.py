from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.db.qdrant import connect_to_qdrant
from app.api.documents import router as documents_router

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
    await connect_to_mongo()
    connect_to_qdrant()
    logger.info("✓ All connections initialized")
    yield
    # Shutdown
    logger.info("🛑 Shutting down Quiz App API...")
    await close_mongo_connection()
    logger.info("✓ Cleanup complete")

app = FastAPI(
    title="Quiz App API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents_router, prefix="/api", tags=["Documents"])

@app.get("/")
async def root():
    return {
        "message": "Quiz App API",
        "version": "1.0.0",
        "docs": "/docs"
    }