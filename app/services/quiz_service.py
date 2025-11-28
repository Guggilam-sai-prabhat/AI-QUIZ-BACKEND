"""
Quiz Service
Business logic for quiz generation and management
FILE: app/services/quiz_service.py
"""
import logging
from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from app.db.mongodb import get_database
from app.db.qdrant import get_qdrant_service
from app.mcp_server.context_loader import load_context
from app.mcp_server.quiz_prompt import build_quiz_prompt
from app.mcp_server.llm_client import generate_quiz, LLMClientError, LLMTimeoutError, LLMAPIError
from app.mcp_server.quiz_parser import parse_quiz_json, QuizParseError

logger = logging.getLogger(__name__)


class QuizGenerationError(Exception):
    """Base exception for quiz generation errors"""
    pass


class MaterialNotFoundError(QuizGenerationError):
    """Raised when material is not found"""
    pass


class ChunkRetrievalError(QuizGenerationError):
    """Raised when chunk retrieval fails"""
    pass


class QuizService:
    """Service for generating and managing quizzes"""
    
    def __init__(self):
        self.db = get_database()
        self.collection = self.db["quizzes"]
        self.qdrant_service = get_qdrant_service()
    
    async def _get_material_chunks(
        self,
        material_id: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-K chunks for a material
        
        Args:
            material_id: MongoDB document ID
            top_k: Number of chunks to retrieve
        
        Returns:
            List of chunk dictionaries
        """
        try:
            # Get chunks from Qdrant filtered by material_id
            # Uses get_material_chunks which returns all chunks sorted by chunk_id
            chunks = await self.qdrant_service.get_material_chunks(
                material_id=material_id,
                with_vectors=False
            )
            
            # Limit to top_k chunks
            if top_k and len(chunks) > top_k:
                chunks = chunks[:top_k]
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve chunks: {e}")
            raise ChunkRetrievalError(f"Failed to retrieve chunks: {e}")
    
    async def generate_quiz(
        self,
        material_id: str,
        num_questions: int = 5,
        top_k_chunks: int = 10,
        llm_provider: str = "grok"
    ) -> Dict[str, Any]:
        """
        Generate a quiz from material content
        
        Args:
            material_id: MongoDB document ID
            num_questions: Number of questions to generate
            top_k_chunks: Number of chunks to use for context
            llm_provider: LLM provider (grok, openai, anthropic)
        
        Returns:
            Dictionary with quiz_id, question_count, status
        """
        logger.info(f"🎯 Generating quiz for material {material_id}")
        
        # Step 1: Retrieve top-K chunks
        logger.info(f"📚 Retrieving top {top_k_chunks} chunks...")
        chunks = await self._get_material_chunks(material_id, top_k_chunks)
        
        if not chunks:
            raise MaterialNotFoundError(
                f"No chunks found for material {material_id}"
            )
        
        logger.info(f"✅ Retrieved {len(chunks)} chunks")
        
        # Step 2: Format context
        logger.info("📝 Formatting context...")
        context = await load_context(material_id, chunks)
        
        # Step 3: Build prompt
        logger.info(f"🔨 Building prompt for {num_questions} questions...")
        prompt = build_quiz_prompt(context, num_questions)
        
        # Step 4: Call LLM
        logger.info(f"🤖 Calling {llm_provider} LLM...")
        try:
            raw_response = await generate_quiz(prompt, provider=llm_provider)
        except LLMTimeoutError as e:
            raise QuizGenerationError(f"LLM request timed out: {e}")
        except LLMAPIError as e:
            raise QuizGenerationError(f"LLM API error: {e}")
        except LLMClientError as e:
            raise QuizGenerationError(f"LLM client error: {e}")
        
        # Step 5: Parse JSON response
        logger.info("🔍 Parsing LLM response...")
        try:
            questions = parse_quiz_json(raw_response)
        except QuizParseError as e:
            logger.error(f"❌ Failed to parse quiz: {e}")
            logger.debug(f"Raw response: {raw_response[:500]}...")
            raise QuizGenerationError(f"Failed to parse quiz response: {e}")
        
        # Step 6: Save to MongoDB
        logger.info("💾 Saving quiz to database...")
        quiz_id = str(uuid4())
        
        quiz_doc = {
            "quiz_id": quiz_id,
            "material_id": material_id,
            "questions": questions,
            "question_count": len(questions),
            "llm_provider": llm_provider,
            "chunks_used": len(chunks),
            "created_at": datetime.now(timezone.utc)
        }
        
        await self.collection.insert_one(quiz_doc)
        
        logger.info(f"✅ Quiz saved with ID: {quiz_id}")
        
        return {
            "quiz_id": quiz_id,
            "question_count": len(questions),
            "status": "success"
        }
    
    async def get_quiz(self, quiz_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a quiz by ID"""
        quiz = await self.collection.find_one({"quiz_id": quiz_id})
        
        if quiz:
            quiz["_id"] = str(quiz["_id"])
            
        return quiz
    
    async def get_quizzes_by_material(
        self,
        material_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get all quizzes for a material"""
        quizzes = []
        cursor = self.collection.find(
            {"material_id": material_id}
        ).sort("created_at", -1).limit(limit)
        
        async for quiz in cursor:
            quiz["_id"] = str(quiz["_id"])
            quizzes.append(quiz)
        
        return quizzes
    
    async def delete_quiz(self, quiz_id: str) -> bool:
        """Delete a quiz by ID"""
        result = await self.collection.delete_one({"quiz_id": quiz_id})
        return result.deleted_count > 0


# Singleton instance
_quiz_service: Optional[QuizService] = None


def get_quiz_service() -> QuizService:
    """Get or create the global QuizService instance"""
    global _quiz_service
    
    if _quiz_service is None:
        _quiz_service = QuizService()
    
    return _quiz_service