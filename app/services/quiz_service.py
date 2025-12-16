"""
Quiz Service
Business logic for quiz generation, submission, and management
FILE: app/services/quiz_service.py
"""
import logging
from uuid import uuid4
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from app.db.mongodb import get_database
from app.db.qdrant import get_qdrant_service
from app.mcp_client.client import get_mcp_client, MCPClientError 
from app.utils.context_loader import load_context
from app.utils.quiz_prompt import build_quiz_prompt
from app.services.llm_client import generate_quiz, LLMClientError, LLMTimeoutError, LLMAPIError
from app.utils.quiz_parser import parse_quiz_json, QuizParseError
from app.db.quiz_db import (
    insert_quiz_attempt,
    get_quiz_attempt,
    get_quiz_attempts_by_quiz,
    get_quiz_statistics
)

logger = logging.getLogger(__name__)


# ==================== CUSTOM EXCEPTIONS ====================

class QuizGenerationError(Exception):
    """Base exception for quiz generation errors"""
    pass


class MaterialNotFoundError(QuizGenerationError):
    """Raised when material is not found"""
    pass


class ChunkRetrievalError(QuizGenerationError):
    """Raised when chunk retrieval fails"""
    pass


class QuizNotFoundError(Exception):
    """Raised when quiz is not found in database"""
    pass


class QuizValidationError(Exception):
    """Raised when quiz submission validation fails"""
    pass


# ==================== QUIZ SERVICE ====================

class QuizService:
    """Service for generating, submitting, and managing quizzes"""
    
    def __init__(self):
        self.db = get_database()
        self.collection = self.db["quizzes"]
        self.mcp_client = get_mcp_client()
    
    # ==================== QUIZ GENERATION ====================
    
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
            result = await self.mcp_client.get_material_chunks(
                material_id=material_id,
                include_vectors=False
            )
            if "error" in result:
                raise ChunkRetrievalError(f"MCP Error: {result['error']}")
            
            chunks = result.get("chunks", [])
            
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
        logger.info("📋 Parsing LLM response...")
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
    
    async def get_all_quizzes(
        self,
        skip: int = 0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all quizzes across all materials with pagination
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
        
        Returns:
            List of quizzes sorted by creation date (newest first)
        """
        print("Fetching all quizzes from database...")
        quizzes = []
        cursor = self.collection.find().sort("created_at", -1).skip(skip).limit(limit)
        
        async for quiz in cursor:
            quiz["_id"] = str(quiz["_id"])
            quizzes.append(quiz)
        
        logger.info(f"📚 Retrieved {len(quizzes)} quizzes (skip={skip}, limit={limit})")
        return quizzes
    
    async def delete_quiz(self, quiz_id: str) -> bool:
        """Delete a quiz by ID"""
        result = await self.collection.delete_one({"quiz_id": quiz_id})
        return result.deleted_count > 0
    
    # ==================== QUIZ SUBMISSION ====================
    
    async def submit_quiz_attempt(
        self,
        quiz_id: str,
        user_answers: List[str]
    ) -> Dict[str, Any]:
        """
        Submit quiz answers and get graded results
        
        Workflow:
        1. Fetch quiz from database
        2. Validate quiz exists
        3. Extract correct answers
        4. Validate answer count
        5. Calculate score (delegated to insert_quiz_attempt)
        6. Save attempt to MongoDB
        7. Return results
        
        Args:
            quiz_id: Quiz UUID
            user_answers: List of user-selected answers
        
        Returns:
            Dictionary with grading results
        
        Raises:
            QuizNotFoundError: If quiz doesn't exist
            QuizValidationError: If validation fails
        """
        try:
            logger.info(f"🎯 Processing quiz submission: {quiz_id}")
            
            # Step 1: Fetch quiz
            quiz = await self.get_quiz(quiz_id)
            
            if not quiz:
                raise QuizNotFoundError(
                    f"Quiz '{quiz_id}' not found. Please verify the quiz ID is correct."
                )
            
            # Step 2: Extract correct answers from questions
            questions = quiz.get("questions", [])
            
            if not questions:
                raise QuizValidationError(
                    "Quiz does not have questions defined. Please contact support."
                )
            
            # Extract 'answer' field from each question
            correct_answers = [q.get("answer") for q in questions]
            
            # Validate all questions have answers
            if None in correct_answers:
                raise QuizValidationError(
                    "Quiz has questions without correct answers. Please contact support."
                )
            
            logger.info(
                f"📋 Quiz loaded: {len(correct_answers)} questions, "
                f"{len(user_answers)} user answers"
            )
            
            # Step 3: Validate answer count
            if len(user_answers) != len(correct_answers):
                raise QuizValidationError(
                    f"Answer count mismatch: expected {len(correct_answers)} answers "
                    f"but received {len(user_answers)}. Please answer all questions."
                )
            
            # Step 4: Save attempt & calculate score
            result = await insert_quiz_attempt(
                quiz_id=quiz_id,
                user_answers=user_answers,
                correct_answers=correct_answers
            )
            
            # Step 5: Enhance result
            result["user_answers"] = user_answers
            result["correct_answers"] = correct_answers
            
            logger.info(
                f"✅ Quiz attempt processed: "
                f"score={result['score']}/{result['total']} ({result['percentage']}%)"
            )
            
            return result
            
        except (QuizNotFoundError, QuizValidationError):
            raise
        except Exception as e:
            logger.error(f"❌ Quiz submission failed: {e}", exc_info=True)
            raise Exception(f"Failed to process quiz submission: {str(e)}")
    
    # ==================== ATTEMPT RETRIEVAL ====================
    
    async def get_attempt_by_id(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        """Get quiz attempt by ID"""
        try:
            return await get_quiz_attempt(attempt_id)
        except Exception as e:
            logger.error(f"❌ Failed to get attempt {attempt_id}: {e}")
            return None
    
    async def get_attempts_for_quiz(
        self,
        quiz_id: str,
        skip: int = 0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get all attempts for a quiz"""
        try:
            return await get_quiz_attempts_by_quiz(quiz_id, skip, limit)
        except Exception as e:
            logger.error(f"❌ Failed to get attempts for quiz {quiz_id}: {e}")
            return []
    
    async def get_quiz_stats(self, quiz_id: str) -> Dict[str, Any]:
        """Get statistical summary for a quiz"""
        try:
            return await get_quiz_statistics(quiz_id)
        except Exception as e:
            logger.error(f"❌ Failed to get statistics for quiz {quiz_id}: {e}")
            return {
                "total_attempts": 0,
                "error": str(e)
            }
    
    async def get_all_attempts_for_quiz(self, quiz_id: str) -> List[Dict[str, Any]]:
        """
        Get ALL attempts for a quiz (no pagination)
        
        Args:
            quiz_id: Quiz identifier
        
        Returns:
            Complete list of all attempts for the quiz
        """
        try:
            # Use a large limit to get all attempts
            # If you expect more than 1000 attempts per quiz, implement proper pagination
            return await get_quiz_attempts_by_quiz(quiz_id, skip=0, limit=1000)
        except Exception as e:
            logger.error(f"❌ Failed to get all attempts for quiz {quiz_id}: {e}")
            return []
    
    async def get_recent_attempts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent quiz attempts across all quizzes
        
        Args:
            limit: Maximum number of attempts to return
        
        Returns:
            List of recent attempts sorted by date (newest first)
        """
        try:
            from app.db.quiz_db import get_all_quiz_attempts
            return await get_all_quiz_attempts(skip=0, limit=limit)
        except Exception as e:
            logger.error(f"❌ Failed to get recent attempts: {e}")
            return []


# ==================== SINGLETON ====================

_quiz_service: Optional[QuizService] = None


def get_quiz_service() -> QuizService:
    """Get or create the global QuizService instance"""
    global _quiz_service
    
    if _quiz_service is None:
        _quiz_service = QuizService()
    
    return _quiz_service