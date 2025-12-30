"""
Quiz Session Service - Extended with Quiz Completion Logic
MongoDB service functions for quiz sessions with answer evaluation and end conditions
FIXED: Matches your existing schema where QuestionRecord includes correctAnswer
"""
from datetime import datetime
from typing import Optional, Literal, Tuple, List
import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.quiz_sessions import (
    QuizSession,
    QuestionRecord,
    QuestionMetadata,
    QuizSessionResponse
)

logger = logging.getLogger(__name__)

# CONFIGURABLE QUIZ END RULES
MAX_QUESTIONS = 8  # Default maximum questions per quiz
ALLOW_CHUNK_REUSE = False  # Whether to allow same chunk twice


class QuizSessionService:
    """Service class for quiz session operations with completion detection"""
    
    COLLECTION_NAME = "quiz_sessions"
    
    def __init__(self, db: AsyncIOMotorDatabase, max_questions: int = MAX_QUESTIONS):
        """
        Initialize quiz session service
        
        Args:
            db: MongoDB database instance
            max_questions: Maximum questions per quiz (default: 8)
        """
        self.db = db
        self.collection = db[self.COLLECTION_NAME]
        self.max_questions = max_questions
    
    async def create_session(self, material_id: str) -> QuizSession:
        """
        Create a new quiz session
        
        Args:
            material_id: ID of the study material
        
        Returns:
            Created QuizSession object
        """
        session = QuizSession(materialId=material_id)
        
        try:
            await self.collection.insert_one(session.model_dump(by_alias=True))
            logger.info(f"✅ Created quiz session: {session.sessionId} for material: {material_id}")
            return session
        
        except Exception as e:
            logger.error(f"❌ Failed to create session for material {material_id}: {e}")
            raise Exception(f"Failed to create quiz session: {str(e)}")
    
    async def store_question_metadata(
        self,
        session_id: str,
        metadata: QuestionMetadata
    ) -> bool:
        """
        Store complete question metadata in session
        
        SECURITY: This includes the correct answer for later evaluation
        
        Args:
            session_id: ID of the quiz session
            metadata: Complete question metadata including correct answer and optional chunkId
        
        Returns:
            True if successful
        """
        try:
            result = await self.collection.update_one(
                {"sessionId": session_id},
                {
                    "$push": {"questionMetadata": metadata.model_dump(by_alias=True)},
                    "$set": {"updatedAt": datetime.utcnow()}
                }
            )
            
            if result.matched_count == 0:
                logger.error(f"❌ Session not found: {session_id}")
                raise ValueError(f"Session not found: {session_id}")
            
            if result.modified_count > 0:
                logger.info(
                    f"✅ Stored question metadata for session {session_id} - "
                    f"Question: {metadata.questionId}"
                )
                return True
            
            logger.warning(f"⚠️ No changes made to session {session_id}")
            return False
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to store question metadata in session {session_id}: {e}")
            raise Exception(f"Failed to store question metadata: {str(e)}")
    
    async def append_question(
        self,
        session_id: str,
        question_id: str,
        difficulty: str,
        correct_answer: str,
        was_correct: Optional[bool] = None
    ) -> bool:
        """
        Append a question answer record to session
        
        UPDATED: Matches your schema where QuestionRecord includes correctAnswer
        
        Args:
            session_id: ID of the quiz session
            question_id: Question identifier
            difficulty: Question difficulty level
            correct_answer: The correct answer (A, B, C, D)
            was_correct: Whether answer was correct (None for first question)
        
        Returns:
            True if successful
        """
        try:
            record = QuestionRecord(
                questionId=question_id,
                difficulty=difficulty,
                correctAnswer=correct_answer,
                wasCorrect=was_correct
            )
            
            result = await self.collection.update_one(
                {"sessionId": session_id},
                {
                    "$push": {"questions": record.model_dump(by_alias=True)},
                    "$set": {"updatedAt": datetime.utcnow()}
                }
            )
            
            if result.matched_count == 0:
                logger.error(f"❌ Session not found: {session_id}")
                raise ValueError(f"Session not found: {session_id}")
            
            if result.modified_count > 0:
                correctness_str = (
                    "N/A (first question)" if was_correct is None 
                    else str(was_correct)
                )
                logger.info(
                    f"✅ Appended answer record to session {session_id} - "
                    f"Question: {question_id}, Correct: {correctness_str}"
                )
                return True
            
            logger.warning(f"⚠️ No changes made to session {session_id}")
            return False
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to append answer record to session {session_id}: {e}")
            raise Exception(f"Failed to append answer record: {str(e)}")
    
    async def evaluate_answer(
        self,
        session_id: str,
        question_id: str,
        selected_answer: str
    ) -> Tuple[bool, str, str]:
        """
        Evaluate submitted answer against stored correct answer
        
        Args:
            session_id: Quiz session ID
            question_id: Question ID being answered
            selected_answer: User's selected answer (A, B, C, D)
        
        Returns:
            Tuple of (was_correct, correct_answer, difficulty)
        
        Raises:
            ValueError: If session or question not found
            Exception: If database operation fails
        """
        try:
            # Fetch session
            session = await self.get_session(session_id)
            
            if not session:
                logger.error(f"❌ Session not found: {session_id}")
                raise ValueError(f"Session not found: {session_id}")
            
            # Find question metadata
            question_metadata = None
            for metadata in session.questionMetadata:
                if metadata.questionId == question_id:
                    question_metadata = metadata
                    break
            
            if not question_metadata:
                logger.error(
                    f"❌ Question {question_id} not found in session {session_id}"
                )
                raise ValueError(
                    f"Question {question_id} not found in session. "
                    f"It may not have been generated yet."
                )
            
            # Evaluate correctness
            correct_answer = question_metadata.correctAnswer
            was_correct = selected_answer.upper() == correct_answer.upper()
            
            logger.info(
                f"✅ Evaluated answer - Session: {session_id}, "
                f"Question: {question_id}, Selected: {selected_answer}, "
                f"Correct: {correct_answer}, Result: {'✓' if was_correct else '✗'}"
            )
            
            return (
                was_correct,
                correct_answer,
                question_metadata.difficulty
            )
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to evaluate answer for session {session_id}: {e}")
            raise Exception(f"Failed to evaluate answer: {str(e)}")
    
    async def is_first_question(self, session_id: str) -> bool:
        """
        Check if this is the first question in the session
        
        Args:
            session_id: Quiz session ID
        
        Returns:
            True if session has no questions yet, False otherwise
        
        Raises:
            ValueError: If session not found
        """
        try:
            session = await self.get_session(session_id)
            
            if not session:
                logger.error(f"❌ Session not found: {session_id}")
                raise ValueError(f"Session not found: {session_id}")
            
            is_first = len(session.questions) == 0
            logger.debug(
                f"📊 Session {session_id} question count: {len(session.questions)}, "
                f"isFirst: {is_first}"
            )
            return is_first
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to check if first question for session {session_id}: {e}")
            raise Exception(f"Failed to check session state: {str(e)}")
    
    # ============================================================================
    # QUIZ COMPLETION LOGIC - NEW METHODS
    # ============================================================================
    
    async def count_questions(self, session_id: str) -> int:
        """
        Count total questions in session
        
        Args:
            session_id: Quiz session ID
        
        Returns:
            Number of questions in session
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Count based on questionMetadata (generated questions)
            count = len(session.questionMetadata)
            logger.debug(f"📊 Session {session_id} has {count} questions")
            return count
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to count questions for session {session_id}: {e}")
            raise Exception(f"Failed to count questions: {str(e)}")
    
    async def get_used_chunk_ids(self, session_id: str) -> List[str]:
        """
        Get list of chunk IDs already used in this session
        
        Args:
            session_id: Quiz session ID
        
        Returns:
            List of chunk IDs used in questionMetadata
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Extract chunk IDs from metadata if stored
            chunk_ids = []
            for metadata in session.questionMetadata:
                # Get chunkId if it exists
                chunk_id = getattr(metadata, 'chunkId', None)
                if chunk_id:
                    chunk_ids.append(chunk_id)
            
            logger.debug(f"📊 Session {session_id} used chunks: {chunk_ids}")
            return chunk_ids
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to get used chunks for session {session_id}: {e}")
            raise Exception(f"Failed to get used chunks: {str(e)}")
    
    async def should_end_quiz(
        self,
        session_id: str,
        retrieved_chunk_id: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if quiz should end based on multiple conditions
        
        Args:
            session_id: Quiz session ID
            retrieved_chunk_id: Optional chunk ID that was just retrieved
        
        Returns:
            Tuple of (should_end, reason)
            - should_end: True if quiz should end
            - reason: Explanation why quiz ended (if applicable)
        
        Completion Rules:
        1. Max questions reached (configurable)
        2. Chunk reuse detected (if not allowed)
        3. No content available (handled upstream)
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Rule 1: Check max questions limit
            question_count = len(session.questionMetadata)
            if question_count >= self.max_questions:
                reason = f"Maximum {self.max_questions} questions reached"
                logger.info(f"🏁 Session {session_id} ending: {reason}")
                return (True, reason)
            
            # Rule 2: Check chunk reuse (if provided and not allowed)
            if not ALLOW_CHUNK_REUSE and retrieved_chunk_id:
                used_chunks = await self.get_used_chunk_ids(session_id)
                if retrieved_chunk_id in used_chunks:
                    reason = "All unique content exhausted (chunk reuse detected)"
                    logger.info(f"🏁 Session {session_id} ending: {reason}")
                    return (True, reason)
            
            # No end condition met
            logger.debug(f"✅ Session {session_id} should continue")
            return (False, None)
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to check end condition for session {session_id}: {e}")
            raise Exception(f"Failed to check end condition: {str(e)}")
    
    # ============================================================================
    # EXISTING METHODS (unchanged)
    # ============================================================================
    
    async def update_session_status(
        self,
        session_id: str,
        status: Literal["ongoing", "completed", "abandoned"]
    ) -> bool:
        """Update the status of a quiz session"""
        valid_statuses = ["ongoing", "completed", "abandoned"]
        if status not in valid_statuses:
            raise ValueError(f"Invalid status: {status}. Must be one of: {valid_statuses}")
        
        try:
            result = await self.collection.update_one(
                {"sessionId": session_id},
                {
                    "$set": {
                        "status": status,
                        "updatedAt": datetime.utcnow()
                    }
                }
            )
            
            if result.matched_count == 0:
                logger.error(f"❌ Session not found: {session_id}")
                raise ValueError(f"Session not found: {session_id}")
            
            if result.modified_count > 0:
                logger.info(f"✅ Updated session {session_id} status to: {status}")
                return True
            
            logger.warning(f"⚠️ No changes made to session {session_id}")
            return False
        
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"❌ Failed to update session {session_id} status: {e}")
            raise Exception(f"Failed to update session status: {str(e)}")
    
    async def get_session(self, session_id: str) -> Optional[QuizSession]:
        """Retrieve a quiz session by ID"""
        try:
            doc = await self.collection.find_one({"sessionId": session_id})
            
            if doc:
                doc.pop("_id", None)
                logger.debug(f"✅ Retrieved session: {session_id}")
                return QuizSession(**doc)
            
            logger.warning(f"⚠️ Session not found: {session_id}")
            return None
        
        except Exception as e:
            logger.error(f"❌ Failed to retrieve session {session_id}: {e}")
            raise Exception(f"Failed to retrieve session: {str(e)}")
    
    async def get_sessions_by_material(
        self,
        material_id: str,
        limit: int = 10
    ) -> list[QuizSession]:
        """Get all sessions for a specific material"""
        try:
            cursor = self.collection.find(
                {"materialId": material_id}
            ).sort("createdAt", -1).limit(limit)
            
            sessions = []
            async for doc in cursor:
                doc.pop("_id", None)
                sessions.append(QuizSession(**doc))
            
            logger.info(f"✅ Retrieved {len(sessions)} sessions for material: {material_id}")
            return sessions
        
        except Exception as e:
            logger.error(f"❌ Failed to retrieve sessions for material {material_id}: {e}")
            raise Exception(f"Failed to retrieve sessions: {str(e)}")
    
    async def get_session_stats(self, session_id: str) -> Optional[QuizSessionResponse]:
        """Get session statistics (no correct answers exposed)"""
        session = await self.get_session(session_id)
        
        if not session:
            return None
        
        questions_count = len(session.questions)
        # Count only questions with wasCorrect != None
        correct_count = sum(
            1 for q in session.questions 
            if q.wasCorrect is not None and q.wasCorrect
        )
        
        return QuizSessionResponse(
            sessionId=session.sessionId,
            materialId=session.materialId,
            questionsCount=questions_count,
            correctCount=correct_count,
            status=session.status,
            createdAt=session.createdAt,
            updatedAt=session.updatedAt
        )
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a quiz session"""
        try:
            result = await self.collection.delete_one({"sessionId": session_id})
            
            if result.deleted_count > 0:
                logger.info(f"✅ Deleted session: {session_id}")
                return True
            
            logger.warning(f"⚠️ Session not found for deletion: {session_id}")
            return False
        
        except Exception as e:
            logger.error(f"❌ Failed to delete session {session_id}: {e}")
            raise Exception(f"Failed to delete session: {str(e)}")