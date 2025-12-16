"""
Quiz Session Service
MongoDB service functions for quiz sessions with answer evaluation
"""
from datetime import datetime
from typing import Optional, Literal, Tuple
import logging

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.models.quiz_sessions import (
    QuizSession,
    QuestionRecord,
    QuestionMetadata,
    QuizSessionResponse
)

logger = logging.getLogger(__name__)


class QuizSessionService:
    """Service class for quiz session operations"""
    
    COLLECTION_NAME = "quiz_sessions"
    
    def __init__(self, db: AsyncIOMotorDatabase):
        """
        Initialize quiz session service
        
        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.collection = db[self.COLLECTION_NAME]
    
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
            metadata: Complete question metadata including correct answer
        
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
        record: QuestionRecord
    ) -> bool:
        """
        Append a question answer record to session
        
        UPDATED: Supports wasCorrect=None for first question
        
        Args:
            session_id: ID of the quiz session
            record: QuestionRecord with answer result (wasCorrect can be None)
        
        Returns:
            True if successful
        """
        try:
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
                    "N/A (first question)" if record.wasCorrect is None 
                    else str(record.wasCorrect)
                )
                logger.info(
                    f"✅ Appended answer record to session {session_id} - "
                    f"Question: {record.questionId}, Correct: {correctness_str}"
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