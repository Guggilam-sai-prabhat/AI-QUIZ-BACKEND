"""
Quiz Database Operations
MongoDB CRUD operations for quiz attempts
FILE: app/db/quiz_db.py
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from bson import ObjectId
import logging

from app.db.mongodb import get_collection
from app.models.quiz import QuizAttemptDocument

logger = logging.getLogger(__name__)


async def insert_quiz_attempt(
    quiz_id: str,
    user_answers: List[str],
    correct_answers: List[str]
) -> Dict[str, Any]:
    """
    Insert a new quiz attempt into MongoDB
    
    Args:
        quiz_id: Unique identifier for the quiz
        user_answers: List of answers submitted by user
        correct_answers: List of correct answers for comparison
    
    Returns:
        Dictionary containing:
        - id: MongoDB ObjectId as string
        - score: Number of correct answers
        - total: Total number of questions
        - percentage: Score as percentage
        - submitted_at: Submission timestamp
    
    Raises:
        ValueError: If answers lists are empty or mismatched length
        Exception: If database operation fails
    """
    try:
        # Validate inputs
        if not user_answers or not correct_answers:
            raise ValueError("Answers lists cannot be empty")
        
        if len(user_answers) != len(correct_answers):
            raise ValueError(
                f"Answer count mismatch: {len(user_answers)} user answers "
                f"vs {len(correct_answers)} correct answers"
            )
        
        # Calculate score
        total = len(correct_answers)
        score = sum(
            1 for user_ans, correct_ans in zip(user_answers, correct_answers)
            if user_ans == correct_ans
        )
        percentage = round((score / total) * 100, 2) if total > 0 else 0.0
        
        # Create document
        attempt_doc = QuizAttemptDocument(
            quiz_id=quiz_id,
            user_answers=user_answers,
            correct_answers=correct_answers,
            score=score,
            total=total,
            percentage=percentage,
            submitted_at=datetime.now(timezone.utc)
        )
        
        # Insert into MongoDB
        collection = get_collection("quiz_attempts")
        result = await collection.insert_one(
            attempt_doc.model_dump(by_alias=True)
        )
        
        attempt_id = str(result.inserted_id)
        
        logger.info(
            f"✅ Quiz attempt saved: {attempt_id} "
            f"(quiz={quiz_id}, score={score}/{total}, {percentage}%)"
        )
        
        return {
            "id": attempt_id,
            "quiz_id": quiz_id,
            "score": score,
            "total": total,
            "percentage": percentage,
            "submitted_at": attempt_doc.submitted_at
        }
        
    except ValueError as ve:
        logger.error(f"❌ Validation error: {ve}")
        raise
    
    except Exception as e:
        logger.error(f"❌ Failed to insert quiz attempt: {e}")
        raise Exception(f"Database error: {str(e)}")


async def get_quiz_attempt(attempt_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a quiz attempt by ID
    
    Args:
        attempt_id: MongoDB ObjectId as string
    
    Returns:
        Quiz attempt document or None if not found
    """
    try:
        collection = get_collection("quiz_attempts")
        attempt = await collection.find_one({"_id": ObjectId(attempt_id)})
        
        if not attempt:
            logger.warning(f"⚠️ Quiz attempt not found: {attempt_id}")
            return None
        
        # Convert ObjectId to string
        attempt["_id"] = str(attempt["_id"])
        
        return attempt
        
    except Exception as e:
        logger.error(f"❌ Failed to get quiz attempt {attempt_id}: {e}")
        return None


async def get_quiz_attempts_by_quiz(
    quiz_id: str,
    skip: int = 0,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get all attempts for a specific quiz
    
    Args:
        quiz_id: Quiz identifier
        skip: Number of records to skip (pagination)
        limit: Maximum number of records to return
    
    Returns:
        List of quiz attempt documents
    """
    try:
        collection = get_collection("quiz_attempts")
        
        cursor = collection.find(
            {"quiz_id": quiz_id}
        ).sort("submitted_at", -1).skip(skip).limit(limit)
        
        attempts = []
        async for attempt in cursor:
            attempt["_id"] = str(attempt["_id"])
            attempts.append(attempt)
        
        logger.info(f"📊 Retrieved {len(attempts)} attempts for quiz {quiz_id}")
        return attempts
        
    except Exception as e:
        logger.error(f"❌ Failed to get quiz attempts: {e}")
        return []


async def get_all_quiz_attempts(
    skip: int = 0,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get all quiz attempts with pagination
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
    
    Returns:
        List of quiz attempt documents
    """
    try:
        collection = get_collection("quiz_attempts")
        
        cursor = collection.find().sort("submitted_at", -1).skip(skip).limit(limit)
        
        attempts = []
        async for attempt in cursor:
            attempt["_id"] = str(attempt["_id"])
            attempts.append(attempt)
        
        logger.info(f"📊 Retrieved {len(attempts)} quiz attempts")
        return attempts
        
    except Exception as e:
        logger.error(f"❌ Failed to get all quiz attempts: {e}")
        return []


async def count_quiz_attempts(quiz_id: Optional[str] = None) -> int:
    """
    Count quiz attempts
    
    Args:
        quiz_id: Optional quiz filter (if None, counts all attempts)
    
    Returns:
        Total count of attempts
    """
    try:
        collection = get_collection("quiz_attempts")
        
        if quiz_id:
            count = await collection.count_documents({"quiz_id": quiz_id})
        else:
            count = await collection.count_documents({})
        
        return count
        
    except Exception as e:
        logger.error(f"❌ Failed to count quiz attempts: {e}")
        return 0


async def get_quiz_statistics(quiz_id: str) -> Dict[str, Any]:
    """
    Get statistical summary for a quiz
    
    Args:
        quiz_id: Quiz identifier
    
    Returns:
        Dictionary with statistics:
        - total_attempts: Number of attempts
        - average_score: Average score across all attempts
        - highest_score: Highest score achieved
        - lowest_score: Lowest score achieved
        - average_percentage: Average percentage across attempts
    """
    try:
        collection = get_collection("quiz_attempts")
        
        pipeline = [
            {"$match": {"quiz_id": quiz_id}},
            {
                "$group": {
                    "_id": None,
                    "total_attempts": {"$sum": 1},
                    "average_score": {"$avg": "$score"},
                    "highest_score": {"$max": "$score"},
                    "lowest_score": {"$min": "$score"},
                    "average_percentage": {"$avg": "$percentage"},
                    "total_questions": {"$first": "$total"}
                }
            }
        ]
        
        cursor = collection.aggregate(pipeline)
        result = await cursor.to_list(length=1)
        
        if not result:
            return {
                "quiz_id": quiz_id,
                "total_attempts": 0,
                "average_score": 0.0,
                "highest_score": 0,
                "lowest_score": 0,
                "average_percentage": 0.0
            }
        
        stats = result[0]
        stats.pop("_id", None)
        stats["quiz_id"] = quiz_id
        
        # Round averages
        stats["average_score"] = round(stats.get("average_score", 0.0), 2)
        stats["average_percentage"] = round(stats.get("average_percentage", 0.0), 2)
        
        logger.info(f"📊 Statistics for quiz {quiz_id}: {stats['total_attempts']} attempts")
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Failed to get quiz statistics: {e}")
        return {
            "quiz_id": quiz_id,
            "total_attempts": 0,
            "error": str(e)
        }


async def delete_quiz_attempt(attempt_id: str) -> bool:
    """
    Delete a quiz attempt by ID
    
    Args:
        attempt_id: MongoDB ObjectId as string
    
    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        collection = get_collection("quiz_attempts")
        result = await collection.delete_one({"_id": ObjectId(attempt_id)})
        
        if result.deleted_count > 0:
            logger.info(f"🗑️ Deleted quiz attempt: {attempt_id}")
            return True
        else:
            logger.warning(f"⚠️ Quiz attempt not found: {attempt_id}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Failed to delete quiz attempt: {e}")
        return False


async def delete_quiz_attempts_by_quiz(quiz_id: str) -> int:
    """
    Delete all attempts for a specific quiz
    
    Args:
        quiz_id: Quiz identifier
    
    Returns:
        Number of attempts deleted
    """
    try:
        collection = get_collection("quiz_attempts")
        result = await collection.delete_many({"quiz_id": quiz_id})
        
        deleted_count = result.deleted_count
        logger.info(f"🗑️ Deleted {deleted_count} attempts for quiz {quiz_id}")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"❌ Failed to delete quiz attempts: {e}")
        return 0