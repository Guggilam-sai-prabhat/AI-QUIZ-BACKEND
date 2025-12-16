"""
Difficulty rating service for learning material chunks.
Provides both rule-based and LLM-based difficulty assessment.
"""

import re
import logging
from typing import Literal

from app.services.llm_client import generate_quiz, LLMClientError, get_available_providers

logger = logging.getLogger(__name__)


def rate_chunk_difficulty_rule_based(chunk: str) -> Literal["easy", "medium", "hard"]:
    """
    Rule-based difficulty rating as a fallback method.
    
    Criteria:
    - Chunk length
    - Average word length
    - Sentence complexity
    
    Args:
        chunk: The text chunk to rate
        
    Returns:
        Difficulty level: "easy", "medium", or "hard"
    """
    if not chunk or not chunk.strip():
        return "easy"
    
    # Calculate metrics
    chunk = chunk.strip()
    words = chunk.split()
    word_count = len(words)
    
    # Count sentences (simple approximation)
    sentences = re.split(r'[.!?]+', chunk)
    sentences = [s for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)
    
    # Average word length
    avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
    
    # Words per sentence
    words_per_sentence = word_count / sentence_count
    
    # Scoring system
    difficulty_score = 0
    
    # Length-based scoring
    if word_count < 50:
        difficulty_score += 0
    elif word_count < 150:
        difficulty_score += 1
    else:
        difficulty_score += 2
    
    # Word complexity scoring
    if avg_word_length < 4.5:
        difficulty_score += 0
    elif avg_word_length < 6.0:
        difficulty_score += 1
    else:
        difficulty_score += 2
    
    # Sentence complexity scoring
    if words_per_sentence < 12:
        difficulty_score += 0
    elif words_per_sentence < 20:
        difficulty_score += 1
    else:
        difficulty_score += 2
    
    # Map score to difficulty level
    if difficulty_score <= 2:
        return "easy"
    elif difficulty_score <= 4:
        return "medium"
    else:
        return "hard"


async def rate_chunk_difficulty_llm(
    chunk: str,
    provider: str = "openai"
) -> Literal["easy", "medium", "hard"]:
    """
    LLM-based difficulty rating using configured LLM provider.
    
    Uses your existing LLM client infrastructure to assess the difficulty
    of learning material based on complexity, technical content, and cognitive load.
    
    Args:
        chunk: The text chunk to rate
        provider: LLM provider to use ("openai", "anthropic", or "grok")
        
    Returns:
        Difficulty level: "easy", "medium", or "hard"
        
    Raises:
        LLMClientError: If API is not configured
        ValueError: If LLM returns invalid response
    """
    if not chunk or not chunk.strip():
        return "easy"
    
    # Prepare the prompt
    prompt = f"""Rate the difficulty of this learning material on a scale: easy, medium, hard. Return ONLY one word.

Learning material:
{chunk}"""
    
    try:
        # Use existing LLM client infrastructure
        response_text = await generate_quiz(prompt, provider=provider, timeout=30.0)
        
        # Extract and validate response
        response_text = response_text.strip().lower()
        
        # Remove any JSON formatting if present
        if response_text.startswith('"') and response_text.endswith('"'):
            response_text = response_text[1:-1]
        
        # Validate and return
        if response_text in ["easy", "medium", "hard"]:
            logger.info(f"✅ LLM difficulty rating: {response_text}")
            return response_text  # type: ignore
        
        # Fallback if response contains the word but has extra text
        if "easy" in response_text:
            logger.warning(f"⚠️ Parsing 'easy' from LLM response: {response_text}")
            return "easy"
        elif "medium" in response_text:
            logger.warning(f"⚠️ Parsing 'medium' from LLM response: {response_text}")
            return "medium"
        elif "hard" in response_text:
            logger.warning(f"⚠️ Parsing 'hard' from LLM response: {response_text}")
            return "hard"
        else:
            raise ValueError(f"Unexpected LLM response: {response_text}")
            
    except Exception as e:
        logger.error(f"❌ LLM difficulty rating failed: {e}")
        raise


async def rate_chunk_difficulty(
    chunk: str,
    provider: str = "openai"
) -> Literal["easy", "medium", "hard"]:
    """
    Rate the difficulty of a text chunk.
    
    Attempts to use LLM-based rating if available, otherwise falls back
    to rule-based rating.
    
    Args:
        chunk: The text chunk to rate
        provider: LLM provider to use (default: "openai")
        
    Returns:
        Difficulty level: "easy", "medium", or "hard"
    """
    # Check if any LLM provider is configured
    available_providers = get_available_providers()
    print(available_providers , "available providers")
    
    if not available_providers:
        logger.info("ℹ️ No LLM provider configured, using rule-based difficulty rating")
        return rate_chunk_difficulty_rule_based(chunk)
    
    # Use specified provider if available, otherwise use first available
    if provider not in available_providers:
        if available_providers:
            provider = available_providers[0]
            logger.info(f"ℹ️ Provider '{provider}' not available, using '{provider}' instead")
        else:
            logger.info("ℹ️ No LLM provider available, using rule-based difficulty rating")
            return rate_chunk_difficulty_rule_based(chunk)
    
    try:
        # Try LLM-based rating first
        return await rate_chunk_difficulty_llm(chunk, provider=provider)
    except (LLMClientError, ValueError, Exception) as e:
        # Fallback to rule-based rating
        logger.warning(f"⚠️ LLM rating failed, falling back to rule-based: {e}")
        return rate_chunk_difficulty_rule_based(chunk)


# Synchronous wrapper for convenience
def rate_chunk_difficulty_sync(chunk: str) -> Literal["easy", "medium", "hard"]:
    """
    Synchronous version that only uses rule-based rating.
    
    Args:
        chunk: The text chunk to rate
        
    Returns:
        Difficulty level: "easy", "medium", or "hard"
    """
    return rate_chunk_difficulty_rule_based(chunk)