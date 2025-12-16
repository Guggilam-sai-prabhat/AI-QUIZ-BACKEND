"""
Quiz Parser
Parses and validates LLM-generated quiz JSON responses
"""
import json
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class QuizParseError(Exception):
    """Base exception for quiz parsing errors"""
    pass


class InvalidJSONError(QuizParseError):
    """Raised when JSON cannot be parsed even after cleanup"""
    pass


class ValidationError(QuizParseError):
    """Raised when quiz structure validation fails"""
    pass


VALID_ANSWERS = {"A", "B", "C", "D"}
REQUIRED_FIELDS = {"question", "options", "answer"}
REQUIRED_OPTIONS_COUNT = 4


def _strip_markdown(text: str) -> str:
    """
    Remove markdown code block formatting
    
    Args:
        text: Raw text possibly containing markdown
    
    Returns:
        Text with markdown code blocks removed
    """
    # Remove ```json ... ``` or ``` ... ```
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    
    # Remove standalone ``` markers
    text = re.sub(r"```", "", text)
    
    return text.strip()


def _extract_json_array(text: str) -> str:
    """
    Extract JSON array from text by finding outermost brackets
    
    Args:
        text: Text containing a JSON array
    
    Returns:
        Extracted JSON array string
    
    Raises:
        InvalidJSONError: If no valid array brackets found
    """
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")
    
    if first_bracket == -1 or last_bracket == -1:
        raise InvalidJSONError("No JSON array found in response")
    
    if first_bracket >= last_bracket:
        raise InvalidJSONError("Invalid JSON array brackets")
    
    return text[first_bracket:last_bracket + 1]


def _fix_common_json_issues(text: str) -> str:
    """
    Fix common JSON formatting issues from LLM output
    
    Args:
        text: JSON string with potential issues
    
    Returns:
        Cleaned JSON string
    """
    # Remove trailing commas before ] or }
    text = re.sub(r",\s*([}\]])", r"\1", text)
    
    # Fix single quotes to double quotes (but not within strings)
    # This is a simple fix - may not work for all edge cases
    text = re.sub(r"(?<![\\])\'", '"', text)
    
    # Remove any control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    
    # Fix unescaped newlines within strings (replace with space)
    # This handles cases where LLM puts actual newlines in question text
    lines = text.split("\n")
    text = " ".join(line.strip() for line in lines if line.strip())
    
    # Re-add proper JSON formatting
    text = text.replace("{ ", "{").replace(" }", "}")
    text = text.replace("[ ", "[").replace(" ]", "]")
    
    return text


def _clean_response(raw_response: str) -> str:
    """
    Apply all cleanup rules to extract valid JSON
    
    Args:
        raw_response: Raw LLM response string
    
    Returns:
        Cleaned JSON string ready for parsing
    """
    text = raw_response.strip()
    
    # Step 1: Strip markdown formatting
    text = _strip_markdown(text)
    
    # Step 2: Extract JSON array
    text = _extract_json_array(text)
    
    # Step 3: Fix common issues
    text = _fix_common_json_issues(text)
    
    return text


def _validate_question(item: Dict[str, Any], index: int) -> None:
    """
    Validate a single quiz question
    
    Args:
        item: Question dictionary
        index: Question index for error messages
    
    Raises:
        ValidationError: If validation fails
    """
    # Check required fields
    missing = REQUIRED_FIELDS - set(item.keys())
    if missing:
        raise ValidationError(
            f"Question {index + 1}: Missing required fields: {missing}"
        )
    
    # Validate question text
    if not isinstance(item["question"], str) or not item["question"].strip():
        raise ValidationError(
            f"Question {index + 1}: 'question' must be a non-empty string"
        )
    
    # Validate options
    options = item["options"]
    if not isinstance(options, list):
        raise ValidationError(
            f"Question {index + 1}: 'options' must be a list"
        )
    
    if len(options) != REQUIRED_OPTIONS_COUNT:
        raise ValidationError(
            f"Question {index + 1}: Expected {REQUIRED_OPTIONS_COUNT} options, "
            f"got {len(options)}"
        )
    
    for i, opt in enumerate(options):
        if not isinstance(opt, str) or not opt.strip():
            raise ValidationError(
                f"Question {index + 1}: Option {i + 1} must be a non-empty string"
            )
    
    # Validate answer
    answer = item["answer"]
    if not isinstance(answer, str):
        raise ValidationError(
            f"Question {index + 1}: 'answer' must be a string"
        )
    
    answer_upper = answer.strip().upper()
    if answer_upper not in VALID_ANSWERS:
        raise ValidationError(
            f"Question {index + 1}: 'answer' must be A, B, C, or D. Got: '{answer}'"
        )


def _normalize_question(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a validated question to consistent format
    
    Args:
        item: Validated question dictionary
    
    Returns:
        Normalized question dictionary
    """
    return {
        "question": item["question"].strip(),
        "options": [opt.strip() for opt in item["options"]],
        "answer": item["answer"].strip().upper()
    }


def parse_quiz_json(raw_response: str) -> List[Dict[str, Any]]:
    """
    Parse and validate quiz JSON from LLM response
    
    Attempts direct JSON parsing first, then applies cleanup rules
    if initial parsing fails.
    
    Args:
        raw_response: Raw string response from LLM
    
    Returns:
        List of validated quiz question dictionaries, each containing:
            - question: str
            - options: List[str] (exactly 4 items)
            - answer: str (A, B, C, or D)
    
    Raises:
        InvalidJSONError: If JSON cannot be parsed after cleanup
        ValidationError: If quiz structure is invalid
    
    Example:
        >>> raw = '[{"question": "What is 2+2?", "options": ["3","4","5","6"], "answer": "B"}]'
        >>> questions = parse_quiz_json(raw)
        >>> print(questions[0]["answer"])
        'B'
    """
    if not raw_response or not raw_response.strip():
        raise InvalidJSONError("Empty response received")
    
    logger.debug(f"Parsing quiz response ({len(raw_response)} chars)")
    
    # Attempt 1: Direct JSON parse
    try:
        data = json.loads(raw_response.strip())
        logger.debug("Direct JSON parse successful")
    except json.JSONDecodeError as e:
        logger.debug(f"Direct parse failed: {e}. Attempting cleanup...")
        
        # Attempt 2: Clean and retry
        try:
            cleaned = _clean_response(raw_response)
            data = json.loads(cleaned)
            logger.debug("JSON parse successful after cleanup")
        except json.JSONDecodeError as e2:
            logger.error(f"JSON parse failed after cleanup: {e2}")
            raise InvalidJSONError(
                f"Failed to parse JSON: {e2}. "
                f"Original error: {e}"
            )
    
    # Validate it's a list
    if not isinstance(data, list):
        raise ValidationError(
            f"Expected a JSON array, got {type(data).__name__}"
        )
    
    if len(data) == 0:
        raise ValidationError("Quiz array is empty")
    
    # Validate and normalize each question
    validated_questions = []
    
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValidationError(
                f"Question {idx + 1}: Expected object, got {type(item).__name__}"
            )
        
        _validate_question(item, idx)
        normalized = _normalize_question(item)
        validated_questions.append(normalized)
    
    logger.info(f"✅ Successfully parsed {len(validated_questions)} quiz questions")
    
    return validated_questions


def safe_parse_quiz_json(raw_response: str) -> tuple[List[Dict[str, Any]], str | None]:
    """
    Parse quiz JSON with error capture instead of raising
    
    Args:
        raw_response: Raw string response from LLM
    
    Returns:
        Tuple of (questions_list, error_message)
        - On success: (questions, None)
        - On failure: ([], error_message)
    """
    try:
        questions = parse_quiz_json(raw_response)
        return questions, None
    except QuizParseError as e:
        logger.error(f"❌ Quiz parse error: {e}")
        return [], str(e)