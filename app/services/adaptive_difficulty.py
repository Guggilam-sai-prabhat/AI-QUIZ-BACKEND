"""
Adaptive Difficulty Module
Adjusts question difficulty based on user performance
"""
import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Difficulty levels in order
DifficultyLevel = Literal["easy", "medium", "hard"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


class AdaptiveDifficultyError(Exception):
    """Custom exception for adaptive difficulty errors"""
    pass


def adjust_difficulty(current: str, was_correct: bool) -> str:
    """
    Adjust difficulty level based on user's answer correctness.
    
    Adaptive logic:
    - Correct answer: Move toward harder levels (if possible)
    - Wrong answer: Move toward easier levels (if possible)
    
    Transition rules:
    - easy + correct → medium
    - medium + correct → hard
    - hard + correct → hard (stays at maximum)
    - medium + wrong → easy
    - hard + wrong → medium
    - easy + wrong → easy (stays at minimum)
    
    Args:
        current: Current difficulty level ("easy", "medium", or "hard")
        was_correct: True if the user answered correctly, False otherwise
    
    Returns:
        New difficulty level as a string
    
    Raises:
        AdaptiveDifficultyError: If current difficulty is invalid
    
    Examples:
        >>> adjust_difficulty("easy", True)
        'medium'
        >>> adjust_difficulty("hard", False)
        'medium'
        >>> adjust_difficulty("easy", False)
        'easy'
    """
    # Validate input
    if current not in DIFFICULTY_LEVELS:
        raise AdaptiveDifficultyError(
            f"Invalid difficulty level: '{current}'. "
            f"Must be one of: {DIFFICULTY_LEVELS}"
        )
    
    # Get current level index
    current_index = DIFFICULTY_LEVELS.index(current)
    
    if was_correct:
        # Move up one level (toward harder) if not already at max
        new_index = min(current_index + 1, len(DIFFICULTY_LEVELS) - 1)
        new_difficulty = DIFFICULTY_LEVELS[new_index]
        
        if new_index > current_index:
            logger.info(f"✅ Correct answer! Difficulty increased: {current} → {new_difficulty}")
        else:
            logger.info(f"✅ Correct answer! Difficulty stays at maximum: {current}")
    
    else:
        # Move down one level (toward easier) if not already at min
        new_index = max(current_index - 1, 0)
        new_difficulty = DIFFICULTY_LEVELS[new_index]
        
        if new_index < current_index:
            logger.info(f"❌ Wrong answer. Difficulty decreased: {current} → {new_difficulty}")
        else:
            logger.info(f"❌ Wrong answer. Difficulty stays at minimum: {current}")
    
    return new_difficulty


def get_difficulty_index(difficulty: str) -> int:
    """
    Get the numeric index of a difficulty level.
    
    Args:
        difficulty: Difficulty level string
    
    Returns:
        Index (0=easy, 1=medium, 2=hard)
    
    Raises:
        AdaptiveDifficultyError: If difficulty is invalid
    """
    if difficulty not in DIFFICULTY_LEVELS:
        raise AdaptiveDifficultyError(
            f"Invalid difficulty level: '{difficulty}'. "
            f"Must be one of: {DIFFICULTY_LEVELS}"
        )
    
    return DIFFICULTY_LEVELS.index(difficulty)


def calculate_performance_level(
    correct_count: int,
    total_count: int,
    current_difficulty: str = "medium"
) -> str:
    """
    Calculate recommended difficulty based on overall performance.
    
    Performance thresholds:
    - >= 80% correct → hard
    - >= 60% correct → medium
    - < 60% correct → easy
    
    Args:
        correct_count: Number of correct answers
        total_count: Total number of questions answered
        current_difficulty: Current difficulty level (used as fallback)
    
    Returns:
        Recommended difficulty level
    
    Example:
        >>> calculate_performance_level(8, 10)
        'hard'
        >>> calculate_performance_level(5, 10)
        'medium'
    """
    if total_count == 0:
        logger.warning("⚠️ No questions answered yet, using current difficulty")
        return current_difficulty
    
    accuracy = correct_count / total_count
    
    if accuracy >= 0.8:
        recommended = "hard"
        logger.info(f"🎯 High accuracy ({accuracy:.1%}) → recommending: {recommended}")
    elif accuracy >= 0.6:
        recommended = "medium"
        logger.info(f"🎯 Medium accuracy ({accuracy:.1%}) → recommending: {recommended}")
    else:
        recommended = "easy"
        logger.info(f"🎯 Low accuracy ({accuracy:.1%}) → recommending: {recommended}")
    
    return recommended


def suggest_next_difficulty(
    recent_results: list[bool],
    current_difficulty: str = "medium",
    window_size: int = 5
) -> str:
    """
    Suggest next difficulty based on recent performance window.
    
    Analyzes the last N questions to determine if difficulty should change.
    More responsive than overall performance calculation.
    
    Args:
        recent_results: List of recent answer results (True=correct, False=wrong)
        current_difficulty: Current difficulty level
        window_size: Number of recent questions to consider
    
    Returns:
        Suggested difficulty level
    
    Example:
        >>> suggest_next_difficulty([True, True, True, True, False], "medium")
        'hard'
    """
    if not recent_results:
        logger.warning("⚠️ No recent results, keeping current difficulty")
        return current_difficulty
    
    # Consider only the most recent results
    window = recent_results[-window_size:]
    correct_in_window = sum(window)
    accuracy = correct_in_window / len(window)
    
    logger.debug(
        f"📊 Recent performance: {correct_in_window}/{len(window)} "
        f"({accuracy:.1%}) over last {len(window)} questions"
    )
    
    # More aggressive thresholds for recent performance
    if accuracy >= 0.8:
        suggested = "hard"
    elif accuracy >= 0.5:
        suggested = "medium"
    else:
        suggested = "easy"
    
    if suggested != current_difficulty:
        logger.info(
            f"📈 Performance trend suggests difficulty change: "
            f"{current_difficulty} → {suggested}"
        )
    
    return suggested


# Example usage and testing
def main():
    """Example usage and test cases"""
    print("=== Adaptive Difficulty System ===\n")
    
    # Test 1: Progressive difficulty increase
    print("Test 1: Correct answers progression")
    difficulty = "easy"
    for i in range(5):
        print(f"  Current: {difficulty}, Answer: Correct")
        difficulty = adjust_difficulty(difficulty, was_correct=True)
        print(f"  New difficulty: {difficulty}\n")
    
    # Test 2: Progressive difficulty decrease
    print("\nTest 2: Wrong answers progression")
    difficulty = "hard"
    for i in range(5):
        print(f"  Current: {difficulty}, Answer: Wrong")
        difficulty = adjust_difficulty(difficulty, was_correct=False)
        print(f"  New difficulty: {difficulty}\n")
    
    # Test 3: Performance-based calculation
    print("\nTest 3: Performance-based recommendation")
    test_cases = [
        (9, 10, "High performer"),
        (7, 10, "Medium performer"),
        (4, 10, "Low performer"),
    ]
    
    for correct, total, label in test_cases:
        recommended = calculate_performance_level(correct, total)
        print(f"  {label}: {correct}/{total} → {recommended}")
    
    # Test 4: Recent performance window
    print("\nTest 4: Recent performance analysis")
    recent = [True, True, False, True, True, True, False, True, True, True]
    suggested = suggest_next_difficulty(recent, "medium", window_size=5)
    print(f"  Recent results: {recent[-5:]}")
    print(f"  Suggested difficulty: {suggested}")
    
    # Test 5: Error handling
    print("\nTest 5: Error handling")
    try:
        adjust_difficulty("invalid", True)
    except AdaptiveDifficultyError as e:
        print(f"  ✅ Caught expected error: {e}")


if __name__ == "__main__":
    main()