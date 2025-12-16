"""
Quiz Prompt Builder
Constructs prompts for generating quiz questions from document context
"""
from typing import Tuple


SYSTEM_INSTRUCTION = "Return ONLY valid JSON."

EXAMPLE_SCHEMA = """[
  {
    "question": "What is the primary function of mitochondria?",
    "options": ["Energy production", "Protein synthesis", "Cell division", "Waste removal"],
    "answer": "A"
  },
  {
    "question": "Which process occurs in the chloroplast?",
    "options": ["Respiration", "Photosynthesis", "Fermentation", "Digestion"],
    "answer": "B"
  }
]"""


def build_quiz_prompt(context: str, num_questions: int = 5) -> str:
    """
    Build a prompt for generating quiz questions from document context
    
    Args:
        context: The formatted document context string
        num_questions: Number of questions to generate (default: 5)
    
    Returns:
        A complete prompt string requesting structured JSON output
    """
    prompt = f"""You are a quiz generator. Generate exactly {num_questions} multiple-choice questions based on the provided context.

STRICT FORMATTING RULES:
- Output ONLY a valid JSON array
- Do NOT include markdown code blocks (no ```)
- Do NOT include any explanation, preamble, or additional text
- Do NOT include trailing commas
- Each question must have exactly 4 options
- The "answer" field must be a single letter: "A", "B", "C", or "D"
- Questions must be directly answerable from the context provided

REQUIRED OUTPUT SCHEMA:
{EXAMPLE_SCHEMA}

CONTEXT:
{context}

Generate {num_questions} questions as a JSON array. Output ONLY the JSON array, nothing else."""

    return prompt


def build_quiz_prompt_with_system(
    context: str,
    num_questions: int = 5
) -> Tuple[str, str]:
    """
    Build quiz prompt with separate system and user messages
    
    Useful for chat-based LLM APIs that accept system/user message pairs.
    
    Args:
        context: The formatted document context string
        num_questions: Number of questions to generate (default: 5)
    
    Returns:
        Tuple of (system_message, user_message)
    """
    system_message = SYSTEM_INSTRUCTION
    
    user_message = f"""Generate exactly {num_questions} multiple-choice questions based on the context below.

STRICT FORMATTING RULES:
- Output ONLY a valid JSON array
- Do NOT include markdown code blocks
- Do NOT include any explanation or additional text
- Each question must have exactly 4 options
- The "answer" field must be "A", "B", "C", or "D"
- Questions must be answerable from the context

REQUIRED OUTPUT SCHEMA:
{EXAMPLE_SCHEMA}

CONTEXT:
{context}

Generate {num_questions} questions. Output ONLY the JSON array."""

    return system_message, user_message


def get_system_instruction() -> str:
    """
    Get the system instruction for quiz generation
    
    Returns:
        System instruction string
    """
    return SYSTEM_INSTRUCTION