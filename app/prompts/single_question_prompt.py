def build_single_question_prompt(chunk: str, difficulty: str) -> str:
    """
    Build a prompt for generating a single multiple-choice question from a text chunk.
    
    Args:
        chunk: The text content to generate a question from
        difficulty: The difficulty level (e.g., "easy", "medium", "hard")
    
    Returns:
        A formatted prompt string for the LLM
    """
    prompt = f"""You are a quiz question generator. Generate EXACTLY ONE multiple-choice question based on the provided text.

CRITICAL INSTRUCTIONS:
- Return ONLY valid JSON with NO explanations, NO markdown, NO code blocks, NO preamble
- Do NOT wrap the JSON in ```json``` or any other formatting
- Return raw JSON that can be parsed directly

TEXT:
{chunk}

DIFFICULTY: {difficulty}

Generate ONE multiple-choice question with 4 options (A, B, C, D).

Required JSON format (return THIS EXACT STRUCTURE with no other text):
{{
  "question": "Your question text here",
  "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
  "answer": "A",
  "difficulty": "{difficulty}"
}}

The "answer" field must be one of: "A", "B", "C", or "D" (the correct answer).

RETURN ONLY THE JSON OBJECT. NO OTHER TEXT."""

    return prompt