"""
Next Question Service - Full MCP Orchestration
LLM orchestrates ALL retrieval, difficulty, generation, and session logic via MCP tools
Backend only forwards tool calls - NO logic injection
"""
import json
import logging
import os
from typing import Dict, Any, Optional, List
from motor.motor_asyncio import AsyncIOMotorDatabase
import openai

from app.models.quiz_sessions import QuestionMetadata, QuestionRecord
from app.services.quiz_session_service import QuizSessionService
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class NextQuestionServiceError(Exception):
    """Base exception for next question service errors"""
    pass


class SessionNotFoundError(NextQuestionServiceError):
    """Session not found error"""
    pass


class MaterialMismatchError(NextQuestionServiceError):
    """Material ID mismatch error"""
    pass


class NoContentAvailableError(NextQuestionServiceError):
    """No content available for difficulty error"""
    pass


class MCPOrchestrationError(NextQuestionServiceError):
    """MCP orchestration error"""
    pass


class NextQuestionService:
    """
    Service for adaptive next question generation via full MCP orchestration.
    
    The LLM orchestrates EVERYTHING:
    - Difficulty adaptation via compute_next_difficulty tool
    - Content retrieval via get_chunk_by_difficulty tool
    - MCQ generation strictly from retrieved chunk (NO fabrication)
    - Session recording via record_question_event tool
    
    Backend role: Forward tool calls and return results (NO logic injection)
    """
    
    # MCP Tool Definitions
    MCP_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "compute_next_difficulty",
                "description": (
                    "Calculate the next difficulty level based on current difficulty "
                    "and whether the user answered correctly. Implements adaptive "
                    "difficulty algorithm. You MUST call this for non-first questions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "currentDifficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                            "description": "Current difficulty level"
                        },
                        "wasCorrect": {
                            "type": "boolean",
                            "description": "Whether the user's answer was correct"
                        }
                    },
                    "required": ["currentDifficulty", "wasCorrect"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_chunk_by_difficulty",
                "description": (
                    "Retrieve a random text chunk from study material at the specified "
                    "difficulty level. You MUST call this tool to get content BEFORE "
                    "generating any question. The returned chunk is your ONLY source material."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "materialId": {
                            "type": "string",
                            "description": "Unique identifier of the study material"
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                            "description": "Difficulty level to filter chunks"
                        }
                    },
                    "required": ["materialId", "difficulty"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "record_question_event",
                "description": (
                    "Record a question answer event in the quiz session history. "
                    "Call this for non-first questions to log the previous answer. "
                    "Stores the question result and updates session."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sessionId": {
                            "type": "string",
                            "description": "Quiz session identifier"
                        },
                        "questionId": {
                            "type": "string",
                            "description": "Unique identifier for the question"
                        },
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                            "description": "Difficulty level of the question"
                        },
                        "wasCorrect": {
                            "type": "boolean",
                            "description": "Whether the user answered correctly"
                        }
                    },
                    "required": ["sessionId", "questionId", "difficulty", "wasCorrect"]
                }
            }
        }
    ]
    

    SYSTEM_PROMPT = """You are a quiz question generator that uses tools to retrieve content and generate questions.

## YOUR ONLY JOB:

1. Call the required tools in order
2. Wait for tool responses
3. Generate a question from the retrieved chunk
4. Return valid JSON with the ACTUAL difficulty from the tool response

## FOR FIRST QUESTION:

Step 1: Call get_chunk_by_difficulty(materialId, difficulty="medium")
Step 2: Wait for chunk text
   - The tool may return a fallback chunk at a different difficulty
   - If "fallback": true, note the actual "difficulty" field
Step 3: Create MCQ from chunk
Step 4: Return JSON using the ACTUAL difficulty from the tool response:
{
  "questionId": "<uuid>",
  "question": "<question text>?",
  "options": ["A text", "B text", "C text", "D text"],
  "answer": "A",
  "difficulty": "<USE DIFFICULTY FROM TOOL RESPONSE>"
}

## FOR NEXT QUESTION:

Step 1: Call compute_next_difficulty(currentDifficulty, wasCorrect)
Step 2: Call get_chunk_by_difficulty(materialId, difficulty=<result from step 1>)
   - The tool may return a fallback chunk at a different difficulty
   - If "fallback": true, note the actual "difficulty" field
Step 3: Call record_question_event(sessionId, questionId, difficulty, wasCorrect)
Step 4: Create MCQ from chunk
Step 5: Return JSON using the ACTUAL difficulty from the tool response

## DIFFICULTY HANDLING:

**CRITICAL:** Always use the "difficulty" value from the tool response, NOT the requested difficulty.

Example:
- You request: difficulty="medium"
- Tool returns: {
    "text": "...",
    "difficulty": "easy",
    "fallback": true,
    "fallbackMessage": "Using 'easy' difficulty ('medium' not available)"
  }
- Your JSON should have: "difficulty": "easy"

## RULES:

✅ ALWAYS call tools before generating questions
✅ ALWAYS use the chunk text as your ONLY source
✅ ALWAYS use the difficulty from the tool response
✅ ALWAYS return clean JSON (NO ```json wrappers)
✅ If tool returns "fallback": true, use it anyway and use its difficulty
✅ If tool returns "error": "no_content_available", return:
{
  "error": "no_content_available",
  "message": "No content available"
}

❌ NEVER use requested difficulty - always use actual difficulty from tool
❌ NEVER generate questions without calling get_chunk_by_difficulty first
❌ NEVER add markdown backticks around JSON
❌ NEVER add explanatory text with the JSON

## EXAMPLES:

**Example 1: Successful retrieval at requested difficulty**
TOOL RESPONSE: {
  "chunkId": "123",
  "text": "Python is a programming language...",
  "difficulty": "medium",
  "fallback": false
}
YOUR JSON: {
  "questionId": "550e8400-e29b-41d4-a716-446655440000",
  "question": "What is Python?",
  "options": ["A language", "A snake", "A framework", "A database"],
  "answer": "A",
  "difficulty": "medium"
}

**Example 2: Fallback to different difficulty**
TOOL RESPONSE: {
  "chunkId": "456",
  "text": "Variables store data...",
  "difficulty": "easy",
  "requestedDifficulty": "medium",
  "fallback": true,
  "fallbackMessage": "Using 'easy' difficulty ('medium' not available)"
}
YOUR JSON: {
  "questionId": "550e8400-e29b-41d4-a716-446655440001",
  "question": "What do variables do?",
  "options": ["Store data", "Run code", "Delete files", "Print text"],
  "answer": "A",
  "difficulty": "easy"  ← USE "easy" from tool response, NOT "medium"
}

**Example 3: No content available**
TOOL RESPONSE: {
  "error": "no_content_available",
  "message": "No content available for material..."
}
YOUR JSON: {
  "error": "no_content_available",
  "message": "No content available"
}

Remember: The difficulty in your JSON must ALWAYS match the difficulty field from the tool response."""
    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        qdrant_service: QdrantService,
        openai_model: str = "gpt-4o",
        max_iterations: int = 10
    ):
        """
        Initialize next question service
        
        Args:
            db: MongoDB database instance
            qdrant_service: Qdrant service for content retrieval
            openai_model: OpenAI model to use (default: gpt-4o)
            max_iterations: Maximum tool call iterations
        """
        self.db = db
        self.qdrant_service = qdrant_service
        self.session_service = QuizSessionService(db)
        self.openai_model = openai_model
        self.max_iterations = max_iterations
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.openai_client = openai.AsyncOpenAI(api_key=api_key)
    
    async def generate_next_question(
        self,
        session_id: str,
        material_id: str,
        current_difficulty: str,
        is_first: bool,
        was_correct: Optional[bool] = None,
        previous_question_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate next question via full MCP orchestration.
        
        The LLM handles ALL logic via tools:
        - Difficulty computation (if not first)
        - Chunk retrieval
        - MCQ generation (strictly from chunk)
        - Session recording (if not first)
        
        Args:
            session_id: Quiz session ID
            material_id: Study material ID
            current_difficulty: Current difficulty level
            is_first: Whether this is the first question
            was_correct: Whether previous answer was correct (required if not first)
            previous_question_id: ID of previous question (required if not first)
        
        Returns:
            Dictionary with question data (WITHOUT correct answer for API)
        
        Raises:
            NextQuestionServiceError: If generation fails
        """
        logger.info(
            f"🎯 Starting {'FIRST' if is_first else 'NEXT'} question via full MCP - "
            f"Session: {session_id}, Material: {material_id}, "
            f"Difficulty: {current_difficulty}"
            + ("" if is_first else f", Correct: {was_correct}")
        )
        
        # Validate session
        await self._validate_session(session_id, material_id)
        
        # Validate subsequent question requirements
        if not is_first:
            if was_correct is None:
                raise NextQuestionServiceError(
                    "was_correct is required for subsequent questions"
                )
            if not previous_question_id:
                raise NextQuestionServiceError(
                    "previous_question_id is required for subsequent questions"
                )
        
        # Build user prompt
        user_prompt = self._build_user_prompt(
            session_id=session_id,
            material_id=material_id,
            current_difficulty=current_difficulty,
            is_first=is_first,
            was_correct=was_correct,
            previous_question_id=previous_question_id
        )
        
        # Orchestrate with LLM
        question_data = await self._orchestrate_with_llm(
            user_prompt=user_prompt,
            session_id=session_id,
            material_id=material_id
        )
        
        # Store complete question metadata in session (includes correct answer)
       
        await self._store_question_metadata(session_id, question_data)

        # Extract difficulty from question data
        new_difficulty = question_data.get("difficulty", current_difficulty)

        # Determine if difficulty changed
        difficulty_changed = new_difficulty != current_difficulty

        # Build API response WITHOUT correct answer
        response = {
            "questionId": question_data["questionId"],
            "question": question_data["question"],
            "options": question_data["options"],
            "difficulty": new_difficulty,
            "difficultyChanged": difficulty_changed,  # ADD THIS LINE
            "previousDifficulty": current_difficulty
        }

        logger.info(
            f"✅ Successfully generated question via MCP - "
            f"ID: {response['questionId']}, "
            f"Difficulty: {new_difficulty} "
            f"({'changed' if difficulty_changed else 'unchanged'} from {current_difficulty})"
        )

        return response
    
    def _build_user_prompt(
        self,
        session_id: str,
        material_id: str,
        current_difficulty: str,
        is_first: bool,
        was_correct: Optional[bool],
        previous_question_id: Optional[str]
    ) -> str:
        """Build user prompt for LLM orchestration"""
        
        if is_first:
            return f"""Generate the FIRST question for this quiz session.

    **Session Context:**
    - Session ID: {session_id}
    - Material ID: {material_id}
    - Preferred Starting Difficulty: medium
    - Is First Question: true

    **Your Task:**

    1. Call get_chunk_by_difficulty(materialId="{material_id}", difficulty="medium")

    2. Wait for the tool response:
    - If you receive a chunk (with or without fallback flag), proceed to step 3
    - The tool will automatically try other difficulties if "medium" is not available
    - ALWAYS use the "difficulty" value from the tool response in your final JSON

    3. Generate a multiple-choice question from the chunk

    4. Return ONLY this JSON (no markdown, no explanation):
    {{
    "questionId": "<generate-uuid-v4>",
    "question": "Your question here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "A",
    "difficulty": "<USE THE 'difficulty' FIELD FROM TOOL RESPONSE>"
    }}

    **CRITICAL:**
    - The difficulty in your JSON must match the difficulty from the tool response
    - If tool returns difficulty="easy", your JSON must have difficulty="easy"
    - If tool returns difficulty="hard", your JSON must have difficulty="hard"
    - DO NOT hardcode difficulty="medium" if the tool returned something else

    Begin now."""
        
        else:
            return f"""Generate the NEXT question for this quiz session.

    **Session Context:**
    - Session ID: {session_id}
    - Material ID: {material_id}
    - Previous Difficulty: {current_difficulty}
    - Previous Question ID: {previous_question_id}
    - User Was Correct: {was_correct}

    **Your Task:**

    1. Call compute_next_difficulty(currentDifficulty="{current_difficulty}", wasCorrect={str(was_correct).lower()})

    2. Call get_chunk_by_difficulty(materialId="{material_id}", difficulty=<nextDifficulty from step 1>)
    - The tool will automatically try other difficulties if needed
    - ALWAYS use the "difficulty" value from the tool response

    3. Call record_question_event(sessionId="{session_id}", questionId="{previous_question_id}", difficulty="{current_difficulty}", wasCorrect={str(was_correct).lower()})

    4. Generate a multiple-choice question from the chunk

    5. Return ONLY this JSON (no markdown, no explanation):
    {{
    "questionId": "<generate-uuid-v4>",
    "question": "Your question here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "A",
    "difficulty": "<USE THE 'difficulty' FIELD FROM TOOL RESPONSE>"
    }}

    **CRITICAL:**
    - Use the difficulty from the get_chunk_by_difficulty response
    - If it returns "easy" due to fallback, use "easy" in your JSON
    - Complete all 5 steps before returning JSON

    Begin now."""
    
    async def _orchestrate_with_llm(
        self,
        user_prompt: str,
        session_id: str,
        material_id: str
    ) -> Dict[str, Any]:
        """
        Orchestrate question generation with LLM via MCP tools.
        Handles tool call loop until final JSON response.
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            iteration = 0

            while iteration < self.max_iterations:
                iteration += 1
                logger.info(f"🔄 MCP orchestration iteration {iteration}")

                # Call OpenAI
                response = await self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    tools=self.MCP_TOOLS,
                    tool_choice="auto",
                    temperature=0.7
                )

                assistant_message = response.choices[0].message

                # ✅ FINAL RESPONSE (no tool calls)
                if not assistant_message.tool_calls:
                    content = assistant_message.content
                    if not content:
                        raise MCPOrchestrationError("No content in final response")

                    content_stripped = content.strip()

                    # Handle structured LLM error responses
                    if content_stripped.startswith("{") and (
                        '"error"' in content_stripped or '"message"' in content_stripped
                    ):
                        try:
                            response_obj = json.loads(content_stripped)
                            if "error" in response_obj:
                                error_type = response_obj.get("error")
                                error_message = response_obj.get("message", "Unknown error")

                                if error_type == "no_content_available":
                                    raise NoContentAvailableError(error_message)
                                elif error_type == "material_mismatch":
                                    raise MaterialMismatchError(error_message)
                                else:
                                    raise MCPOrchestrationError(
                                        f"LLM returned error: {error_message}"
                                    )
                        except json.JSONDecodeError:
                            pass  # Not valid JSON → continue normal parsing

                    # Parse & validate final MCQ JSON
                    question_data = self._parse_json_response(content)

                    logger.info(
                        f"✅ LLM returned final MCQ: {question_data['questionId']} "
                        f"(difficulty: {question_data.get('difficulty', 'unknown')})"
                    )
                    return question_data

                # 🔧 TOOL CALL PROCESSING
                logger.info(
                    f"🔧 Processing {len(assistant_message.tool_calls)} tool calls"
                )

                # Add assistant tool-call message
                messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })

                # Execute each MCP tool
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    logger.info(
                        f"🛠️ Executing tool: {function_name} "
                        f"with args: {function_args}"
                    )

                    tool_result = await self._execute_mcp_tool(
                        function_name=function_name,
                        arguments=function_args,
                        session_id=session_id,
                        material_id=material_id
                    )

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_result)
                    })

                    logger.info(f"✅ Tool {function_name} executed successfully")

            raise MCPOrchestrationError(
                f"Max iterations ({self.max_iterations}) reached without final response"
            )

        except openai.OpenAIError as e:
            logger.error(f"❌ OpenAI API error: {e}")
            raise MCPOrchestrationError(f"OpenAI API error: {str(e)}")

        except Exception as e:
            logger.error(f"❌ Orchestration error: {e}")
            raise MCPOrchestrationError(f"Orchestration failed: {str(e)}")

    

    async def _execute_mcp_tool(
        self,
        function_name: str,
        arguments: Dict[str, Any],
        session_id: str,
        material_id: str
    ) -> Dict[str, Any]:
        """
        Execute MCP tool and return result.
        Backend only forwards tool calls - NO logic injection.
        UPDATED: Added intelligent fallback for missing chunks
        """
        
        try:
            if function_name == "compute_next_difficulty":
                # Import adaptive difficulty function<|fim_middle|><|fim_middle|>
                from app.services.adaptive_difficulty import adjust_difficulty
                
                current = arguments["currentDifficulty"]
                was_correct = arguments["wasCorrect"]
                
                next_difficulty = adjust_difficulty(current, was_correct)
                
                logger.info(
                    f"📊 Difficulty adjusted: {current} -> {next_difficulty} "
                    f"(correct: {was_correct})"
                )
                
                return {"nextDifficulty": next_difficulty}
            
            elif function_name == "get_chunk_by_difficulty":
                material_id_arg = arguments["materialId"]
                difficulty = arguments["difficulty"]
                
                # Validate material ID matches
                if material_id_arg != material_id:
                    return {
                        "error": f"Material ID mismatch: expected '{material_id}', got '{material_id_arg}'"
                    }
                
                # Try requested difficulty first
                chunk = await self.qdrant_service.get_chunk_by_difficulty(
                    material_id=material_id_arg,
                    difficulty=difficulty
                )
                
                # FALLBACK LOGIC: Try other difficulties if requested one is empty
                if not chunk:
                    logger.warning(
                        f"⚠️ No chunks at '{difficulty}' difficulty for material {material_id_arg}, "
                        f"attempting fallback..."
                    )
                    
                    # Define fallback order based on requested difficulty
                    fallback_order = {
                        "easy": ["medium", "hard"],
                        "medium": ["easy", "hard"],
                        "hard": ["medium", "easy"]
                    }
                    
                    for fallback_diff in fallback_order.get(difficulty, []):
                        logger.info(f"🔄 Trying fallback difficulty: {fallback_diff}")
                        chunk = await self.qdrant_service.get_chunk_by_difficulty(
                            material_id=material_id_arg,
                            difficulty=fallback_diff
                        )
                        if chunk:
                            logger.info(
                                f"✅ Found chunk at fallback difficulty '{fallback_diff}' "
                                f"(originally requested '{difficulty}')"
                            )
                            # Return chunk with fallback metadata
                            return {
                                "chunkId": chunk["id"],
                                "text": chunk["text"],
                                "difficulty": fallback_diff,  # Return actual difficulty used
                                "requestedDifficulty": difficulty,  # Include what was requested
                                "materialId": material_id_arg,
                                "fallback": True,
                                "fallbackMessage": f"Using '{fallback_diff}' difficulty ('{difficulty}' not available)"
                            }
                    
                    # If still no chunk found, try ANY chunk from this material
                    logger.warning(
                        f"⚠️ No chunks found at any specific difficulty, "
                        f"retrieving any available chunk for material {material_id_arg}"
                    )
                    chunk = await self.qdrant_service.get_any_chunk_for_material(
                        material_id=material_id_arg
                    )
                    
                    if not chunk:
                        return {
                            "error": "no_content_available",
                            "message": f"No content available for material '{material_id_arg}' at any difficulty level. "
                                    f"Please ensure the material has been properly ingested with difficulty ratings.",
                            "materialId": material_id_arg
                        }
                    
                    logger.info(f"✅ Retrieved chunk without difficulty filter")
                    return {
                        "chunkId": chunk["id"],
                        "text": chunk["text"],
                        "difficulty": chunk["payload"].get("difficulty", "unknown"),
                        "requestedDifficulty": difficulty,
                        "materialId": material_id_arg,
                        "fallback": True,
                        "fallbackMessage": f"Using any available chunk (no '{difficulty}' chunks found)"
                    }
                
                # Success with requested difficulty
                logger.info(
                    f"📄 Retrieved chunk: {chunk['id']} "
                    f"(difficulty: {difficulty}, length: {len(chunk['text'])} chars)"
                )
                
                return {
                    "chunkId": chunk["id"],
                    "text": chunk["text"],
                    "difficulty": chunk["payload"].get("difficulty", difficulty),
                    "materialId": material_id_arg,
                    "fallback": False
                }
            
            elif function_name == "record_question_event":
                session_id_arg = arguments["sessionId"]
                question_id = arguments["questionId"]
                difficulty = arguments["difficulty"]
                was_correct = arguments["wasCorrect"]
                
                # Validate session ID matches
                if session_id_arg != session_id:
                    return {
                        "error": f"Session ID mismatch: expected '{session_id}', got '{session_id_arg}'"
                    }
                
                # Create question record
                record = QuestionRecord(
                    questionId=question_id,
                    difficulty=difficulty,
                    wasCorrect=was_correct
                )
                
                # Append to session
                await self.session_service.append_question(
                    session_id=session_id_arg,
                    record=record
                )
                
                logger.info(
                    f"📝 Recorded question event: {question_id} "
                    f"(difficulty: {difficulty}, correct: {was_correct})"
                )
                
                return {"status": "recorded"}
            
            else:
                return {"error": f"Unknown tool: {function_name}"}
        
        except Exception as e:
            logger.error(f"❌ Tool execution failed for {function_name}: {e}")
            # Return error as tool result so LLM can handle it
            return {
                "error": "tool_execution_failed",
                "message": str(e),
                "tool": function_name
            }
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse and clean JSON response from LLM"""
        try:
            # Clean response
            cleaned = content.strip()
            
            # Remove markdown code blocks if present
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            cleaned = cleaned.strip()
            
            # Parse JSON
            question_data = json.loads(cleaned)
            
            # Validate required fields
            required_fields = ["questionId", "question", "options", "answer", "difficulty"]
            if not all(field in question_data for field in required_fields):
                raise MCPOrchestrationError(
                    f"Missing required fields. Got: {list(question_data.keys())}, "
                    f"Expected: {required_fields}"
                )
            
            # Validate options is list of 4 strings
            if not isinstance(question_data["options"], list) or len(question_data["options"]) != 4:
                raise MCPOrchestrationError(
                    f"Options must be list of 4 strings, got: {type(question_data['options'])} "
                    f"with {len(question_data.get('options', []))} items"
                )
            
            # Validate answer
            if question_data["answer"] not in ["A", "B", "C", "D"]:
                raise MCPOrchestrationError(
                    f"Answer must be A, B, C, or D, got: {question_data['answer']}"
                )
            
            # Validate difficulty
            valid_difficulties = ["easy", "medium", "hard"]
            if question_data["difficulty"] not in valid_difficulties:
                logger.warning(
                    f"⚠️ Invalid difficulty '{question_data['difficulty']}', "
                    f"defaulting to 'medium'"
                )
                question_data["difficulty"] = "medium"
            
            return question_data
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON: {e}")
            logger.error(f"Response preview: {content[:500]}")
            raise MCPOrchestrationError(
                f"Failed to parse JSON response: {str(e)}\n"
                f"Response preview: {content[:200]}"
            )
    
    async def _validate_session(
        self,
        session_id: str,
        material_id: str
    ) -> None:
        """Validate session exists and material matches"""
        session = await self.session_service.get_session(session_id)
        
        if not session:
            logger.error(f"❌ Session not found: {session_id}")
            raise SessionNotFoundError(f"Quiz session not found: {session_id}")
        
        if session.materialId != material_id:
            logger.error(
                f"❌ Material mismatch - "
                f"Session: {session.materialId}, Request: {material_id}"
            )
            raise MaterialMismatchError(
                f"Material ID mismatch - "
                f"session has '{session.materialId}', "
                f"request has '{material_id}'"
            )
        
        logger.debug(f"✓ Session validated: {session_id}")
    
    async def _store_question_metadata(
        self,
        session_id: str,
        question_data: Dict[str, Any]
    ) -> None:
        """Store complete question metadata in session (includes correct answer)"""
        try:
            metadata = QuestionMetadata(
                questionId=question_data["questionId"],
                question=question_data["question"],
                options=question_data["options"],
                correctAnswer=question_data["answer"],
                difficulty=question_data["difficulty"]
            )
            
            await self.session_service.store_question_metadata(
                session_id=session_id,
                metadata=metadata
            )
            
            logger.info(
                f"✓ Stored question metadata - "
                f"Session: {session_id}, Question: {metadata.questionId}"
            )
        
        except Exception as e:
            logger.error(
                f"❌ Failed to store question metadata for session {session_id}: {e}"
            )
            raise NextQuestionServiceError(
                f"Failed to store question metadata: {str(e)}"
            )


# Example usage
async def main():
    """Example demonstrating full MCP orchestration"""
    from motor.motor_asyncio import AsyncIOMotorClient
    
    # Initialize services
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["quiz_db"]
    qdrant_service = QdrantService(url="http://localhost:6333")
    
    service = NextQuestionService(
        db=db,
        qdrant_service=qdrant_service,
        openai_model="gpt-4o"
    )
    
    try:
        # Generate first question
        print("🎯 Generating FIRST question...")
        first_question = await service.generate_next_question(
            session_id="session_123",
            material_id="material_456",
            current_difficulty="medium",
            is_first=True
        )
        
        print("\n✅ First Question Generated:")
        print(json.dumps(first_question, indent=2))
        print("\n" + "="*60 + "\n")
        
        # Generate next question (user got it right)
        print("🎯 Generating NEXT question (user was correct)...")
        next_question = await service.generate_next_question(
            session_id="session_123",
            material_id="material_456",
            current_difficulty="medium",
            is_first=False,
            was_correct=True,
            previous_question_id=first_question["questionId"]
        )
        
        print("\n✅ Next Question Generated:")
        print(json.dumps(next_question, indent=2))
    
    except NextQuestionServiceError as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())