"""
Question Generator Service - MCP-Driven Architecture
LLM orchestrates ALL logic via MCP tools - no backend question generation
"""
import json
import logging
import os
from typing import Dict, Any, Literal, List, Optional
import openai

logger = logging.getLogger(__name__)


class QuestionGenerationError(Exception):
    """Custom exception for question generation errors"""
    pass


class MCPOrchestrationError(QuestionGenerationError):
    """MCP orchestration error"""
    pass


class QuestionGeneratorService:
    """
    MCP-driven question generator.
    
    The LLM orchestrates the entire workflow:
    1. Retrieves chunk via get_chunk_by_difficulty MCP tool
    2. Generates MCQ strictly from the retrieved chunk
    3. Returns structured JSON (no commentary, no markdown)
    
    Backend only forwards MCP tool calls and results.
    """
    
    # MCP Tool Schema
    MCP_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_chunk_by_difficulty",
                "description": (
                    "Retrieve a random text chunk from study material at the specified "
                    "difficulty level. You MUST call this tool BEFORE generating any question. "
                    "The returned chunk text is the ONLY source you can use for question generation."
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
        }
    ]
    
    # System prompt for LLM
    SYSTEM_PROMPT = """You are a precise multiple-choice question generator. Your task is to generate ONE high-quality MCQ by following this MANDATORY workflow:

**STRICT WORKFLOW (NO SHORTCUTS):**

1. **RETRIEVE CONTENT (MANDATORY):**
   - You MUST call get_chunk_by_difficulty(materialId, difficulty) FIRST
   - You MUST NOT generate any question until you receive the chunk text
   - You MUST NOT fabricate, hallucinate, or invent any content
   - The returned chunk is your ONLY allowed source material

2. **GENERATE MCQ (STRICT REQUIREMENTS):**
   - Base your question ENTIRELY on the retrieved chunk text
   - Create exactly 4 answer options (A, B, C, D)
   - Ensure ONE option is unambiguously correct based on the chunk
   - Make all options plausible but only one truly accurate
   - The question must be answerable ONLY using information from the chunk
   - Do NOT use external knowledge or make assumptions beyond the chunk

3. **RETURN STRICT JSON (NO EXTRAS):**
   - Return ONLY valid JSON with this EXACT structure:
   ```json
   {
     "questionId": "uuid-v4-format",
     "question": "Your question text here?",
     "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
     "answer": "A",
     "difficulty": "medium"
   }
   ```
   - Do NOT wrap JSON in markdown code blocks (```json)
   - Do NOT add any commentary, explanation, or extra text
   - Do NOT include field descriptions or examples
   - Return ONLY the raw JSON object

**CRITICAL RULES:**
- ❌ NEVER generate questions without calling get_chunk_by_difficulty first
- ❌ NEVER fabricate content not present in the retrieved chunk
- ❌ NEVER return markdown-wrapped JSON or add explanatory text
- ✅ ALWAYS call the tool before generating
- ✅ ALWAYS use ONLY the chunk text as your source
- ✅ ALWAYS return clean, parseable JSON

**VALIDATION CHECKLIST:**
Before returning your JSON, verify:
- [ ] Did I call get_chunk_by_difficulty?
- [ ] Is my question based ONLY on the chunk text?
- [ ] Do I have exactly 4 options in an array?
- [ ] Is my answer one of: A, B, C, D?
- [ ] Is my JSON valid with no markdown wrapper?
- [ ] Did I include questionId in uuid4 format?

If any checkbox is unchecked, DO NOT PROCEED. Fix the issue first."""
    
    def __init__(
        self,
        qdrant_service,
        openai_model: str = "gpt-4o",
        max_iterations: int = 5
    ):
        """
        Initialize question generator service
        
        Args:
            qdrant_service: Qdrant service for chunk retrieval
            openai_model: OpenAI model to use
            max_iterations: Maximum tool call iterations
        """
        self.qdrant_service = qdrant_service
        self.openai_model = openai_model
        self.max_iterations = max_iterations
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.openai_client = openai.AsyncOpenAI(api_key=api_key)
    
    async def generate_single_question(
        self,
        material_id: str,
        difficulty: Literal["easy", "medium", "hard"] = "medium",
        include_answer: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a single MCQ via LLM + MCP orchestration.
        
        The LLM handles ALL logic:
        - Calls get_chunk_by_difficulty to retrieve content
        - Generates MCQ strictly from the retrieved chunk
        - Returns structured JSON
        
        Args:
            material_id: Study material ID to generate question from
            difficulty: Difficulty level
            include_answer: If True, includes correct answer (for internal storage)
        
        Returns:
            Dictionary with question data (answer included only if include_answer=True)
        
        Raises:
            QuestionGenerationError: If generation fails
        """
        logger.info(
            f"🎯 Generating {difficulty} question via MCP orchestration - "
            f"Material: {material_id}"
        )
        
        # Build user prompt
        user_prompt = self._build_user_prompt(
            material_id=material_id,
            difficulty=difficulty
        )
        
        # Orchestrate with LLM
        question_data = await self._orchestrate_with_llm(
            user_prompt=user_prompt,
            material_id=material_id
        )
        
        # Validate structure
        self._validate_question_structure(question_data)
        
        # Build response
        result = {
            "questionId": question_data["questionId"],
            "question": question_data["question"],
            "options": question_data["options"],
            "difficulty": question_data["difficulty"]
        }
        
        # Conditionally include answer
        if include_answer:
            result["answer"] = question_data["answer"]
            logger.info(f"✅ Generated question {result['questionId']} with answer (internal mode)")
        else:
            logger.info(f"✅ Generated question {result['questionId']} without answer (public mode)")
        
        return result
    
    def _build_user_prompt(
        self,
        material_id: str,
        difficulty: str
    ) -> str:
        """Build user prompt for question generation"""
        return f"""Generate a {difficulty} difficulty multiple-choice question.

**Parameters:**
- Material ID: {material_id}
- Difficulty: {difficulty}

**Instructions:**
1. Call get_chunk_by_difficulty(materialId="{material_id}", difficulty="{difficulty}")
2. Wait for the chunk text to be returned
3. Generate an MCQ based ONLY on the retrieved chunk
4. Return the JSON in the exact format specified in the system prompt

Begin the workflow now. Remember: NO question generation until you have retrieved the chunk via the tool."""
    
    async def _orchestrate_with_llm(
        self,
        user_prompt: str,
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
                
                # Check if done (no tool calls)
                if not assistant_message.tool_calls:
                    content = assistant_message.content
                    if not content:
                        raise MCPOrchestrationError("No content in final response")
                    
                    # Parse and validate JSON
                    question_data = self._parse_json_response(content)
                    
                    logger.info(f"✅ LLM returned final MCQ: {question_data['questionId']}")
                    return question_data
                
                # Process tool calls
                logger.info(f"🔧 Processing {len(assistant_message.tool_calls)} tool calls")
                
                # Add assistant message to history
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
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    logger.info(f"🛠️ Executing tool: {function_name} with args: {function_args}")
                    
                    # Execute MCP tool
                    tool_result = await self._execute_mcp_tool(
                        function_name=function_name,
                        arguments=function_args,
                        material_id=material_id
                    )
                    
                    # Add tool result to messages
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
        material_id: str
    ) -> Dict[str, Any]:
        """Execute MCP tool and return result"""
        
        try:
            if function_name == "get_chunk_by_difficulty":
                material_id_arg = arguments["materialId"]
                difficulty = arguments["difficulty"]
                
                # Validate material ID matches
                if material_id_arg != material_id:
                    return {
                        "error": f"Material ID mismatch: expected '{material_id}', got '{material_id_arg}'"
                    }
                
                # Retrieve chunk via Qdrant
                chunk = await self.qdrant_service.get_chunk_by_difficulty(
                    material_id=material_id_arg,
                    difficulty=difficulty
                )
                
                if not chunk:
                    return {
                        "error": f"No content available for material '{material_id_arg}' at difficulty '{difficulty}'"
                    }
                
                return {
                    "chunkId": chunk["id"],
                    "text": chunk["text"],
                    "difficulty": chunk["payload"].get("difficulty", difficulty),
                    "materialId": material_id_arg
                }
            
            else:
                return {"error": f"Unknown tool: {function_name}"}
        
        except Exception as e:
            logger.error(f"❌ Tool execution failed for {function_name}: {e}")
            return {"error": str(e)}
    
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
            
            return question_data
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse JSON: {e}")
            logger.error(f"Response preview: {content[:500]}")
            raise MCPOrchestrationError(
                f"Failed to parse JSON response: {str(e)}\n"
                f"Response preview: {content[:200]}"
            )
    
    def _validate_question_structure(self, data: dict) -> None:
        """Validate the generated question structure"""
        required_fields = ["questionId", "question", "options", "answer", "difficulty"]
        
        # Check required fields
        for field in required_fields:
            if field not in data:
                raise QuestionGenerationError(f"Missing required field: {field}")
        
        # Validate question
        if not data["question"] or not isinstance(data["question"], str):
            raise QuestionGenerationError("Question must be a non-empty string")
        
        # Validate options (should be a list of 4 strings)
        if not isinstance(data["options"], list) or len(data["options"]) != 4:
            raise QuestionGenerationError(
                f"Must have exactly 4 options as list, got {len(data.get('options', []))}"
            )
        
        # Validate all options are non-empty strings
        for i, option in enumerate(data["options"]):
            if not option or not isinstance(option, str):
                raise QuestionGenerationError(f"Option {i} must be a non-empty string")
        
        # Validate answer is one of A, B, C, D
        if data["answer"] not in ["A", "B", "C", "D"]:
            raise QuestionGenerationError(
                f"Answer must be one of: A, B, C, D (got: {data.get('answer')})"
            )
        
        # Validate difficulty
        valid_difficulties = ["easy", "medium", "hard"]
        if data["difficulty"] not in valid_difficulties:
            raise QuestionGenerationError(
                f"Difficulty must be one of: {valid_difficulties} (got: {data.get('difficulty')})"
            )
        
        logger.debug(f"✅ Question structure validated successfully")
    
    async def generate_multiple_questions(
        self,
        material_id: str,
        count: int,
        difficulty: Literal["easy", "medium", "hard"] = "medium",
        include_answer: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple questions from a material.
        
        Args:
            material_id: Study material ID
            count: Number of questions to generate
            difficulty: Difficulty level for all questions
            include_answer: If True, includes correct answers
        
        Returns:
            List of question dictionaries
        """
        import asyncio
        
        logger.info(f"📚 Generating {count} questions at {difficulty} difficulty")
        
        tasks = [
            self.generate_single_question(material_id, difficulty, include_answer)
            for _ in range(count)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        questions = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Failed to generate question {i+1}: {result}")
                errors.append((i, str(result)))
            else:
                questions.append(result)
        
        if errors:
            logger.warning(f"⚠️ Generated {len(questions)}/{count} questions successfully")
        else:
            logger.info(f"✅ Generated all {len(questions)} questions successfully")
        
        return questions


# Example usage
async def main():
    """Example usage demonstrating MCP-driven question generation"""
    from app.services.qdrant_service import QdrantService
    
    # Initialize services
    qdrant_service = QdrantService(url="http://localhost:6333")
    generator = QuestionGeneratorService(qdrant_service=qdrant_service)
    
    try:
        # Generate question (public mode - no answer)
        public_question = await generator.generate_single_question(
            material_id="material_123",
            difficulty="medium",
            include_answer=False
        )
        
        print("Public Question (for API response):")
        print(json.dumps(public_question, indent=2))
        print("\n" + "="*50 + "\n")
        
        # Generate question (internal mode - with answer)
        internal_question = await generator.generate_single_question(
            material_id="material_123",
            difficulty="medium",
            include_answer=True
        )
        
        print("Internal Question (for database storage):")
        print(json.dumps(internal_question, indent=2))
    
    except QuestionGenerationError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())