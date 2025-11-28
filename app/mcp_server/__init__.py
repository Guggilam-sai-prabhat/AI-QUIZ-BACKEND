"""
MCP Server Module
Exposes uploaded materials as context provider via Model Context Protocol
"""
from .server import MCPServer, get_mcp_server
from .routes import router as mcp_router
from .handlers import handle_mcp_request
from .context_loader import load_context, load_multi_material_context
from .quiz_prompt import build_quiz_prompt, build_quiz_prompt_with_system, get_system_instruction
from .llm_client import (
    generate_quiz,
    health_check,
    get_available_providers,
    LLMProvider,
    LLMClientError,
    LLMTimeoutError,
    LLMAPIError
)
from .quiz_parser import (
    parse_quiz_json,
    safe_parse_quiz_json,
    QuizParseError,
    InvalidJSONError,
    ValidationError
)

__all__ = [
    # Server
    "MCPServer",
    "get_mcp_server",
    "mcp_router",
    "handle_mcp_request",
    # Context
    "load_context",
    "load_multi_material_context",
    # Quiz Prompt
    "build_quiz_prompt",
    "build_quiz_prompt_with_system",
    "get_system_instruction",
    # LLM Client
    "generate_quiz",
    "health_check",
    "get_available_providers",
    "LLMProvider",
    "LLMClientError",
    "LLMTimeoutError",
    "LLMAPIError",
    # Quiz Parser
    "parse_quiz_json",
    "safe_parse_quiz_json",
    "QuizParseError",
    "InvalidJSONError",
    "ValidationError",
]