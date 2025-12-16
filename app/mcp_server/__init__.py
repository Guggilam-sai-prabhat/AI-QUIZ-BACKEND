"""
MCP Server Module
Exposes quiz tools and materials context via Model Context Protocol (Anthropic MCP)

This module provides:
1. MCP Server - Standalone server exposing tools for adaptive quiz functionality
2. Context utilities - Load and manage study material context
3. Quiz prompts - Build prompts for LLM quiz generation
4. LLM client - Interface to various LLM providers
5. Quiz parser - Parse and validate LLM-generated quiz JSON
"""

# MCP Server (Anthropic MCP Protocol)
from app.mcp_server.server import run_mcp_server

# Legacy/HTTP routes (if still needed)
try:
    
    from .handlers import handle_mcp_request
except ImportError:
    mcp_router = None
    handle_mcp_request = None

# Context utilities
from ..utils.context_loader import load_context, load_multi_material_context

# Quiz prompt builders
from ..utils.quiz_prompt import (
    build_quiz_prompt,
    build_quiz_prompt_with_system,
    get_system_instruction
)

# LLM client
from ..services.llm_client import (
    generate_quiz,
    health_check,
    get_available_providers,
    LLMProvider,
    LLMClientError,
    LLMTimeoutError,
    LLMAPIError
)

# Quiz parser
from ..utils.quiz_parser import (
    parse_quiz_json,
    safe_parse_quiz_json,
    QuizParseError,
    InvalidJSONError,
    ValidationError
)

__all__ = [
    # MCP Server
    "run_mcp_server",
    
    # Legacy HTTP (optional)
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

# Version info
__version__ = "1.0.0"
__mcp_version__ = "0.9.0" 