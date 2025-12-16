"""
LLM Client
Unified interface for generating content via OpenAI, Anthropic, and Grok (xAI) APIs
"""
import os
import asyncio
import logging
from typing import Optional
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROK = "grok"


class LLMClientError(Exception):
    """Base exception for LLM client errors"""
    pass


class LLMTimeoutError(LLMClientError):
    """Raised when LLM request times out"""
    pass


class LLMAPIError(LLMClientError):
    """Raised when LLM API returns an error"""
    pass


# API Endpoints
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

# Configuration
DEFAULT_TIMEOUT = 60.0  # seconds
MAX_RETRIES = 3
INITIAL_BACKOFF = 1.0  # seconds

# Model defaults
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-mini-beta")


async def _retry_with_backoff(
    coro_func,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF
):
    """
    Execute coroutine with exponential backoff retry
    
    Args:
        coro_func: Async function to call (no arguments)
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff delay in seconds
    
    Returns:
        Result from successful coroutine execution
    
    Raises:
        Last exception if all retries exhausted
    """
    last_exception = None
    backoff = initial_backoff
    
    for attempt in range(max_retries + 1):
        try:
            return await coro_func()
        except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last_exception = e
            
            if attempt == max_retries:
                break
            
            # Don't retry on client errors (4xx) except rate limits (429)
            if isinstance(e, httpx.HTTPStatusError):
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    raise
            
            logger.warning(
                f"⚠️ Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                f"Retrying in {backoff:.1f}s..."
            )
            
            await asyncio.sleep(backoff)
            backoff *= 2  # Exponential backoff
    
    raise last_exception


async def _call_openai_compatible(
    prompt: str,
    api_url: str,
    api_key: str,
    model: str,
    provider_name: str,
    timeout: float = DEFAULT_TIMEOUT
) -> str:
    """
    Call OpenAI-compatible Chat Completions API
    Works with OpenAI, Grok, and other compatible APIs
    
    Args:
        prompt: The prompt to send
        api_url: API endpoint URL
        api_key: API key for authentication
        model: Model identifier
        provider_name: Name for logging
        timeout: Request timeout in seconds
    
    Returns:
        Raw response content string
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4096
    }
    
    async def make_request():
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
    
    try:
        data = await _retry_with_backoff(make_request)
        content = data["choices"][0]["message"]["content"]
        logger.info(f"✅ {provider_name} response received ({len(content)} chars)")
        return content
        
    except httpx.TimeoutException as e:
        logger.error(f"❌ {provider_name} request timed out after {timeout}s")
        raise LLMTimeoutError(f"{provider_name} request timed out: {e}")
    
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ {provider_name} API error: {e.response.status_code}")
        raise LLMAPIError(f"{provider_name} API error: {e.response.text}")


async def _call_openai(prompt: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Call OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LLMClientError("OPENAI_API_KEY environment variable not set")
    
    return await _call_openai_compatible(
        prompt=prompt,
        api_url=OPENAI_API_URL,
        api_key=api_key,
        model=OPENAI_MODEL,
        provider_name="OpenAI",
        timeout=timeout
    )


async def _call_grok(prompt: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Call Grok (xAI) API - OpenAI compatible"""
    api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
    if not api_key:
        raise LLMClientError("GROK_API_KEY or XAI_API_KEY environment variable not set")
    
    return await _call_openai_compatible(
        prompt=prompt,
        api_url=GROK_API_URL,
        api_key=api_key,
        model=GROK_MODEL,
        provider_name="Grok",
        timeout=timeout
    )


async def _call_anthropic(prompt: str, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Call Anthropic Messages API"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMClientError("ANTHROPIC_API_KEY environment variable not set")
    
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 4096,
        "system": "Return ONLY valid JSON.",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    async def make_request():
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(ANTHROPIC_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
    
    try:
        data = await _retry_with_backoff(make_request)
        content = data["content"][0]["text"]
        logger.info(f"✅ Anthropic response received ({len(content)} chars)")
        return content
        
    except httpx.TimeoutException as e:
        logger.error(f"❌ Anthropic request timed out after {timeout}s")
        raise LLMTimeoutError(f"Anthropic request timed out: {e}")
    
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Anthropic API error: {e.response.status_code}")
        raise LLMAPIError(f"Anthropic API error: {e.response.text}")


async def generate_quiz(
    prompt: str,
    provider: str = "grok",
    timeout: Optional[float] = None
) -> str:
    """
    Generate quiz content using specified LLM provider
    
    Args:
        prompt: The quiz generation prompt
        provider: LLM provider - "openai", "anthropic", or "grok" (default)
        timeout: Optional custom timeout (uses DEFAULT_TIMEOUT if not specified)
    
    Returns:
        Raw string output from the LLM (no parsing)
    
    Raises:
        LLMClientError: If API key is missing
        LLMTimeoutError: If request times out after retries
        LLMAPIError: If API returns an error
        ValueError: If invalid provider specified
    
    Example:
        response = await generate_quiz(prompt, provider="grok")
    """
    timeout = timeout or DEFAULT_TIMEOUT
    provider = provider.lower()
    
    logger.info(f"🤖 Generating quiz via {provider} (timeout: {timeout}s)")
    
    if provider == LLMProvider.OPENAI:
        return await _call_openai(prompt, timeout)
    
    elif provider == LLMProvider.ANTHROPIC:
        return await _call_anthropic(prompt, timeout)
    
    elif provider == LLMProvider.GROK:
        return await _call_grok(prompt, timeout)
    
    else:
        raise ValueError(
            f"Invalid provider: {provider}. "
            f"Supported providers: {[p.value for p in LLMProvider]}"
        )


async def health_check(provider: str = "grok") -> dict:
    """
    Check if LLM provider is configured and reachable
    
    Args:
        provider: LLM provider to check
    
    Returns:
        Health status dictionary
    """
    provider = provider.lower()
    
    if provider == LLMProvider.OPENAI:
        api_key = os.getenv("OPENAI_API_KEY")
        model = OPENAI_MODEL
    elif provider == LLMProvider.ANTHROPIC:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = ANTHROPIC_MODEL
    elif provider == LLMProvider.GROK:
        api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY")
        model = GROK_MODEL
    else:
        return {"provider": provider, "status": "error", "message": "Invalid provider"}
    
    configured = bool(api_key)
    
    return {
        "provider": provider,
        "configured": configured,
        "model": model,
        "status": "ready" if configured else "not_configured"
    }


def get_available_providers() -> list:
    """
    Get list of configured providers
    
    Returns:
        List of provider names that have API keys configured
    """
    available = []
    
    if os.getenv("OPENAI_API_KEY"):
        logger.info(f"✅ OpenAI API key found, will use for difficulty rating")
        available.append("openai")
    # if os.getenv("ANTHROPIC_API_KEY"):
    #     available.append("anthropic")
    # if os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY"):
    #     available.append("grok")
    else:
         logger.warning("⚠️ No OpenAI API key found, will use rule-based difficulty rating")
    
    return available