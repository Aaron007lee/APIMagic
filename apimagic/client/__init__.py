"""Clients package for APIMagic."""

from apimagic.client.base import BaseLLMClient
from apimagic.client.openai_client import OpenAIClient
from apimagic.client.anthropic_client import AnthropicClient
from apimagic.client.gemini_client import GeminiClient
from apimagic.client.mock_client import MockLLMClient
from apimagic.client.factory import get_client

__all__ = [
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "MockLLMClient",
    "get_client",
]
