"""Factory for instantiating LLM clients."""

from typing import Optional
from apimagic.client.base import BaseLLMClient
from apimagic.client.openai_client import OpenAIClient
from apimagic.client.anthropic_client import AnthropicClient
from apimagic.client.gemini_client import GeminiClient
from apimagic.client.mock_client import MockLLMClient


def get_client(
    provider: str = "mock",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> BaseLLMClient:
    """Returns an LLM client instance for the specified provider."""
    provider_lower = provider.lower().strip()

    if provider_lower == "mock":
        return MockLLMClient(**kwargs)

    elif provider_lower in ("openai", "azure"):
        return OpenAIClient(api_key=api_key, base_url=base_url, **kwargs)

    elif provider_lower == "ollama":
        base = base_url or "http://localhost:11434/v1"
        return OpenAIClient(api_key=api_key or "ollama", base_url=base, **kwargs)

    elif provider_lower == "vllm":
        base = base_url or "http://localhost:8000/v1"
        return OpenAIClient(api_key=api_key or "vllm", base_url=base, **kwargs)

    elif provider_lower == "groq":
        base = base_url or "https://api.groq.com/openai/v1"
        return OpenAIClient(api_key=api_key, base_url=base, **kwargs)

    elif provider_lower == "deepseek":
        base = base_url or "https://api.deepseek.com/v1"
        return OpenAIClient(api_key=api_key, base_url=base, **kwargs)

    elif provider_lower in ("anthropic", "claude"):
        return AnthropicClient(api_key=api_key, base_url=base_url, **kwargs)

    elif provider_lower in ("gemini", "google"):
        return GeminiClient(api_key=api_key, base_url=base_url, **kwargs)

    elif provider_lower in ("custom", "generic"):
        return OpenAIClient(api_key=api_key, base_url=base_url, **kwargs)

    else:
        # Default to OpenAI compatible format
        return OpenAIClient(api_key=api_key, base_url=base_url, **kwargs)
