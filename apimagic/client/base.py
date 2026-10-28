"""Base client interface for LLM providers."""

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
from apimagic.core.models import LLMRequest, LLMResponse, StreamChunk


class BaseLLMClient(ABC):
    """Abstract base class for all LLM API providers."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Asynchronously executes an LLM completion request and returns an LLMResponse."""
        pass

    @abstractmethod
    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        """Asynchronously executes a streaming LLM completion request yielding chunks."""
        pass

    def generate_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous wrapper for generate, safe inside or outside running event loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.generate(request))
                return future.result()
        else:
            return asyncio.run(self.generate(request))
