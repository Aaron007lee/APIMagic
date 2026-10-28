"""Base class for all evaluators and assertions."""

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any, Optional
from apimagic.core.models import LLMResponse, MetricResult


class BaseEvaluator(ABC):
    """Abstract base class for all evaluation metrics and assertions."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        """Asynchronously evaluates the LLM response."""
        pass

    def evaluate_sync(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        """Synchronous wrapper for evaluation, safe inside or outside running event loops."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If an event loop is already running (e.g. in pytest-asyncio or jupyter)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run,
                    self.evaluate(response, context, expected, **kwargs)
                )
                return future.result()
        else:
            return asyncio.run(self.evaluate(response, context, expected, **kwargs))
