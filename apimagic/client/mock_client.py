"""Mock LLM client for fast, deterministic, zero-cost tests and simulation."""

import asyncio
import json
import re
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Pattern, Union

from apimagic.client.base import BaseLLMClient
from apimagic.core.exceptions import LLMClientError, RateLimitError
from apimagic.core.models import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    MessageRole,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from apimagic.pricing import calculate_cost


class MockRule:
    """A matching rule for mock responses."""

    def __init__(
        self,
        pattern: Union[str, Pattern],
        response_content: Optional[str] = None,
        tool_calls: Optional[Union[List[ToolCall], List[Dict[str, Any]]]] = None,
        latency_ms: float = 25.0,
        tokens: Optional[TokenUsage] = None,
        error_to_raise: Optional[Exception] = None,
        handler: Optional[Callable[[LLMRequest], LLMResponse]] = None,
    ):
        self.pattern = re.compile(pattern) if isinstance(pattern, str) else pattern
        self.response_content = response_content
        self.tool_calls = [
            ToolCall(**tc) if isinstance(tc, dict) else tc
            for tc in (tool_calls or [])
        ]
        self.latency_ms = latency_ms
        self.tokens = tokens
        self.error_to_raise = error_to_raise
        self.handler = handler

    def matches(self, request: LLMRequest) -> bool:
        combined_text = " ".join([m.content or "" for m in request.messages])
        return bool(self.pattern.search(combined_text))


class MockLLMClient(BaseLLMClient):
    """Deterministic mock LLM client supporting response rules, tool simulation, and streaming."""

    def __init__(
        self,
        default_response: str = "This is a default mock response from APIMagic.",
        default_latency_ms: float = 15.0,
    ):
        super().__init__(api_key="mock-key", base_url="http://mock.apimagic.local")
        self.default_response = default_response
        self.default_latency_ms = default_latency_ms
        self.rules: List[MockRule] = []
        self.call_history: List[LLMRequest] = []

    def add_rule(
        self,
        pattern: Union[str, Pattern],
        response_content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        latency_ms: float = 20.0,
        tokens: Optional[TokenUsage] = None,
        error_to_raise: Optional[Exception] = None,
        handler: Optional[Callable[[LLMRequest], LLMResponse]] = None,
    ) -> "MockLLMClient":
        """Adds a rule to match against incoming requests."""
        rule = MockRule(
            pattern=pattern,
            response_content=response_content,
            tool_calls=tool_calls,
            latency_ms=latency_ms,
            tokens=tokens,
            error_to_raise=error_to_raise,
            handler=handler,
        )
        self.rules.append(rule)
        return self

    def clear_rules(self) -> None:
        self.rules.clear()

    def clear_history(self) -> None:
        self.call_history.clear()

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_history.append(request)
        start_time = time.perf_counter()

        # Check rules in order
        matched_rule: Optional[MockRule] = None
        for rule in self.rules:
            if rule.matches(request):
                matched_rule = rule
                break

        if matched_rule:
            if matched_rule.error_to_raise:
                if matched_rule.latency_ms > 0:
                    await asyncio.sleep(matched_rule.latency_ms / 1000.0)
                raise matched_rule.error_to_raise

            if matched_rule.handler:
                resp = matched_rule.handler(request)
                resp.latency_ms = (time.perf_counter() - start_time) * 1000.0
                return resp

            if matched_rule.latency_ms > 0:
                await asyncio.sleep(matched_rule.latency_ms / 1000.0)

            content = matched_rule.response_content
            tool_calls = matched_rule.tool_calls
        else:
            if self.default_latency_ms > 0:
                await asyncio.sleep(self.default_latency_ms / 1000.0)
            content = self.default_response
            tool_calls = []

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Estimate tokens
        prompt_len = sum(len(m.content or "") for m in request.messages)
        prompt_tokens = max(1, prompt_len // 4)
        comp_len = len(content or "") + sum(len(str(tc.arguments)) for tc in tool_calls)
        completion_tokens = max(1, comp_len // 4)
        tokens = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

        cost = calculate_cost(request.model, tokens.prompt_tokens, tokens.completion_tokens)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            role=MessageRole.ASSISTANT,
            finish_reason="tool_calls" if tool_calls else "stop",
            latency_ms=round(total_latency_ms, 2),
            ttft_ms=round(total_latency_ms * 0.4, 2),
            tokens=tokens,
            cost_usd=cost,
            model=request.model,
            provider="mock",
            raw_response={
                "mock": True,
                "content": content,
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            },
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        self.call_history.append(request)
        start_time = time.perf_counter()

        response = await self.generate(request)
        text = response.content or ""
        words = text.split(" ")

        for i, word in enumerate(words):
            chunk_text = word if i == 0 else " " + word
            await asyncio.sleep(0.01)
            chunk_latency = (time.perf_counter() - start_time) * 1000.0
            is_last = (i == len(words) - 1)
            yield StreamChunk(
                delta_content=chunk_text,
                delta_tool_calls=response.tool_calls if is_last else None,
                finish_reason="stop" if is_last else None,
                latency_from_start_ms=round(chunk_latency, 2),
            )
