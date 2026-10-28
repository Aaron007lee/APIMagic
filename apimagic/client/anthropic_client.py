"""Anthropic Claude API client."""

import json
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional
import httpx

from apimagic.client.base import BaseLLMClient
from apimagic.core.exceptions import AuthenticationError, LLMClientError, RateLimitError
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


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude Messages API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        anthropic_version: str = "2023-06-01",
    ):
        super().__init__(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
            base_url=(base_url or os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")).rstrip("/"),
        )
        self.anthropic_version = anthropic_version

    def _build_headers(self, request: LLMRequest) -> Dict[str, str]:
        key = request.api_key or self.api_key
        headers = {
            "Content-Type": "application/json",
            "x-api-key": key or "",
            "anthropic-version": self.anthropic_version,
            "User-Agent": "APIMagic-LLM-Test-Framework/0.1.0",
        }
        if request.extra_headers:
            headers.update(request.extra_headers)
        return headers

    def _build_payload(self, request: LLMRequest, stream: bool = False) -> Dict[str, Any]:
        system_content = None
        messages_payload: List[Dict[str, Any]] = []

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                system_content = msg.content
            else:
                role = "assistant" if msg.role == MessageRole.ASSISTANT else "user"
                messages_payload.append({"role": role, "content": msg.content or ""})

        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": messages_payload,
            "max_tokens": request.max_tokens or 1024,
            "stream": stream,
        }

        if system_content:
            payload["system"] = system_content
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p

        if request.tools:
            anthropic_tools = []
            for t in request.tools:
                anthropic_tools.append({
                    "name": t.function.name,
                    "description": t.function.description or "",
                    "input_schema": t.function.parameters,
                })
            payload["tools"] = anthropic_tools

        if request.extra_params:
            payload.update(request.extra_params)

        return payload

    def _handle_http_error(self, response: httpx.Response) -> None:
        if response.status_code == 401:
            raise AuthenticationError(
                f"Anthropic Authentication failed (401): {response.text}",
                status_code=401,
                response_body=response.text,
            )
        elif response.status_code == 429:
            raise RateLimitError(
                f"Anthropic Rate limit exceeded (429): {response.text}",
                status_code=429,
                response_body=response.text,
            )
        elif response.status_code >= 400:
            raise LLMClientError(
                f"Anthropic API error ({response.status_code}): {response.text}",
                status_code=response.status_code,
                response_body=response.text,
            )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        url = f"{request.base_url or self.base_url}/v1/messages"
        headers = self._build_headers(request)
        payload = self._build_payload(request, stream=False)

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
            except Exception as e:
                raise LLMClientError(f"HTTP connection failed: {str(e)}") from e

        if response.is_error:
            self._handle_http_error(response)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        data = response.json()

        text_content_parts = []
        tool_calls: List[ToolCall] = []

        for block in data.get("content", []):
            if block.get("type") == "text":
                text_content_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id"),
                        type="function",
                        name=block.get("name", ""),
                        arguments=block.get("input", {}),
                    )
                )

        combined_text = "".join(text_content_parts) if text_content_parts else None

        usage = data.get("usage", {})
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        tokens = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cached_tokens=usage.get("cache_read_input_tokens", 0),
        )

        cost = calculate_cost(request.model, tokens.prompt_tokens, tokens.completion_tokens)

        return LLMResponse(
            content=combined_text,
            tool_calls=tool_calls,
            role=MessageRole.ASSISTANT,
            finish_reason=data.get("stop_reason", "end_turn"),
            latency_ms=round(latency_ms, 2),
            tokens=tokens,
            cost_usd=cost,
            model=data.get("model", request.model),
            provider="anthropic",
            raw_response=data,
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        url = f"{request.base_url or self.base_url}/v1/messages"
        headers = self._build_headers(request)
        payload = self._build_payload(request, stream=True)

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.is_error:
                    await response.aread()
                    self._handle_http_error(response)

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type")
                        chunk_latency = (time.perf_counter() - start_time) * 1000.0

                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield StreamChunk(
                                    delta_content=delta.get("text"),
                                    latency_from_start_ms=round(chunk_latency, 2),
                                )
                        elif event_type == "message_stop":
                            yield StreamChunk(
                                finish_reason="stop",
                                latency_from_start_ms=round(chunk_latency, 2),
                            )
                    except json.JSONDecodeError:
                        continue
