"""OpenAI and OpenAI-compatible API client."""

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


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI and any OpenAI-compatible endpoint (Ollama, vLLM, Groq, etc.)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=(base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/"),
        )
        self.organization = organization

    def _build_headers(self, request: LLMRequest) -> Dict[str, str]:
        key = request.api_key or self.api_key
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "APIMagic-LLM-Test-Framework/0.1.0",
        }
        if key:
            headers["Authorization"] = f"Bearer {key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if request.extra_headers:
            headers.update(request.extra_headers)
        return headers

    def _build_payload(self, request: LLMRequest, stream: bool = False) -> Dict[str, Any]:
        messages_payload: List[Dict[str, Any]] = []
        for msg in request.messages:
            m_dict: Dict[str, Any] = {"role": msg.role.value, "content": msg.content}
            if msg.name:
                m_dict["name"] = msg.name
            if msg.tool_call_id:
                m_dict["tool_call_id"] = msg.tool_call_id
            if msg.tool_calls:
                m_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            messages_payload.append(m_dict)

        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": messages_payload,
            "stream": stream,
        }

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.top_p is not None:
            payload["top_p"] = request.top_p

        if request.tools:
            payload["tools"] = [t.model_dump() for t in request.tools]
            if request.tool_choice:
                payload["tool_choice"] = request.tool_choice

        if request.response_format:
            payload["response_format"] = request.response_format.model_dump(exclude_none=True)

        if stream:
            payload["stream_options"] = {"include_usage": True}

        if request.extra_params:
            payload.update(request.extra_params)

        return payload

    def _handle_http_error(self, response: httpx.Response) -> None:
        if response.status_code == 401:
            raise AuthenticationError(
                f"OpenAI Authentication failed (401): {response.text}",
                status_code=401,
                response_body=response.text,
            )
        elif response.status_code == 429:
            raise RateLimitError(
                f"OpenAI Rate limit exceeded (429): {response.text}",
                status_code=429,
                response_body=response.text,
            )
        elif response.status_code >= 400:
            raise LLMClientError(
                f"OpenAI API error ({response.status_code}): {response.text}",
                status_code=response.status_code,
                response_body=response.text,
            )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        url = f"{request.base_url or self.base_url}/chat/completions"
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

        choice = data.get("choices", [{}])[0]
        message_data = choice.get("message", {})
        content = message_data.get("content")
        finish_reason = choice.get("finish_reason", "stop")

        raw_tool_calls = message_data.get("tool_calls", [])
        tool_calls: List[ToolCall] = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    parsed_args = json.loads(args)
                except Exception:
                    parsed_args = args
            else:
                parsed_args = args
            tool_calls.append(
                ToolCall(
                    id=tc.get("id"),
                    type=tc.get("type", "function"),
                    name=fn.get("name", ""),
                    arguments=parsed_args,
                )
            )

        usage_data = data.get("usage", {})
        tokens = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            cached_tokens=usage_data.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        )

        cost = calculate_cost(request.model, tokens.prompt_tokens, tokens.completion_tokens)

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            role=MessageRole.ASSISTANT,
            finish_reason=finish_reason,
            latency_ms=round(latency_ms, 2),
            tokens=tokens,
            cost_usd=cost,
            model=data.get("model", request.model),
            provider="openai",
            raw_response=data,
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        url = f"{request.base_url or self.base_url}/chat/completions"
        headers = self._build_headers(request)
        payload = self._build_payload(request, stream=True)

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.is_error:
                    await response.aread()
                    self._handle_http_error(response)

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    line = line.strip()
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choice = data.get("choices", [{}])[0]
                            delta = choice.get("delta", {})
                            content_delta = delta.get("content")
                            finish_reason = choice.get("finish_reason")
                            chunk_latency = (time.perf_counter() - start_time) * 1000.0

                            yield StreamChunk(
                                delta_content=content_delta,
                                finish_reason=finish_reason,
                                latency_from_start_ms=round(chunk_latency, 2),
                            )
                        except json.JSONDecodeError:
                            continue
