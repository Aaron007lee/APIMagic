"""Google Gemini API client."""

import json
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
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


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(
            api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", ""),
            base_url=(base_url or "https://generativelanguage.googleapis.com/v1beta").rstrip("/"),
        )

    def _build_payload(self, request: LLMRequest) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        system_instruction = None
        contents: List[Dict[str, Any]] = []

        for msg in request.messages:
            if msg.role == MessageRole.SYSTEM:
                system_instruction = {"parts": [{"text": msg.content or ""}]}
            else:
                role = "model" if msg.role == MessageRole.ASSISTANT else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content or ""}],
                })

        generation_config: Dict[str, Any] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.response_format and request.response_format.type == "json_object":
            generation_config["responseMimeType"] = "application/json"

        payload: Dict[str, Any] = {
            "contents": contents,
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction
        if generation_config:
            payload["generationConfig"] = generation_config

        if request.tools:
            function_declarations = []
            for t in request.tools:
                function_declarations.append({
                    "name": t.function.name,
                    "description": t.function.description or "",
                    "parameters": t.function.parameters,
                })
            payload["tools"] = [{"functionDeclarations": function_declarations}]

        if request.extra_params:
            payload.update(request.extra_params)

        return payload

    async def generate(self, request: LLMRequest) -> LLMResponse:
        key = request.api_key or self.api_key
        model = request.model
        if not model.startswith("models/"):
            model = f"models/{model}"

        url = f"{request.base_url or self.base_url}/{model}:generateContent"
        params = {"key": key} if key else {}
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "APIMagic-LLM-Test-Framework/0.1.0",
        }
        if request.extra_headers:
            headers.update(request.extra_headers)

        payload = self._build_payload(request)

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=request.timeout) as client:
            try:
                response = await client.post(url, headers=headers, params=params, json=payload)
            except Exception as e:
                raise LLMClientError(f"HTTP connection failed: {str(e)}") from e

        if response.status_code == 400 and "API key" in response.text:
            raise AuthenticationError(f"Gemini API Key Error (400): {response.text}", status_code=400)
        elif response.status_code == 429:
            raise RateLimitError(f"Gemini Rate Limit (429): {response.text}", status_code=429)
        elif response.is_error:
            raise LLMClientError(f"Gemini Error ({response.status_code}): {response.text}", status_code=response.status_code)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        data = response.json()

        candidates = data.get("candidates", [{}])
        candidate = candidates[0] if candidates else {}
        content_obj = candidate.get("content", {})
        parts = content_obj.get("parts", [])

        content_parts = []
        tool_calls: List[ToolCall] = []

        for p in parts:
            if "text" in p:
                content_parts.append(p["text"])
            elif "functionCall" in p:
                fc = p["functionCall"]
                tool_calls.append(
                    ToolCall(
                        name=fc.get("name", ""),
                        arguments=fc.get("args", {}),
                    )
                )

        text_content = "".join(content_parts) if content_parts else None

        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        tokens = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=usage.get("totalTokenCount", prompt_tokens + completion_tokens),
            cached_tokens=usage.get("cachedContentTokenCount", 0),
        )

        cost = calculate_cost(request.model, tokens.prompt_tokens, tokens.completion_tokens)

        return LLMResponse(
            content=text_content,
            tool_calls=tool_calls,
            role=MessageRole.ASSISTANT,
            finish_reason=candidate.get("finishReason", "STOP"),
            latency_ms=round(latency_ms, 2),
            tokens=tokens,
            cost_usd=cost,
            model=request.model,
            provider="gemini",
            raw_response=data,
        )

    async def generate_stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        # Simple streaming fallback wrapping generate for gemini
        resp = await self.generate(request)
        if resp.content:
            yield StreamChunk(
                delta_content=resp.content,
                finish_reason="STOP",
                latency_from_start_ms=resp.latency_ms,
            )
