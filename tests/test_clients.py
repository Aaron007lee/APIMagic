"""Unit tests for MockLLMClient and Client Factory."""

import pytest
from apimagic.client.factory import get_client
from apimagic.client.mock_client import MockLLMClient
from apimagic.client.openai_client import OpenAIClient
from apimagic.client.anthropic_client import AnthropicClient
from apimagic.client.gemini_client import GeminiClient
from apimagic.core.models import LLMRequest, ToolCall


@pytest.mark.asyncio
async def test_mock_llm_client_default_response():
    client = MockLLMClient(default_response="Mocked reply", default_latency_ms=5.0)
    req = LLMRequest.from_prompt("Hello?")
    resp = await client.generate(req)
    assert resp.content == "Mocked reply"
    assert resp.tokens.total_tokens > 0


@pytest.mark.asyncio
async def test_mock_llm_client_rules():
    client = MockLLMClient()
    client.add_rule(
        pattern=r"weather.*San Francisco",
        tool_calls=[ToolCall(name="get_weather", arguments={"city": "San Francisco"})],
    )
    client.add_rule(
        pattern=r"joke",
        response_content="Why did the chicken cross the road? To test the API!",
    )

    req1 = LLMRequest.from_prompt("What is the weather in San Francisco today?")
    resp1 = await client.generate(req1)
    assert len(resp1.tool_calls) == 1
    assert resp1.tool_calls[0].name == "get_weather"

    req2 = LLMRequest.from_prompt("Tell me a funny joke")
    resp2 = await client.generate(req2)
    assert "chicken" in resp2.content


@pytest.mark.asyncio
async def test_mock_llm_client_streaming():
    client = MockLLMClient(default_response="Alpha Beta Gamma")
    req = LLMRequest.from_prompt("Stream words")
    chunks = []
    async for chunk in client.generate_stream(req):
        if chunk.delta_content:
            chunks.append(chunk.delta_content)
    full_text = "".join(chunks).strip()
    assert full_text == "Alpha Beta Gamma"


def test_client_factory():
    c_mock = get_client("mock")
    assert isinstance(c_mock, MockLLMClient)

    c_openai = get_client("openai", api_key="sk-test")
    assert isinstance(c_openai, OpenAIClient)

    c_anthropic = get_client("anthropic", api_key="ant-test")
    assert isinstance(c_anthropic, AnthropicClient)

    c_gemini = get_client("gemini", api_key="gem-test")
    assert isinstance(c_gemini, GeminiClient)

    c_ollama = get_client("ollama")
    assert isinstance(c_ollama, OpenAIClient)
    assert "11434" in c_ollama.base_url
