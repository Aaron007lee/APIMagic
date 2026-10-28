"""Unit tests for APIMagic data models and pricing calculation."""

import pytest
from apimagic.core.models import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    MessageRole,
    ResponseFormat,
    TokenUsage,
    ToolCall,
)
from apimagic.pricing import calculate_cost


def test_llm_message_factories():
    sys_msg = LLMMessage.system("System prompt")
    assert sys_msg.role == MessageRole.SYSTEM
    assert sys_msg.content == "System prompt"

    usr_msg = LLMMessage.user("Hello world")
    assert usr_msg.role == MessageRole.USER
    assert usr_msg.content == "Hello world"

    tc = ToolCall(name="fetch_user", arguments={"user_id": 42})
    asst_msg = LLMMessage.assistant("Done", tool_calls=[tc])
    assert asst_msg.role == MessageRole.ASSISTANT
    assert len(asst_msg.tool_calls) == 1
    assert asst_msg.tool_calls[0].name == "fetch_user"


def test_token_usage_total_auto_compute():
    tu = TokenUsage(prompt_tokens=150, completion_tokens=50)
    assert tu.total_tokens == 200


def test_pricing_calculation():
    # gpt-4o: $2.50 / $10.00 per 1M tokens
    cost = calculate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=500)
    # (1000 / 1e6)*2.50 = 0.0025; (500 / 1e6)*10.0 = 0.0050 => total 0.0075
    assert cost == pytest.approx(0.0075, rel=1e-3)

    # claude-3-5-sonnet: $3.00 / $15.00 per 1M tokens
    cost_claude = calculate_cost("claude-3-5-sonnet-20241022", prompt_tokens=2000, completion_tokens=1000)
    # (2000 / 1e6)*3.0 + (1000 / 1e6)*15.0 = 0.006 + 0.015 = 0.021
    assert cost_claude == pytest.approx(0.021, rel=1e-3)


def test_llm_request_from_prompt():
    req = LLMRequest.from_prompt("Summarize this text", system="You are an editor", model="gpt-4o")
    assert len(req.messages) == 2
    assert req.messages[0].role == MessageRole.SYSTEM
    assert req.messages[1].role == MessageRole.USER
    assert req.model == "gpt-4o"
