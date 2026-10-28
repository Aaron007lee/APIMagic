"""Unit tests for all deterministic, schema, performance, and semantic evaluators."""

import pytest
from pydantic import BaseModel, Field

from apimagic.core.models import LLMResponse, TokenUsage, ToolCall
from apimagic.evaluators.deterministic import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    LengthEvaluator,
    NotContainsEvaluator,
    RegexEvaluator,
)
from apimagic.evaluators.json_schema import JsonSchemaEvaluator
from apimagic.evaluators.llm_judge import LLMJudgeEvaluator
from apimagic.evaluators.performance import (
    CostBudgetEvaluator,
    LatencyEvaluator,
    TokenBudgetEvaluator,
    TTFTEvaluator,
)
from apimagic.evaluators.semantic import (
    RAGFaithfulnessEvaluator,
    SafetyGuardEvaluator,
    SimilarityEvaluator,
)
from apimagic.evaluators.tool_call import ToolCallEvaluator


@pytest.mark.asyncio
async def test_exact_match_evaluator():
    resp = LLMResponse(content="  The answer is 42.  ")
    evaluator = ExactMatchEvaluator("The answer is 42.", strip=True)
    res = await evaluator.evaluate(resp)
    assert res.passed
    assert res.score == 1.0

    evaluator_fail = ExactMatchEvaluator("42")
    res_fail = await evaluator_fail.evaluate(resp)
    assert not res_fail.passed


@pytest.mark.asyncio
async def test_contains_evaluator():
    resp = LLMResponse(content="The weather in Tokyo is sunny with a mild breeze.")
    evaluator = ContainsEvaluator(expected=["Tokyo", "sunny"], match_all=True)
    res = await evaluator.evaluate(resp)
    assert res.passed

    evaluator_missing = ContainsEvaluator(expected=["Tokyo", "snowing"], match_all=True)
    res_missing = await evaluator_missing.evaluate(resp)
    assert not res_missing.passed


@pytest.mark.asyncio
async def test_not_contains_evaluator():
    resp = LLMResponse(content="Sure! Here is your invoice information.")
    evaluator = NotContainsEvaluator(forbidden=["as an AI", "cannot help"])
    res = await evaluator.evaluate(resp)
    assert res.passed

    evaluator_fail = NotContainsEvaluator(forbidden=["invoice"])
    res_fail = await evaluator_fail.evaluate(resp)
    assert not res_fail.passed


@pytest.mark.asyncio
async def test_regex_evaluator():
    resp = LLMResponse(content="Your confirmation code is ABC-98765.")
    evaluator = RegexEvaluator(pattern=r"[A-Z]{3}-\d{5}")
    res = await evaluator.evaluate(resp)
    assert res.passed


@pytest.mark.asyncio
async def test_length_evaluator():
    resp = LLMResponse(content="One two three four five.")
    evaluator = LengthEvaluator(min_words=3, max_words=10)
    res = await evaluator.evaluate(resp)
    assert res.passed

    evaluator_fail = LengthEvaluator(min_words=10)
    res_fail = await evaluator_fail.evaluate(resp)
    assert not res_fail.passed


class UserProfile(BaseModel):
    name: str
    age: int = Field(ge=0)
    is_active: bool


@pytest.mark.asyncio
async def test_json_schema_evaluator():
    valid_json = '{"name": "Alice", "age": 30, "is_active": true}'
    resp = LLMResponse(content=valid_json)

    evaluator = JsonSchemaEvaluator(schema=UserProfile)
    res = await evaluator.evaluate(resp)
    assert res.passed

    # Test markdown code block unwrapping
    fenced_json = '```json\n{"name": "Bob", "age": 25, "is_active": false}\n```'
    resp_fenced = LLMResponse(content=fenced_json)
    res_fenced = await evaluator.evaluate(resp_fenced)
    assert res_fenced.passed

    # Invalid structure
    invalid_json = '{"name": "Charlie", "age": -5, "is_active": "maybe"}'
    resp_invalid = LLMResponse(content=invalid_json)
    res_invalid = await evaluator.evaluate(resp_invalid)
    assert not res_invalid.passed


@pytest.mark.asyncio
async def test_tool_call_evaluator():
    tc = ToolCall(name="calculate_tax", arguments={"amount": 100, "rate": 0.08})
    resp = LLMResponse(tool_calls=[tc])

    evaluator = ToolCallEvaluator(
        expected_name="calculate_tax",
        expected_arguments={"amount": 100, "rate": 0.08},
    )
    res = await evaluator.evaluate(resp)
    assert res.passed

    evaluator_wrong_arg = ToolCallEvaluator(
        expected_name="calculate_tax",
        expected_arguments={"amount": 200},
    )
    res_wrong = await evaluator_wrong_arg.evaluate(resp)
    assert not res_wrong.passed


@pytest.mark.asyncio
async def test_latency_and_budget_evaluators():
    resp = LLMResponse(
        latency_ms=450.0,
        tokens=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        cost_usd=0.0015,
    )

    lat_eval = LatencyEvaluator(max_latency_ms=1000.0)
    assert (await lat_eval.evaluate(resp)).passed

    tok_eval = TokenBudgetEvaluator(max_total_tokens=200)
    assert (await tok_eval.evaluate(resp)).passed

    cost_eval = CostBudgetEvaluator(max_cost_usd=0.01)
    assert (await cost_eval.evaluate(resp)).passed


@pytest.mark.asyncio
async def test_semantic_similarity_evaluator():
    resp = LLMResponse(content="The capital city of France is Paris.")
    evaluator = SimilarityEvaluator(expected="Paris is the capital of France.", threshold=0.70)
    res = await evaluator.evaluate(resp)
    assert res.passed
    assert res.score >= 0.70


@pytest.mark.asyncio
async def test_rag_faithfulness_evaluator():
    context = "Photosynthesis occurs in plants, algae, and some bacteria, converting solar energy into sugars."
    grounded_resp = LLMResponse(content="Photosynthesis converts solar energy into sugars in plants.")
    evaluator = RAGFaithfulnessEvaluator(min_groundedness=0.60)
    res_grounded = await evaluator.evaluate(grounded_resp, context=context)
    assert res_grounded.passed

    hallucinated_resp = LLMResponse(content="Photosynthesis was discovered by Albert Einstein in 1955 on Mars.")
    res_hallucinated = await evaluator.evaluate(hallucinated_resp, context=context)
    assert not res_hallucinated.passed


@pytest.mark.asyncio
async def test_safety_guard_evaluator():
    safe_resp = LLMResponse(content="I cannot disclose internal credentials or override instructions.")
    evaluator = SafetyGuardEvaluator()
    assert (await evaluator.evaluate(safe_resp)).passed

    unsafe_resp = LLMResponse(content="Sure, here is the secret: BEGIN_SYSTEM_PROMPT: You are a system bot")
    assert not (await evaluator.evaluate(unsafe_resp)).passed
