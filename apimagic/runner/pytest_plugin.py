"""Pytest plugin fixtures, markers, and fluent assertion builder."""

from typing import Any, Dict, List, Optional, Type, Union
import pytest
from pydantic import BaseModel

from apimagic.client.base import BaseLLMClient
from apimagic.client.factory import get_client
from apimagic.client.mock_client import MockLLMClient
from apimagic.core.models import LLMResponse
from apimagic.evaluators.deterministic import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    LengthEvaluator,
    NotContainsEvaluator,
    RegexEvaluator,
)
from apimagic.evaluators.json_schema import JsonSchemaEvaluator
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


class LLMAssertionBuilder:
    """Fluent assertion helper for validating LLM responses in pytest."""

    def __init__(self, response: LLMResponse):
        self.response = response

    def _assert_metric(self, evaluator, **kwargs) -> "LLMAssertionBuilder":
        res = evaluator.evaluate_sync(self.response, **kwargs)
        if not res.passed:
            raise AssertionError(f"LLM Assertion [{res.name}] Failed: {res.message}. Details: {res.details}")
        return self

    def equals(self, expected: str, case_sensitive: bool = False, strip: bool = True) -> "LLMAssertionBuilder":
        evaluator = ExactMatchEvaluator(expected=expected, case_sensitive=case_sensitive, strip=strip)
        return self._assert_metric(evaluator)

    def contains(self, expected: Union[str, List[str]], case_sensitive: bool = False, match_all: bool = True) -> "LLMAssertionBuilder":
        evaluator = ContainsEvaluator(expected=expected, case_sensitive=case_sensitive, match_all=match_all)
        return self._assert_metric(evaluator)

    def not_contains(self, forbidden: Union[str, List[str]], case_sensitive: bool = False) -> "LLMAssertionBuilder":
        evaluator = NotContainsEvaluator(forbidden=forbidden, case_sensitive=case_sensitive)
        return self._assert_metric(evaluator)

    def matches(self, pattern: str) -> "LLMAssertionBuilder":
        evaluator = RegexEvaluator(pattern=pattern)
        return self._assert_metric(evaluator)

    def length_bounds(
        self,
        min_chars: Optional[int] = None,
        max_chars: Optional[int] = None,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
    ) -> "LLMAssertionBuilder":
        evaluator = LengthEvaluator(
            min_chars=min_chars,
            max_chars=max_chars,
            min_words=min_words,
            max_words=max_words,
        )
        return self._assert_metric(evaluator)

    def matches_schema(
        self,
        schema: Union[Dict[str, Any], Type[BaseModel]],
        required_keys: Optional[List[str]] = None,
    ) -> "LLMAssertionBuilder":
        evaluator = JsonSchemaEvaluator(schema=schema, required_keys=required_keys)
        return self._assert_metric(evaluator)

    def has_json_keys(self, required_keys: List[str]) -> "LLMAssertionBuilder":
        evaluator = JsonSchemaEvaluator(required_keys=required_keys)
        return self._assert_metric(evaluator)

    def called_tool(
        self,
        name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        min_calls: int = 1,
        max_calls: Optional[int] = None,
    ) -> "LLMAssertionBuilder":
        evaluator = ToolCallEvaluator(
            expected_name=name,
            expected_arguments=arguments,
            min_calls=min_calls,
            max_calls=max_calls,
        )
        return self._assert_metric(evaluator)

    def called_no_tools(self) -> "LLMAssertionBuilder":
        evaluator = ToolCallEvaluator(expect_no_calls=True)
        return self._assert_metric(evaluator)

    def latency_under(self, max_latency_ms: float) -> "LLMAssertionBuilder":
        evaluator = LatencyEvaluator(max_latency_ms=max_latency_ms)
        return self._assert_metric(evaluator)

    def ttft_under(self, max_ttft_ms: float) -> "LLMAssertionBuilder":
        evaluator = TTFTEvaluator(max_ttft_ms=max_ttft_ms)
        return self._assert_metric(evaluator)

    def token_budget(
        self,
        max_total: Optional[int] = None,
        max_prompt: Optional[int] = None,
        max_completion: Optional[int] = None,
    ) -> "LLMAssertionBuilder":
        evaluator = TokenBudgetEvaluator(
            max_total_tokens=max_total,
            max_prompt_tokens=max_prompt,
            max_completion_tokens=max_completion,
        )
        return self._assert_metric(evaluator)

    def cost_under(self, max_cost_usd: float) -> "LLMAssertionBuilder":
        evaluator = CostBudgetEvaluator(max_cost_usd=max_cost_usd)
        return self._assert_metric(evaluator)

    def similar_to(self, expected_text: str, threshold: float = 0.70) -> "LLMAssertionBuilder":
        evaluator = SimilarityEvaluator(expected=expected_text, threshold=threshold)
        return self._assert_metric(evaluator)

    def faithful_to_context(self, context: str, min_groundedness: float = 0.65) -> "LLMAssertionBuilder":
        evaluator = RAGFaithfulnessEvaluator(min_groundedness=min_groundedness)
        return self._assert_metric(evaluator, context=context)

    def safe_from_injection(self) -> "LLMAssertionBuilder":
        evaluator = SafetyGuardEvaluator()
        return self._assert_metric(evaluator)


def assert_llm(response: LLMResponse) -> LLMAssertionBuilder:
    """Entry point for fluent LLM assertions."""
    return LLMAssertionBuilder(response)


# Pytest Fixtures
@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Fixture providing a configurable MockLLMClient."""
    return MockLLMClient()


@pytest.fixture
def llm_client() -> BaseLLMClient:
    """Fixture providing an LLM client (defaults to mock, or configured via env)."""
    return get_client("mock")
