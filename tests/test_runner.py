"""Unit tests for TestRunner, suite loader, and multi-trial statistics."""

import pytest
from apimagic.client.mock_client import MockLLMClient
from apimagic.core.models import AssertionConfig, LLMRequest, TestCaseConfig
from apimagic.runner.engine import TestRunner
from apimagic.runner.loader import load_suite_from_dict
from apimagic.runner.pytest_plugin import assert_llm


@pytest.mark.asyncio
async def test_runner_multi_trials():
    client = MockLLMClient(default_response="Healthy output")

    case = TestCaseConfig(
        id="stability_check",
        name="Stability Test",
        request=LLMRequest.from_prompt("Ping"),
        assertions=[
            AssertionConfig(type="contains", expected="Healthy"),
            AssertionConfig(type="latency", threshold=500.0),
        ],
        trials=3,
        pass_threshold=1.0,
    )

    runner = TestRunner(client=client, concurrency=2)
    res = await runner.run_case(case)

    assert res.passed
    assert res.pass_rate == 1.0
    assert len(res.trials) == 3
    assert res.avg_latency_ms > 0


@pytest.mark.asyncio
async def test_runner_suite_execution():
    client = MockLLMClient(default_response="API response")

    cases = [
        TestCaseConfig(
            id=f"case_{i}",
            request=LLMRequest.from_prompt(f"Prompt {i}"),
            assertions=[AssertionConfig(type="contains", expected="API")],
        )
        for i in range(5)
    ]

    runner = TestRunner(client=client, concurrency=3)
    suite_res = await runner.run_suite(cases, suite_name="Batch Test")

    assert suite_res.total_cases == 5
    assert suite_res.passed_cases == 5
    assert suite_res.pass_rate == 1.0


def test_suite_loader():
    suite_data = {
        "name": "Integration Suite",
        "defaults": {"provider": "mock", "model": "mock-gpt-4o"},
        "cases": [
            {
                "id": "c1",
                "prompt": "Test query",
                "assertions": [{"type": "contains", "expected": ["Test"]}],
            }
        ],
    }
    name, cases = load_suite_from_dict(suite_data)
    assert name == "Integration Suite"
    assert len(cases) == 1
    assert cases[0].id == "c1"


def test_pytest_fluent_assertions():
    from apimagic.core.models import LLMResponse, TokenUsage

    resp = LLMResponse(
        content='{"status": "ok", "code": 200}',
        latency_ms=120.0,
        tokens=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        cost_usd=0.0001,
    )

    # Fluent chain should pass
    (
        assert_llm(resp)
        .contains("status")
        .not_contains("error")
        .has_json_keys(["status", "code"])
        .latency_under(500.0)
        .token_budget(max_total=100)
        .cost_under(0.01)
        .called_no_tools()
    )

    # Failure should raise AssertionError
    with pytest.raises(AssertionError):
        assert_llm(resp).contains("non_existent_key")
