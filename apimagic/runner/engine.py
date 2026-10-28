"""Async execution engine for LLM test suites with retries, concurrency, and multi-trial statistics."""

import asyncio
import math
import re
import time
from typing import Callable, List, Optional
from apimagic.client.base import BaseLLMClient
from apimagic.client.factory import get_client
from apimagic.client.mock_client import MockLLMClient
from apimagic.core.exceptions import RateLimitError
from apimagic.core.models import (
    AssertionConfig,
    MessageRole,
    MetricResult,
    SingleTrialResult,
    TestCaseConfig,
    TestCaseResult,
    TestSuiteResult,
)
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


def build_evaluator_from_config(config: AssertionConfig):
    """Instantiates an evaluator from an AssertionConfig object."""
    atype = config.type.lower()
    p = config.params

    if atype in ("exact_match", "equals"):
        return ExactMatchEvaluator(
            expected=config.expected or "",
            case_sensitive=p.get("case_sensitive", False),
            strip=p.get("strip", True),
        )
    elif atype in ("contains", "includes"):
        return ContainsEvaluator(
            expected=config.expected or [],
            case_sensitive=p.get("case_sensitive", False),
            match_all=p.get("match_all", True),
        )
    elif atype in ("not_contains", "excludes", "forbidden"):
        return NotContainsEvaluator(
            forbidden=config.expected or [],
            case_sensitive=p.get("case_sensitive", False),
        )
    elif atype in ("regex", "pattern"):
        return RegexEvaluator(pattern=str(config.expected or ""))
    elif atype in ("length", "bounds"):
        return LengthEvaluator(
            min_chars=p.get("min_chars"),
            max_chars=p.get("max_chars"),
            min_words=p.get("min_words"),
            max_words=p.get("max_words"),
        )
    elif atype in ("json_schema", "schema", "json"):
        return JsonSchemaEvaluator(
            schema=config.expected if isinstance(config.expected, dict) else None,
            required_keys=p.get("required_keys"),
            key_values=p.get("key_values"),
        )
    elif atype in ("tool_call", "function_call", "tool"):
        return ToolCallEvaluator(
            expected_name=str(config.expected) if config.expected else p.get("name"),
            expected_arguments=p.get("arguments"),
            min_calls=p.get("min_calls", 1),
            max_calls=p.get("max_calls"),
            expect_no_calls=p.get("expect_no_calls", False),
        )
    elif atype in ("latency", "latency_ms", "max_latency"):
        max_lat = float(config.expected or config.threshold or 2000.0)
        return LatencyEvaluator(max_latency_ms=max_lat)
    elif atype in ("ttft", "ttft_ms"):
        max_ttft = float(config.expected or config.threshold or 800.0)
        return TTFTEvaluator(max_ttft_ms=max_ttft)
    elif atype in ("token_budget", "tokens"):
        return TokenBudgetEvaluator(
            max_prompt_tokens=p.get("max_prompt_tokens"),
            max_completion_tokens=p.get("max_completion_tokens"),
            max_total_tokens=p.get("max_total_tokens") or (int(config.expected) if config.expected else None),
        )
    elif atype in ("cost", "cost_usd", "budget"):
        max_c = float(config.expected or config.threshold or 0.05)
        return CostBudgetEvaluator(max_cost_usd=max_c)
    elif atype in ("similarity", "semantic_similarity"):
        thresh = float(config.threshold or p.get("threshold", 0.70))
        return SimilarityEvaluator(expected=str(config.expected or ""), threshold=thresh)
    elif atype in ("faithfulness", "rag_faithfulness", "groundedness"):
        min_g = float(config.threshold or p.get("threshold", 0.65))
        return RAGFaithfulnessEvaluator(min_groundedness=min_g)
    elif atype in ("safety", "safety_guard", "jailbreak_check"):
        return SafetyGuardEvaluator(forbidden_patterns=p.get("patterns"))
    elif atype in ("llm_judge", "judge"):
        thresh = float(config.threshold or p.get("threshold", 0.70))
        return LLMJudgeEvaluator(rubric=str(config.expected or p.get("rubric", "")), threshold=thresh)
    else:
        raise ValueError(f"Unknown assertion type: {config.type}")


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculates percentile (0 to 100) from list of numbers."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    d0 = sorted_v[int(f)] * (c - k)
    d1 = sorted_v[int(c)] * (k - f)
    return d0 + d1


class TestRunner:
    """Asynchronous test runner executing cases with rate-limiting, retries, and statistics."""
    __test__ = False

    def __init__(
        self,
        client: Optional[BaseLLMClient] = None,
        concurrency: int = 4,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
    ):
        self.client = client
        self.concurrency = concurrency
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    async def _execute_with_retry(self, client: BaseLLMClient, req):
        """Executes LLM call with exponential backoff on rate limits."""
        for attempt in range(self.max_retries + 1):
            try:
                return await client.generate(req)
            except RateLimitError:
                if attempt == self.max_retries:
                    raise
                wait_time = self.retry_delay_seconds * (2 ** attempt)
                await asyncio.sleep(wait_time)
            except Exception:
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(self.retry_delay_seconds)

    async def run_single_trial(
        self,
        case: TestCaseConfig,
        trial_num: int,
        client: BaseLLMClient,
    ) -> SingleTrialResult:
        """Runs a single trial for a test case."""
        start = time.perf_counter()
        try:
            response = await self._execute_with_retry(client, case.request)
            latency = (time.perf_counter() - start) * 1000.0

            # Evaluate all assertions
            metric_results: List[MetricResult] = []
            prompt_text = " ".join([m.content or "" for m in case.request.messages])

            for assertion_cfg in case.assertions:
                evaluator = build_evaluator_from_config(assertion_cfg)
                res = await evaluator.evaluate(
                    response=response,
                    context=case.context,
                    expected=assertion_cfg.expected,
                    user_prompt=prompt_text,
                )
                metric_results.append(res)

            all_passed = all(m.passed for m in metric_results) if metric_results else True

            return SingleTrialResult(
                trial_number=trial_num,
                passed=all_passed,
                latency_ms=round(latency, 2),
                response=response,
                metrics=metric_results,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000.0
            return SingleTrialResult(
                trial_number=trial_num,
                passed=False,
                latency_ms=round(latency, 2),
                error=str(e),
                metrics=[],
            )

    async def run_case(
        self,
        case: TestCaseConfig,
        client: Optional[BaseLLMClient] = None,
    ) -> TestCaseResult:
        """Runs all trials for a test case under the concurrency semaphore."""
        async with self.semaphore:
            active_client = client or self.client or get_client(case.request.provider)

            if isinstance(active_client, MockLLMClient) and (case.mock_response is not None or case.mock_tool_calls is not None):
                user_msgs = [m.content or "" for m in case.request.messages if m.role == MessageRole.USER]
                if user_msgs:
                    active_client.add_rule(
                        pattern=re.escape(user_msgs[0]),
                        response_content=case.mock_response,
                        tool_calls=case.mock_tool_calls,
                    )

            trials: List[SingleTrialResult] = []

            for i in range(1, case.trials + 1):
                trial_res = await self.run_single_trial(case, i, active_client)
                trials.append(trial_res)

            passed_trials = sum(1 for t in trials if t.passed)
            pass_rate = passed_trials / max(1, len(trials))
            case_passed = (pass_rate >= case.pass_threshold)

            latencies = [t.latency_ms for t in trials]
            avg_lat = sum(latencies) / max(1, len(latencies))
            p95_lat = calculate_percentile(latencies, 95)
            min_lat = min(latencies) if latencies else 0.0
            max_lat = max(latencies) if latencies else 0.0

            total_cost = sum(t.response.cost_usd for t in trials if t.response)

            # Metrics summary aggregation
            metrics_summary = {}
            for t in trials:
                for m in t.metrics:
                    if m.name not in metrics_summary:
                        metrics_summary[m.name] = {"passed": 0, "total": 0, "scores": []}
                    metrics_summary[m.name]["total"] += 1
                    if m.passed:
                        metrics_summary[m.name]["passed"] += 1
                    if m.score is not None:
                        metrics_summary[m.name]["scores"].append(m.score)

            errors = [t.error for t in trials if t.error]

            return TestCaseResult(
                id=case.id,
                name=case.name or case.id,
                tags=case.tags,
                passed=case_passed,
                pass_rate=round(pass_rate, 3),
                trials=trials,
                avg_latency_ms=round(avg_lat, 2),
                p95_latency_ms=round(p95_lat, 2),
                min_latency_ms=round(min_lat, 2),
                max_latency_ms=round(max_lat, 2),
                total_cost_usd=round(total_cost, 6),
                metrics_summary=metrics_summary,
                error="; ".join(errors) if errors else None,
            )

    async def run_suite(
        self,
        cases: List[TestCaseConfig],
        suite_name: str = "APIMagic Test Suite",
        on_case_complete: Optional[Callable[[TestCaseResult], None]] = None,
    ) -> TestSuiteResult:
        """Executes an entire suite of test cases concurrently."""
        start_time = time.perf_counter()

        tasks = []
        for case in cases:
            task = asyncio.create_task(self.run_case(case))
            if on_case_complete:
                task.add_done_callback(lambda t: on_case_complete(t.result()))
            tasks.append(task)

        results: List[TestCaseResult] = await asyncio.gather(*tasks)

        total_duration = (time.perf_counter() - start_time) * 1000.0

        total_cases = len(results)
        passed_cases = sum(1 for r in results if r.passed)
        failed_cases = total_cases - passed_cases
        total_trials = sum(len(r.trials) for r in results)
        suite_pass_rate = (passed_cases / total_cases) if total_cases > 0 else 1.0

        all_latencies = [t.latency_ms for r in results for t in r.trials]
        avg_lat = sum(all_latencies) / max(1, len(all_latencies)) if all_latencies else 0.0
        p95_lat = calculate_percentile(all_latencies, 95) if all_latencies else 0.0
        total_cost = sum(r.total_cost_usd for r in results)

        return TestSuiteResult(
            suite_name=suite_name,
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            total_trials=total_trials,
            pass_rate=round(suite_pass_rate, 3),
            duration_ms=round(total_duration, 2),
            total_cost_usd=round(total_cost, 6),
            avg_latency_ms=round(avg_lat, 2),
            p95_latency_ms=round(p95_lat, 2),
            case_results=results,
        )
