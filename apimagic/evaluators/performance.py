"""Performance, latency, and budget evaluators."""

from typing import Any, Optional
from apimagic.core.models import LLMResponse, MetricResult
from apimagic.evaluators.base import BaseEvaluator


class LatencyEvaluator(BaseEvaluator):
    """Evaluates whether end-to-end response latency is within the threshold."""

    def __init__(self, max_latency_ms: float = 2000.0):
        super().__init__(name="latency_ms")
        self.max_latency_ms = max_latency_ms

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        threshold = float(expected) if expected is not None else self.max_latency_ms
        actual_latency = response.latency_ms
        passed = (actual_latency <= threshold)

        score = max(0.0, min(1.0, 1.0 - (actual_latency / (threshold * 2.0)))) if threshold > 0 else 1.0

        return MetricResult(
            name=self.name,
            passed=passed,
            score=round(score, 3),
            threshold=threshold,
            message=f"Latency {actual_latency:.1f}ms <= {threshold:.1f}ms" if passed else f"Latency {actual_latency:.1f}ms exceeded max {threshold:.1f}ms",
            details={"actual_latency_ms": actual_latency, "max_latency_ms": threshold},
        )


class TTFTEvaluator(BaseEvaluator):
    """Evaluates Time to First Token (TTFT) for streaming responses."""

    def __init__(self, max_ttft_ms: float = 800.0):
        super().__init__(name="ttft_ms")
        self.max_ttft_ms = max_ttft_ms

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        threshold = float(expected) if expected is not None else self.max_ttft_ms
        ttft = response.ttft_ms

        if ttft is None:
            return MetricResult(
                name=self.name,
                passed=True,
                score=1.0,
                threshold=threshold,
                message="TTFT not measured (non-streaming request)",
                details={"ttft_ms": None},
            )

        passed = (ttft <= threshold)
        return MetricResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            threshold=threshold,
            message=f"TTFT {ttft:.1f}ms <= {threshold:.1f}ms" if passed else f"TTFT {ttft:.1f}ms exceeded max {threshold:.1f}ms",
            details={"actual_ttft_ms": ttft, "max_ttft_ms": threshold},
        )


class TokenBudgetEvaluator(BaseEvaluator):
    """Evaluates token usage against prompt, completion, or total token budgets."""

    def __init__(
        self,
        max_prompt_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ):
        super().__init__(name="token_budget")
        self.max_prompt_tokens = max_prompt_tokens
        self.max_completion_tokens = max_completion_tokens
        self.max_total_tokens = max_total_tokens

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        tokens = response.tokens
        violations = []

        max_total = int(expected) if expected is not None else self.max_total_tokens

        if self.max_prompt_tokens is not None and tokens.prompt_tokens > self.max_prompt_tokens:
            violations.append(f"Prompt tokens {tokens.prompt_tokens} > max {self.max_prompt_tokens}")

        if self.max_completion_tokens is not None and tokens.completion_tokens > self.max_completion_tokens:
            violations.append(f"Completion tokens {tokens.completion_tokens} > max {self.max_completion_tokens}")

        if max_total is not None and tokens.total_tokens > max_total:
            violations.append(f"Total tokens {tokens.total_tokens} > max {max_total}")

        passed = len(violations) == 0
        return MetricResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            threshold=float(max_total) if max_total is not None else None,
            message="Token budget respected" if passed else f"Budget violations: {', '.join(violations)}",
            details={
                "prompt_tokens": tokens.prompt_tokens,
                "completion_tokens": tokens.completion_tokens,
                "total_tokens": tokens.total_tokens,
                "violations": violations,
            },
        )


class CostBudgetEvaluator(BaseEvaluator):
    """Evaluates request monetary cost against a maximum USD threshold."""

    def __init__(self, max_cost_usd: float = 0.05):
        super().__init__(name="cost_budget")
        self.max_cost_usd = max_cost_usd

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        threshold = float(expected) if expected is not None else self.max_cost_usd
        cost = response.cost_usd
        passed = (cost <= threshold)

        return MetricResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            threshold=threshold,
            message=f"Cost ${cost:.6f} <= ${threshold:.6f}" if passed else f"Cost ${cost:.6f} exceeded max ${threshold:.6f}",
            details={"actual_cost_usd": cost, "max_cost_usd": threshold},
        )
