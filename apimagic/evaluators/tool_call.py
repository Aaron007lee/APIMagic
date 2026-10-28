"""Tool and function calling evaluator."""

from typing import Any, Dict, List, Optional
from apimagic.core.models import LLMResponse, MetricResult, ToolCall
from apimagic.evaluators.base import BaseEvaluator


class ToolCallEvaluator(BaseEvaluator):
    """Evaluates whether the expected tool(s) were called with valid arguments."""

    def __init__(
        self,
        expected_name: Optional[str] = None,
        expected_arguments: Optional[Dict[str, Any]] = None,
        min_calls: int = 1,
        max_calls: Optional[int] = None,
        expect_no_calls: bool = False,
    ):
        super().__init__(name="tool_call_match")
        self.expected_name = expected_name
        self.expected_arguments = expected_arguments or {}
        self.min_calls = 0 if expect_no_calls else min_calls
        self.max_calls = 0 if expect_no_calls else max_calls
        self.expect_no_calls = expect_no_calls

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        actual_calls = response.tool_calls or []
        call_count = len(actual_calls)

        if self.expect_no_calls:
            passed = (call_count == 0)
            return MetricResult(
                name=self.name,
                passed=passed,
                score=1.0 if passed else 0.0,
                threshold=1.0,
                message="No tools called as expected" if passed else f"Expected 0 tool calls, but {call_count} were called",
                details={"actual_calls": [tc.model_dump() for tc in actual_calls]},
            )

        # Check call counts
        if call_count < self.min_calls:
            return MetricResult(
                name=self.name,
                passed=False,
                score=0.0,
                threshold=1.0,
                message=f"Expected at least {self.min_calls} tool calls, got {call_count}",
                details={"actual_calls": [tc.model_dump() for tc in actual_calls]},
            )

        if self.max_calls is not None and call_count > self.max_calls:
            return MetricResult(
                name=self.name,
                passed=False,
                score=0.0,
                threshold=1.0,
                message=f"Expected at most {self.max_calls} tool calls, got {call_count}",
                details={"actual_calls": [tc.model_dump() for tc in actual_calls]},
            )

        # If specific tool name requested
        target_name = expected if isinstance(expected, str) else self.expected_name
        if target_name:
            matching_calls = [tc for tc in actual_calls if tc.name == target_name]
            if not matching_calls:
                return MetricResult(
                    name=self.name,
                    passed=False,
                    score=0.0,
                    threshold=1.0,
                    message=f"Expected tool '{target_name}' was not called. Called: {[tc.name for tc in actual_calls]}",
                    details={"actual_calls": [tc.model_dump() for tc in actual_calls]},
                )

            # Check arguments if provided
            if self.expected_arguments:
                arg_failures = []
                best_match = matching_calls[0]
                actual_args = best_match.arguments if isinstance(best_match.arguments, dict) else {}

                for key, val in self.expected_arguments.items():
                    if key not in actual_args:
                        arg_failures.append(f"Missing argument '{key}'")
                    elif actual_args[key] != val:
                        arg_failures.append(f"Argument '{key}': expected {val!r}, got {actual_args[key]!r}")

                if arg_failures:
                    return MetricResult(
                        name=self.name,
                        passed=False,
                        score=0.5,
                        threshold=1.0,
                        message=f"Tool '{target_name}' arguments mismatch: {'; '.join(arg_failures)}",
                        details={"actual_args": actual_args, "expected_args": self.expected_arguments},
                    )

        return MetricResult(
            name=self.name,
            passed=True,
            score=1.0,
            threshold=1.0,
            message="Tool call validation passed",
            details={"calls": [tc.model_dump() for tc in actual_calls]},
        )
