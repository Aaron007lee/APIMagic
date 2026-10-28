"""LLM-as-a-Judge evaluator for qualitative and rubric-based evaluations."""

import json
from typing import Any, Optional
from apimagic.client.base import BaseLLMClient
from apimagic.client.mock_client import MockLLMClient
from apimagic.core.models import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
    MetricResult,
    ResponseFormat,
)
from apimagic.evaluators.base import BaseEvaluator
from apimagic.evaluators.json_schema import extract_json_from_text

DEFAULT_JUDGE_PROMPT = """You are an expert impartial AI evaluator.
Your task is to evaluate an LLM's response based on the evaluation criteria.

[Prompt / User Input]:
{prompt}

[Context (if any)]:
{context}

[Candidate LLM Response]:
{response}

[Evaluation Criteria / Rubric]:
{rubric}

You must return valid JSON only, with the following keys:
- "score": a number from 0.0 to 1.0 (where 1.0 is exceptional and 0.0 is completely inadequate)
- "passed": boolean (true if the response meets or exceeds the required threshold, false otherwise)
- "reasoning": a concise explanation of your score and evaluation.
"""


class LLMJudgeEvaluator(BaseEvaluator):
    """Evaluates response using another LLM as a judge against a specific rubric."""

    def __init__(
        self,
        rubric: str,
        judge_client: Optional[BaseLLMClient] = None,
        judge_model: str = "gpt-4o",
        threshold: float = 0.70,
        judge_prompt_template: str = DEFAULT_JUDGE_PROMPT,
    ):
        super().__init__(name="llm_judge")
        self.rubric = rubric
        self.judge_client = judge_client or MockLLMClient(
            default_response='{"score": 0.95, "passed": true, "reasoning": "Output followed all instructions and maintained professional tone."}'
        )
        self.judge_model = judge_model
        self.threshold = threshold
        self.template = judge_prompt_template

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        user_prompt: Optional[str] = None,
        **kwargs
    ) -> MetricResult:
        rubric = expected if isinstance(expected, str) else self.rubric

        rendered_prompt = self.template.format(
            prompt=user_prompt or "(N/A)",
            context=context or "(None)",
            response=response.content or "",
            rubric=rubric,
        )

        judge_req = LLMRequest(
            messages=[
                LLMMessage.system("You are a strict evaluation judge. Output valid JSON only."),
                LLMMessage.user(rendered_prompt),
            ],
            model=self.judge_model,
            temperature=0.0,
            response_format=ResponseFormat(type="json_object"),
        )

        try:
            judge_resp = await self.judge_client.generate(judge_req)
            parsed = extract_json_from_text(judge_resp.content or "{}")

            score = float(parsed.get("score", 0.0))
            reasoning = str(parsed.get("reasoning", "No reasoning provided"))
            passed_flag = bool(parsed.get("passed", score >= self.threshold))
            passed = passed_flag and (score >= self.threshold)

            return MetricResult(
                name=self.name,
                passed=passed,
                score=score,
                threshold=self.threshold,
                message=reasoning,
                details={
                    "score": score,
                    "reasoning": reasoning,
                    "judge_model": self.judge_model,
                    "rubric": rubric,
                },
            )
        except Exception as e:
            return MetricResult(
                name=self.name,
                passed=False,
                score=0.0,
                threshold=self.threshold,
                message=f"Judge evaluation failed: {str(e)}",
                details={"error": str(e)},
            )
