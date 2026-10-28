"""Example 4: LLM-as-a-Judge Rubric Evaluation."""

import asyncio
from apimagic import LLMResponse, LLMJudgeEvaluator, MockLLMClient

async def main():
    # Setup mock judge client (or use OpenAIClient(api_key="sk-...") in production)
    judge_client = MockLLMClient(
        default_response='{"score": 0.92, "passed": true, "reasoning": "Clear, professional, and directly answered the customer question with actionable next steps."}'
    )

    candidate_response = LLMResponse(
        content=(
            "Thank you for contacting billing support. We have initiated a refund of $49.99 "
            "to your original payment method. Please allow 3-5 business days for your bank to process it."
        )
    )

    evaluator = LLMJudgeEvaluator(
        rubric=(
            "Assess the response on two dimensions:\n"
            "1. Tone: Must be polite, empathetic, and professional.\n"
            "2. Clarity: Must state the specific refund amount and expected timeframe."
        ),
        judge_client=judge_client,
        judge_model="gpt-4o",
        threshold=0.80,
    )

    res = await evaluator.evaluate(
        response=candidate_response,
        user_prompt="I was charged twice, can I get a refund?",
    )

    print("--- LLM Judge Evaluation Result ---")
    print(f"Passed: {res.passed}")
    print(f"Score: {res.score:.2f} (Threshold: {res.threshold})")
    print(f"Reasoning: {res.message}")

if __name__ == "__main__":
    asyncio.run(main())
