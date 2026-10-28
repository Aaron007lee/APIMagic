"""Example 1: Basic LLM API test execution using Python API."""

import asyncio
from apimagic import (
    LLMRequest,
    MockLLMClient,
    ContainsEvaluator,
    LatencyEvaluator,
    CostBudgetEvaluator,
)

async def main():
    # 1. Setup client (or use OpenAIClient, AnthropicClient, GeminiClient)
    client = MockLLMClient()
    client.add_rule(
        pattern=r"capital.*France",
        response_content="The capital of France is Paris.",
    )

    # 2. Build LLM request
    request = LLMRequest.from_prompt(
        prompt="What is the capital of France?",
        model="gpt-4o-mini",
    )

    # 3. Call the model
    print(f"Calling model {request.model}...")
    response = await client.generate(request)
    print(f"Response: {response.content}")
    print(f"Latency: {response.latency_ms:.1f}ms | Cost: ${response.cost_usd:.6f}")

    # 4. Run evaluators
    evaluators = [
        ContainsEvaluator(expected=["Paris"]),
        LatencyEvaluator(max_latency_ms=1000.0),
        CostBudgetEvaluator(max_cost_usd=0.01),
    ]

    print("\nEvaluations:")
    for ev in evaluators:
        res = await ev.evaluate(response)
        status = "PASS" if res.passed else "FAIL"
        print(f"  [{status}] {res.name}: {res.message}")

if __name__ == "__main__":
    asyncio.run(main())
