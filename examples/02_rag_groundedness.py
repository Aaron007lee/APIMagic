"""Example 2: RAG Groundedness & Hallucination Evaluation."""

import asyncio
from apimagic import LLMResponse, RAGFaithfulnessEvaluator

async def main():
    context = (
        "Project Titan is an open-source distributed cache developed in 2024. "
        "It supports in-memory clustering with automatic replication across 3 nodes "
        "and provides sub-millisecond read latencies."
    )

    # Response A: Grounded in context
    good_response = LLMResponse(
        content="Project Titan is an open-source distributed cache from 2024 with 3-node automatic replication."
    )

    # Response B: Hallucinated / ungrounded claims
    hallucinated_response = LLMResponse(
        content="Project Titan was created by Microsoft in 1999 to replace Redis and uses quantum computing."
    )

    evaluator = RAGFaithfulnessEvaluator(min_groundedness=0.60)

    print("--- Evaluating Grounded Response ---")
    res_a = await evaluator.evaluate(good_response, context=context)
    print(f"Passed: {res_a.passed} | Score: {res_a.score:.1%} | {res_a.message}")

    print("\n--- Evaluating Hallucinated Response ---")
    res_b = await evaluator.evaluate(hallucinated_response, context=context)
    print(f"Passed: {res_b.passed} | Score: {res_b.score:.1%} | {res_b.message}")
    print(f"Unsupported tokens detected: {res_b.details.get('unsupported_sample_tokens')}")

if __name__ == "__main__":
    asyncio.run(main())
