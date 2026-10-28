"""Example 3: Function & Tool Calling Verification."""

import asyncio
from apimagic import (
    LLMRequest,
    MockLLMClient,
    ToolCall,
    ToolCallEvaluator,
    ToolDefinition,
    ToolFunctionDef,
)

async def main():
    client = MockLLMClient()
    # Mock LLM calls get_flight tool when prompt mentions flights
    client.add_rule(
        pattern=r"flights? to (\w+)",
        tool_calls=[
            ToolCall(
                name="search_flights",
                arguments={"destination": "Tokyo", "max_stops": 0, "currency": "USD"},
            )
        ],
    )

    request = LLMRequest(
        messages=[LLMRequest.from_prompt("Find direct flights to Tokyo").messages[0]],
        tools=[
            ToolDefinition(
                function=ToolFunctionDef(
                    name="search_flights",
                    description="Search for available commercial flights",
                    parameters={
                        "type": "object",
                        "properties": {
                            "destination": {"type": "string"},
                            "max_stops": {"type": "integer"},
                        },
                        "required": ["destination"],
                    },
                )
            )
        ],
        model="gpt-4o",
    )

    response = await client.generate(request)
    print("LLM invoked tools:")
    for tc in response.tool_calls:
        print(f"  Tool: {tc.name} with args: {tc.arguments}")

    # Evaluate tool invocation
    evaluator = ToolCallEvaluator(
        expected_name="search_flights",
        expected_arguments={"destination": "Tokyo", "max_stops": 0},
    )

    res = await evaluator.evaluate(response)
    print(f"\nEvaluation Status: {'PASS' if res.passed else 'FAIL'}")
    print(f"Details: {res.message}")

if __name__ == "__main__":
    asyncio.run(main())
