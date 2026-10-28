"""Example 5: Pytest Integration with assert_llm fluent assertions."""

import pytest
from pydantic import BaseModel
from apimagic import LLMRequest, MockLLMClient, assert_llm

class FlightInfo(BaseModel):
    airline: str
    price: float
    seats_available: int

@pytest.fixture
def mock_travel_agent():
    client = MockLLMClient()
    client.add_rule(
        pattern=r"cheap flight",
        response_content='{"airline": "SkyWays", "price": 289.50, "seats_available": 4}',
    )
    return client

@pytest.mark.asyncio
async def test_flight_search_structured_output(mock_travel_agent):
    req = LLMRequest.from_prompt("Find me a cheap flight to Chicago")
    response = await mock_travel_agent.generate(req)

    # Fluent assertion pipeline
    (
        assert_llm(response)
        .matches_schema(FlightInfo)
        .has_json_keys(["airline", "price"])
        .latency_under(500.0)
        .token_budget(max_total=200)
        .cost_under(0.005)
        .safe_from_injection()
    )
