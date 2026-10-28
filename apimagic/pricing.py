"""Pricing table and cost calculator for LLM requests."""

from typing import Dict, Tuple

# Pricing per 1,000,000 tokens (USD)
# (prompt_price_per_1m, completion_price_per_1m)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),

    # Anthropic
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),

    # Google Gemini
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-exp": (0.0, 0.0),

    # Mock / Local
    "mock": (0.0, 0.0),
    "ollama": (0.0, 0.0),
    "local": (0.0, 0.0),
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates the estimated cost in USD for a given model and token count.

    Args:
        model: Model identifier.
        prompt_tokens: Number of prompt/input tokens.
        completion_tokens: Number of completion/output tokens.

    Returns:
        Estimated cost in USD (rounded to 6 decimal places).
    """
    cleaned_model = model.lower().strip()

    rates = None
    for key, val in MODEL_PRICING.items():
        if key in cleaned_model:
            rates = val
            break

    if not rates:
        # Fallback default: $1.00 / $3.00 per 1M tokens
        rates = (1.00, 3.00)

    prompt_cost = (prompt_tokens / 1_000_000.0) * rates[0]
    completion_cost = (completion_tokens / 1_000_000.0) * rates[1]
    return round(prompt_cost + completion_cost, 6)
