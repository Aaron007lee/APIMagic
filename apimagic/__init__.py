"""APIMagic: Modern LLM API Testing and Evaluation Framework in Python."""

__version__ = "0.1.0"

from apimagic.core.models import (
    AssertionConfig,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    MessageRole,
    MetricResult,
    ResponseFormat,
    SingleTrialResult,
    TestCaseConfig,
    TestCaseResult,
    TestSuiteResult,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolFunctionDef,
)
from apimagic.client import (
    AnthropicClient,
    BaseLLMClient,
    GeminiClient,
    MockLLMClient,
    OpenAIClient,
    get_client,
)
from apimagic.evaluators import (
    BaseEvaluator,
    ContainsEvaluator,
    CostBudgetEvaluator,
    ExactMatchEvaluator,
    JsonSchemaEvaluator,
    LatencyEvaluator,
    LengthEvaluator,
    LLMJudgeEvaluator,
    NotContainsEvaluator,
    RAGFaithfulnessEvaluator,
    RegexEvaluator,
    SafetyGuardEvaluator,
    SimilarityEvaluator,
    TokenBudgetEvaluator,
    ToolCallEvaluator,
    TTFTEvaluator,
)
from apimagic.pricing import calculate_cost
from apimagic.runner import (
    TestRunner,
    assert_llm,
    load_suite_from_dict,
    load_suite_from_file,
    load_suites_from_path,
)

__all__ = [
    # Core
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "MessageRole",
    "MetricResult",
    "ResponseFormat",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "ToolFunctionDef",
    "TestCaseConfig",
    "TestCaseResult",
    "TestSuiteResult",
    "SingleTrialResult",
    "AssertionConfig",
    # Clients
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "MockLLMClient",
    "get_client",
    # Pricing
    "calculate_cost",
    # Evaluators
    "BaseEvaluator",
    "ExactMatchEvaluator",
    "ContainsEvaluator",
    "NotContainsEvaluator",
    "RegexEvaluator",
    "LengthEvaluator",
    "JsonSchemaEvaluator",
    "ToolCallEvaluator",
    "LatencyEvaluator",
    "TTFTEvaluator",
    "TokenBudgetEvaluator",
    "CostBudgetEvaluator",
    "SimilarityEvaluator",
    "RAGFaithfulnessEvaluator",
    "SafetyGuardEvaluator",
    "LLMJudgeEvaluator",
    # Runner & Pytest
    "TestRunner",
    "assert_llm",
    "load_suite_from_dict",
    "load_suite_from_file",
    "load_suites_from_path",
]
