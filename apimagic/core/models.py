"""Core domain data models for APIMagic LLM testing framework."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolFunctionDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: str = "function"
    function: ToolFunctionDef


class ToolCall(BaseModel):
    id: Optional[str] = None
    type: str = "function"
    name: str
    arguments: Union[Dict[str, Any], str] = Field(default_factory=dict)


class LLMMessage(BaseModel):
    role: MessageRole
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

    @classmethod
    def system(cls, content: str) -> "LLMMessage":
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "LLMMessage":
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: Optional[str] = None, tool_calls: Optional[List[ToolCall]] = None) -> "LLMMessage":
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: Optional[str] = None) -> "LLMMessage":
        return cls(role=MessageRole.TOOL, content=content, tool_call_id=tool_call_id, name=name)


class ResponseFormat(BaseModel):
    type: str = "text"  # "text" | "json_object" | "json_schema"
    json_schema: Optional[Dict[str, Any]] = None


class LLMRequest(BaseModel):
    messages: List[LLMMessage]
    model: str = "mock-gpt-4o"
    provider: str = "mock"  # "openai", "anthropic", "gemini", "mock", "custom"
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[ResponseFormat] = None
    stream: bool = False
    timeout: float = 60.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    extra_params: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_prompt(cls, prompt: str, system: Optional[str] = None, **kwargs) -> "LLMRequest":
        msgs: List[LLMMessage] = []
        if system:
            msgs.append(LLMMessage.system(system))
        msgs.append(LLMMessage.user(prompt))
        return cls(messages=msgs, **kwargs)


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    @model_validator(mode="after")
    def compute_total(self) -> "TokenUsage":
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self


class LLMResponse(BaseModel):
    content: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    role: MessageRole = MessageRole.ASSISTANT
    finish_reason: Optional[str] = "stop"
    latency_ms: float = 0.0
    ttft_ms: Optional[float] = None  # Time to first token for streaming
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    cost_usd: float = 0.0
    model: str = ""
    provider: str = ""
    raw_response: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class StreamChunk(BaseModel):
    delta_content: Optional[str] = None
    delta_tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    latency_from_start_ms: float = 0.0


class MetricResult(BaseModel):
    name: str
    passed: bool
    score: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)


class AssertionConfig(BaseModel):
    type: str  # e.g. "contains", "exact_match", "regex", "json_schema", "tool_call", "latency", "cost", "similarity", "faithfulness", "llm_judge"
    expected: Optional[Any] = None
    threshold: Optional[float] = None
    metric_name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class TestCaseConfig(BaseModel):
    __test__ = False
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    request: LLMRequest
    assertions: List[AssertionConfig] = Field(default_factory=list)
    trials: int = 1  # For non-determinism and stability testing
    pass_threshold: float = 1.0  # Fraction of trials that must pass (e.g. 0.8 = 80%)
    context: Optional[str] = None  # Reference context for RAG faithfulness
    mock_response: Optional[str] = None  # Optional simulated response for mock/offline testing
    mock_tool_calls: Optional[List[ToolCall]] = None  # Optional simulated tool calls for mock testing


class SingleTrialResult(BaseModel):
    trial_number: int
    passed: bool
    latency_ms: float
    response: Optional[LLMResponse] = None
    metrics: List[MetricResult] = Field(default_factory=list)
    error: Optional[str] = None


class TestCaseResult(BaseModel):
    __test__ = False
    id: str
    name: str
    tags: List[str] = Field(default_factory=list)
    passed: bool
    pass_rate: float = 0.0  # e.g. 1.0 = 100%
    trials: List[SingleTrialResult] = Field(default_factory=list)
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    metrics_summary: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    error: Optional[str] = None


class TestSuiteResult(BaseModel):
    __test__ = False
    suite_name: str
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    total_trials: int = 0
    pass_rate: float = 0.0
    duration_ms: float = 0.0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    case_results: List[TestCaseResult] = Field(default_factory=list)
