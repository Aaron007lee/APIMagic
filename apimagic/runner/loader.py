"""Test suite loader for YAML, JSON, and directories."""

import glob
import json
import os
from typing import Any, Dict, List, Tuple
import yaml

from apimagic.core.exceptions import SuiteConfigurationError
from apimagic.core.models import (
    AssertionConfig,
    LLMMessage,
    LLMRequest,
    MessageRole,
    ResponseFormat,
    TestCaseConfig,
    ToolCall,
    ToolDefinition,
    ToolFunctionDef,
)


def _parse_case(raw: Dict[str, Any], defaults: Dict[str, Any], index: int) -> TestCaseConfig:
    case_id = raw.get("id", f"case_{index + 1}")
    name = raw.get("name", case_id)
    description = raw.get("description")
    tags = raw.get("tags", [])
    trials = raw.get("trials", defaults.get("trials", 1))
    pass_threshold = raw.get("pass_threshold", defaults.get("pass_threshold", 1.0))
    context = raw.get("context")

    # Build messages
    messages: List[LLMMessage] = []
    system_text = raw.get("system", defaults.get("system"))
    if system_text:
        messages.append(LLMMessage.system(system_text))

    if "messages" in raw:
        for m in raw["messages"]:
            role = MessageRole(m.get("role", "user"))
            messages.append(LLMMessage(role=role, content=m.get("content")))
    elif "prompt" in raw:
        messages.append(LLMMessage.user(raw["prompt"]))
    else:
        raise SuiteConfigurationError(f"Test case '{case_id}' missing 'prompt' or 'messages'")

    # Tools
    tools = None
    if "tools" in raw:
        tools = []
        for t in raw["tools"]:
            fn = t.get("function", t)
            tools.append(
                ToolDefinition(
                    type=t.get("type", "function"),
                    function=ToolFunctionDef(
                        name=fn["name"],
                        description=fn.get("description"),
                        parameters=fn.get("parameters", {}),
                    ),
                )
            )

    # Response format
    resp_format = None
    if "response_format" in raw:
        rf = raw["response_format"]
        if isinstance(rf, str):
            resp_format = ResponseFormat(type=rf)
        elif isinstance(rf, dict):
            resp_format = ResponseFormat(**rf)

    mock_tool_calls = None
    if "mock_tool_calls" in raw:
        mock_tool_calls = [
            ToolCall(**tc) if isinstance(tc, dict) else tc
            for tc in raw["mock_tool_calls"]
        ]

    request = LLMRequest(
        messages=messages,
        model=raw.get("model", defaults.get("model", "mock-gpt-4o")),
        provider=raw.get("provider", defaults.get("provider", "mock")),
        temperature=raw.get("temperature", defaults.get("temperature", 0.0)),
        max_tokens=raw.get("max_tokens", defaults.get("max_tokens", 1024)),
        top_p=raw.get("top_p", defaults.get("top_p")),
        tools=tools,
        tool_choice=raw.get("tool_choice"),
        response_format=resp_format,
        stream=raw.get("stream", defaults.get("stream", False)),
        timeout=raw.get("timeout", defaults.get("timeout", 60.0)),
        base_url=raw.get("base_url", defaults.get("base_url")),
        api_key=raw.get("api_key", defaults.get("api_key")),
        extra_headers=raw.get("extra_headers", defaults.get("extra_headers", {})),
        extra_params=raw.get("extra_params", defaults.get("extra_params", {})),
    )

    assertions: List[AssertionConfig] = []
    for a in raw.get("assertions", []):
        if isinstance(a, dict):
            assertions.append(
                AssertionConfig(
                    type=a.get("type", "contains"),
                    expected=a.get("expected"),
                    threshold=a.get("threshold"),
                    metric_name=a.get("metric_name"),
                    params=a.get("params", {}),
                )
            )

    return TestCaseConfig(
        id=case_id,
        name=name,
        description=description,
        tags=tags,
        request=request,
        assertions=assertions,
        trials=trials,
        pass_threshold=pass_threshold,
        context=context,
        mock_response=raw.get("mock_response"),
        mock_tool_calls=mock_tool_calls,
    )


def load_suite_from_dict(data: Dict[str, Any]) -> Tuple[str, List[TestCaseConfig]]:
    """Loads a suite name and test cases from a dictionary."""
    suite_name = data.get("name", "APIMagic Suite")
    defaults = data.get("defaults", {})
    raw_cases = data.get("cases", [])

    if not isinstance(raw_cases, list):
        raise SuiteConfigurationError("Expected 'cases' to be a list in suite definition")

    cases = [_parse_case(c, defaults, i) for i, c in enumerate(raw_cases)]
    return suite_name, cases


def load_suite_from_file(file_path: str) -> Tuple[str, List[TestCaseConfig]]:
    """Loads test suite from a YAML or JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Suite file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if file_path.endswith((".yaml", ".yml")):
        data = yaml.safe_load(content)
    elif file_path.endswith(".json"):
        data = json.loads(content)
    else:
        # Try YAML first, then JSON
        try:
            data = yaml.safe_load(content)
        except Exception:
            data = json.loads(content)

    if not isinstance(data, dict):
        raise SuiteConfigurationError(f"Root object in {file_path} must be a dictionary/mapping")

    return load_suite_from_dict(data)


def load_suites_from_path(path: str) -> List[Tuple[str, List[TestCaseConfig]]]:
    """Loads test suites from a file or directory path."""
    suites = []
    if os.path.isdir(path):
        patterns = [os.path.join(path, "*.yaml"), os.path.join(path, "*.yml"), os.path.join(path, "*.json")]
        files = []
        for p in patterns:
            files.extend(glob.glob(p))
        for f in sorted(files):
            suites.append(load_suite_from_file(f))
    else:
        suites.append(load_suite_from_file(path))
    return suites
