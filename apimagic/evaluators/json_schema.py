"""JSON Schema and Pydantic structured output evaluator."""

import json
import re
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, ValidationError

from apimagic.core.models import LLMResponse, MetricResult
from apimagic.evaluators.base import BaseEvaluator


def extract_json_from_text(text: str) -> Any:
    """Extracts JSON from text, handling markdown code fences if present."""
    if not text:
        raise ValueError("Empty response text")

    text = text.strip()

    # Look for ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()

    return json.loads(text)


class JsonSchemaEvaluator(BaseEvaluator):
    """Evaluates whether the response content is valid JSON conforming to a schema or Pydantic model."""

    def __init__(
        self,
        schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        required_keys: Optional[List[str]] = None,
        key_values: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="json_schema_valid")
        self.schema = schema
        self.required_keys = required_keys or []
        self.key_values = key_values or {}

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        content = response.content or ""
        target_schema = expected if expected is not None else self.schema

        try:
            parsed = extract_json_from_text(content)
        except Exception as e:
            return MetricResult(
                name=self.name,
                passed=False,
                score=0.0,
                threshold=1.0,
                message=f"Invalid JSON in response: {str(e)}",
                details={"raw_content": content, "error": str(e)},
            )

        # Check required keys if dict
        if isinstance(parsed, dict) and self.required_keys:
            missing_keys = [k for k in self.required_keys if k not in parsed]
            if missing_keys:
                return MetricResult(
                    name=self.name,
                    passed=False,
                    score=0.0,
                    threshold=1.0,
                    message=f"JSON missing required keys: {missing_keys}",
                    details={"parsed": parsed, "missing_keys": missing_keys},
                )

        # Check key_values if specified
        if isinstance(parsed, dict) and self.key_values:
            mismatches = {}
            for k, expected_v in self.key_values.items():
                if k not in parsed or parsed[k] != expected_v:
                    mismatches[k] = {"expected": expected_v, "actual": parsed.get(k)}
            if mismatches:
                return MetricResult(
                    name=self.name,
                    passed=False,
                    score=0.0,
                    threshold=1.0,
                    message=f"JSON values mismatch: {mismatches}",
                    details={"parsed": parsed, "mismatches": mismatches},
                )

        # Validate with Pydantic model
        if target_schema is not None:
            if isinstance(target_schema, type) and issubclass(target_schema, BaseModel):
                try:
                    if isinstance(parsed, dict):
                        target_schema.model_validate(parsed)
                    else:
                        raise ValueError(f"Expected JSON object for model {target_schema.__name__}, got {type(parsed)}")
                except ValidationError as ve:
                    return MetricResult(
                        name=self.name,
                        passed=False,
                        score=0.0,
                        threshold=1.0,
                        message=f"Pydantic validation failed: {ve.errors()}",
                        details={"errors": ve.errors(), "parsed": parsed},
                    )
            elif isinstance(target_schema, dict):
                # Basic JSON Schema validation
                req = target_schema.get("required", [])
                if isinstance(parsed, dict) and req:
                    missing = [f for f in req if f not in parsed]
                    if missing:
                        return MetricResult(
                            name=self.name,
                            passed=False,
                            score=0.0,
                            threshold=1.0,
                            message=f"JSON schema required fields missing: {missing}",
                            details={"parsed": parsed, "missing": missing},
                        )

        return MetricResult(
            name=self.name,
            passed=True,
            score=1.0,
            threshold=1.0,
            message="JSON is valid and conforms to expected structure",
            details={"parsed": parsed},
        )
