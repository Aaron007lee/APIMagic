"""Deterministic assertions and evaluators (exact match, contains, regex, length)."""

import re
from typing import Any, List, Optional, Union
from apimagic.core.models import LLMResponse, MetricResult
from apimagic.evaluators.base import BaseEvaluator


class ExactMatchEvaluator(BaseEvaluator):
    """Evaluates whether the response content exactly matches expected text."""

    def __init__(self, expected: str, case_sensitive: bool = False, strip: bool = True):
        super().__init__(name="exact_match")
        self.expected = expected.strip() if strip else expected
        self.case_sensitive = case_sensitive
        self.strip = strip

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        exp = expected if expected is not None else self.expected
        content = response.content or ""
        if self.strip:
            content = content.strip()
            if isinstance(exp, str):
                exp = exp.strip()

        if not self.case_sensitive:
            passed = (str(content).lower() == str(exp).lower())
        else:
            passed = (str(content) == str(exp))

        return MetricResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            threshold=1.0,
            message="Matches expected exact text" if passed else f"Expected '{exp}', got '{content}'",
            details={"actual": content, "expected": exp},
        )


class ContainsEvaluator(BaseEvaluator):
    """Evaluates whether the response contains expected substring(s)."""

    def __init__(
        self,
        expected: Union[str, List[str]],
        case_sensitive: bool = False,
        match_all: bool = True,
    ):
        super().__init__(name="contains")
        self.expected = [expected] if isinstance(expected, str) else expected
        self.case_sensitive = case_sensitive
        self.match_all = match_all

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        targets = expected if expected is not None else self.expected
        if isinstance(targets, str):
            targets = [targets]

        content = response.content or ""
        search_content = content if self.case_sensitive else content.lower()

        matched: List[str] = []
        missing: List[str] = []

        for item in targets:
            query = str(item) if self.case_sensitive else str(item).lower()
            if query in search_content:
                matched.append(str(item))
            else:
                missing.append(str(item))

        if self.match_all:
            passed = len(missing) == 0
        else:
            passed = len(matched) > 0

        score = len(matched) / max(1, len(targets))

        msg = f"Contained {len(matched)}/{len(targets)} required strings."
        if missing:
            msg += f" Missing: {missing}"

        return MetricResult(
            name=self.name,
            passed=passed,
            score=score,
            threshold=1.0 if self.match_all else (1.0 / len(targets)),
            message=msg,
            details={"matched": matched, "missing": missing, "match_all": self.match_all},
        )


class NotContainsEvaluator(BaseEvaluator):
    """Evaluates whether the response does NOT contain forbidden substring(s)."""

    def __init__(self, forbidden: Union[str, List[str]], case_sensitive: bool = False):
        super().__init__(name="not_contains")
        self.forbidden = [forbidden] if isinstance(forbidden, str) else forbidden
        self.case_sensitive = case_sensitive

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        forbidden_list = expected if expected is not None else self.forbidden
        if isinstance(forbidden_list, str):
            forbidden_list = [forbidden_list]

        content = response.content or ""
        search_content = content if self.case_sensitive else content.lower()

        found: List[str] = []
        for item in forbidden_list:
            query = str(item) if self.case_sensitive else str(item).lower()
            if query in search_content:
                found.append(str(item))

        passed = len(found) == 0
        return MetricResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            threshold=1.0,
            message="No forbidden words found" if passed else f"Forbidden words found: {found}",
            details={"forbidden_found": found},
        )


class RegexEvaluator(BaseEvaluator):
    """Evaluates whether the response matches a regular expression pattern."""

    def __init__(self, pattern: str, flags: int = 0):
        super().__init__(name="regex_match")
        self.pattern_str = pattern
        self.regex = re.compile(pattern, flags)

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        pattern = expected or self.pattern_str
        regex = re.compile(pattern) if expected else self.regex

        content = response.content or ""
        match = regex.search(content)
        passed = match is not None

        return MetricResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            threshold=1.0,
            message=f"Matched regex '{pattern}'" if passed else f"Did not match regex '{pattern}'",
            details={"pattern": pattern, "matched_group": match.group(0) if match else None},
        )


class LengthEvaluator(BaseEvaluator):
    """Evaluates response character or word length limits."""

    def __init__(
        self,
        min_chars: Optional[int] = None,
        max_chars: Optional[int] = None,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
    ):
        super().__init__(name="length_bounds")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.min_words = min_words
        self.max_words = max_words

    async def evaluate(
        self,
        response: LLMResponse,
        context: Optional[str] = None,
        expected: Optional[Any] = None,
        **kwargs
    ) -> MetricResult:
        content = response.content or ""
        char_len = len(content)
        word_len = len(content.split()) if content else 0

        failures = []
        if self.min_chars is not None and char_len < self.min_chars:
            failures.append(f"chars {char_len} < min {self.min_chars}")
        if self.max_chars is not None and char_len > self.max_chars:
            failures.append(f"chars {char_len} > max {self.max_chars}")
        if self.min_words is not None and word_len < self.min_words:
            failures.append(f"words {word_len} < min {self.min_words}")
        if self.max_words is not None and word_len > self.max_words:
            failures.append(f"words {word_len} > max {self.max_words}")

        passed = len(failures) == 0
        return MetricResult(
            name=self.name,
            passed=passed,
            score=1.0 if passed else 0.0,
            threshold=1.0,
            message="Length constraints met" if passed else f"Length violations: {', '.join(failures)}",
            details={"char_length": char_len, "word_length": word_len, "failures": failures},
        )
