"""Core exceptions for APIMagic LLM testing framework."""

class APIMagicError(Exception):
    """Base exception for all APIMagic errors."""
    pass

class LLMClientError(APIMagicError):
    """Raised when an LLM provider client fails."""
    def __init__(self, message: str, status_code: int | None = None, response_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body

class RateLimitError(LLMClientError):
    """Raised when an LLM provider returns 429 Too Many Requests."""
    pass

class AuthenticationError(LLMClientError):
    """Raised when API credentials are missing or invalid."""
    pass

class EvaluationError(APIMagicError):
    """Raised when an evaluator fails to execute."""
    pass

class SchemaValidationError(EvaluationError):
    """Raised when structured response schema validation fails."""
    pass

class SuiteConfigurationError(APIMagicError):
    """Raised when test suite definition is malformed."""
    pass
