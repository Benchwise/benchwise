"""
Benchwise Custom Exceptions

Provides specific exception types for better error handling.
"""


class BenchwiseError(Exception):
    """Base exception for all Benchwise errors."""
    pass


class AuthenticationError(BenchwiseError):
    """Raised when authentication fails."""
    pass


class RateLimitError(BenchwiseError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after


class ValidationError(BenchwiseError):
    """Raised when input validation fails."""
    pass


class NetworkError(BenchwiseError):
    """Raised when network requests fail."""
    pass


class ConfigurationError(BenchwiseError):
    """Raised when configuration is invalid or missing."""
    pass


class DatasetError(BenchwiseError):
    """Raised when dataset operations fail."""
    pass


class ModelError(BenchwiseError):
    """Raised when model operations fail."""
    pass


class MetricError(BenchwiseError):
    """Raised when metric calculation fails."""
    pass
