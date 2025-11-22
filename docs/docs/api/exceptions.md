---
sidebar_position: 9
---

# Exceptions

Custom exception classes for Benchwise.

## Exception Hierarchy

```
BenchwiseError (base)
├── AuthenticationError
├── RateLimitError
├── ValidationError
├── NetworkError
├── ConfigurationError
├── DatasetError
├── ModelError
└── MetricError
```

## BenchwiseError

Base exception for all Benchwise errors.

## AuthenticationError

Raised when authentication fails.

```python
from benchwise.exceptions import AuthenticationError

raise AuthenticationError("Invalid API key or token")
```

## RateLimitError

Raised when API rate limit is exceeded.

```python
from benchwise.exceptions import RateLimitError

raise RateLimitError("API rate limit exceeded, please try again later")
```

## ValidationError

Raised when input validation fails.

```python
from benchwise.exceptions import ValidationError

raise ValidationError("Invalid input data provided")
```

## NetworkError

Raised when network requests fail.

```python
from benchwise.exceptions import NetworkError

raise NetworkError("Failed to connect to the Benchwise API")
```

## ConfigurationError

Configuration and setup errors.

```python
from benchwise.exceptions import ConfigurationError

raise ConfigurationError("Invalid API key")
```

## ModelError

Model adapter and API errors.

```python
from benchwise.exceptions import ModelError

raise ModelError("Failed to generate response")
```

## MetricError

Raised when metric calculation fails.

```python
from benchwise.exceptions import MetricError

raise MetricError("Metric calculation failed")
```

## DatasetError

Dataset loading and validation errors.

```python
from benchwise.exceptions import DatasetError

raise DatasetError("Invalid dataset format")
```



## See Also

- [Error Handling Guide](../advanced/error-handling.md)
