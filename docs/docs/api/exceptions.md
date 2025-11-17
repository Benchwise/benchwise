---
sidebar_position: 9
---

# Exceptions

Custom exception classes for Benchwise.

## Exception Hierarchy

```python
BenchWiseError (base)
├── ConfigurationError
├── ModelError
├── DatasetError
└── EvaluationError
```

## BenchWiseError

Base exception for all Benchwise errors.

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

## DatasetError

Dataset loading and validation errors.

```python
from benchwise.exceptions import DatasetError

raise DatasetError("Invalid dataset format")
```

## EvaluationError

Evaluation execution errors.

```python
from benchwise.exceptions import EvaluationError

raise EvaluationError("Evaluation failed")
```

## See Also

- [Error Handling Guide](../advanced/error-handling.md)
