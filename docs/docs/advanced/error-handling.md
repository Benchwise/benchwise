---
sidebar_position: 4
---

# Error Handling

Handle errors gracefully in evaluations.

## Automatic Error Handling

Understand how Benchwise automatically manages and reports errors during evaluations.

Benchwise automatically handles errors in evaluations:

```python
import asyncio
from benchwise import evaluate

@evaluate("gpt-4", "invalid-model")
async def my_test(model, dataset):
    responses = await model.generate(dataset.prompts)
    return {"responses": responses}

results = asyncio.run(my_test(dataset))

# Check for failures
for result in results:
    if not result.success:
        print(f"Error in {result.model_name}: {result.error}")
```

## Custom Error Handling

Implement custom error handling logic within your evaluation functions.

```python
@evaluate("gpt-4")
async def robust_test(model, dataset):
    try:
        responses = await model.generate(dataset.prompts)
        return {"responses": responses}
    except Exception as e:
        # Custom error handling
        return {"error": str(e), "partial_results": None}
```

## Retry Logic

Strategies for implementing retry mechanisms for robust evaluations.

```python
import asyncio

@evaluate("gpt-4")
async def test_with_retry(model, dataset):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            responses = await model.generate(dataset.prompts)
            return {"responses": responses}
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## Custom Exceptions

Learn about Benchwise's custom exception classes for more granular error management.

```python
from benchwise.exceptions import BenchwiseError, ModelError, DatasetError

try:
    dataset = load_dataset("invalid.json")
except DatasetError as e:
    print(f"Dataset error: {e}")

try:
    responses = await model.generate(prompts)
except ModelError as e:
    print(f"Model error: {e}")
```

## See Also

- [Exceptions API](../api/exceptions.md)
