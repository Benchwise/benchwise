---
sidebar_position: 2
---

# API Integration

Integrate with the Benchwise platform API.

## Setup

How to configure Benchwise for API integration.

```python
from benchwise import configure_benchwise

configure_benchwise(
    api_url="https://api.benchwise.ai",
    api_key="your_api_key",
    upload_enabled=True
)
```

## Auto-Upload Results

Automatically upload evaluation results to the Benchwise platform.

```python
from benchwise import evaluate, benchmark, accuracy

# Note: Evaluation functions should be async to perform model generation
@benchmark("My Benchmark", "Description")
@evaluate("gpt-4", upload=True)
async def my_test(model, dataset):
    # Generate responses asynchronously
    responses = await model.generate(dataset.prompts)
    scores = accuracy(responses, dataset.references)
    # Results will be automatically uploaded
    return {"accuracy": scores["accuracy"]}
```

## Manual Upload

Manually upload evaluation results for fine-grained control.

```python
import asyncio
from benchwise.client import upload_results
from benchwise import BenchmarkResult, EvaluationResult

# Create benchmark and add results
benchmark = BenchmarkResult(
    name="My Benchmark",
    metadata={"version": "1.0", "date": "2024-11-16"}
)

# Add evaluation results
result = EvaluationResult(
    model_name="gpt-4",
    result={"accuracy": 0.85, "latency": 1.2},
    success=True,
    duration=10.5
)
benchmark.add_result(result)

# Upload manually
asyncio.run(upload_results(benchmark.results, benchmark.name, benchmark.metadata))
```

## See Also

- [Configuration](./configuration.md)
