---
sidebar_position: 2
---

# API Integration

Integrate with the Benchwise platform API.

## Setup

```python
from benchwise import configure_benchwise

configure_benchwise(
    api_url="https://api.benchwise.ai",
    api_key="your_api_key",
    upload_enabled=True
)
```

## Auto-Upload Results

```python
from benchwise import evaluate, benchmark

@benchmark("My Benchmark", "Description")
@evaluate("gpt-4", upload=True)
async def my_test(model, dataset):
    # Results will be automatically uploaded
    return {"accuracy": 0.85}
```

## Manual Upload

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

## Offline Mode

```python
from benchwise import get_offline_config, configure_benchwise

# Enable offline mode
configure_benchwise(**get_offline_config())

# Results are queued locally

# Later, sync when online
from benchwise.client import sync_offline_results
await sync_offline_results()
```

## See Also

- [Configuration](./configuration.md)
- [Offline Mode](./offline-mode.md)
