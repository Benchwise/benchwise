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
from benchwise.client import upload_results
from benchwise import BenchmarkResult

benchmark = BenchmarkResult("My Benchmark")
# ... add results ...

# Upload manually
await upload_results(benchmark)
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
