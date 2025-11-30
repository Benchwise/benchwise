---
sidebar_position: 8
---

# Benchwise Client

API client for Benchwise platform.

## Methods

Detailed documentation for the Benchwise client's public methods.

### upload_results
```python
async def upload_results(results: List[EvaluationResult], test_name: str, dataset_info: Dict[str, Any]) -> bool:
    ...
```

Upload results to Benchwise API.

### sync_offline_results
```python
async def sync_offline_results() -> int:
    ...
```

Sync queued offline results.

### close_client
```python
async def close_client():
    ...
```

Close the context-local client and release HTTP connections. Optional but recommended for clean shutdown.

## Usage

Examples and guidelines for interacting with the Benchwise client.

The client is managed internally via context variables (singleton pattern). Calling `close_client()` is **optional** but recommended for clean shutdown.

```python
import asyncio
from benchwise.client import upload_results, close_client
from benchwise import BenchmarkResult

# Create a benchmark result
benchmark = BenchmarkResult(
    name="My Benchmark",
    metadata={"version": "1.0"}
)

# Upload results
asyncio.run(upload_results(benchmark.results, benchmark.name, benchmark.metadata))

# Optionally close client when completely done (recommended at program exit)
asyncio.run(close_client())
```

**Note:** If you don't call `close_client()`, the HTTP connection may remain open until garbage collection. For long-running applications or scripts, call `close_client()` at the end to ensure proper cleanup and avoid resource leaks.

## See Also

- [API Integration](../advanced/api-integration.md)
- [Configuration](./config.md)
