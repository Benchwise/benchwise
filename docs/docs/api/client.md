---
sidebar_position: 8
---

# Benchwise Client

API client for Benchwise platform.

## Methods

### upload_results
```python
async def upload_results(results: BenchmarkResult) -> Dict[str, Any]
```

Upload results to Benchwise API.

### sync_offline_results
```python
async def sync_offline_results() -> None
```

Sync queued offline results.

## Usage

```python
import asyncio
from benchwise.client import upload_results, close_client
from benchwise import BenchmarkResult

# Create a benchmark result
benchmark = BenchmarkResult(
    name="My Benchmark",
    metadata={"version": "1.0"}
)

# Upload results (client is managed internally)
asyncio.run(upload_results(benchmark))

# Close client when done
asyncio.run(close_client())
```

## See Also

- [API Integration](../advanced/api-integration.md)
- [Configuration](./config.md)
