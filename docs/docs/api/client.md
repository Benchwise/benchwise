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
from benchwise.client import get_client, upload_results

client = get_client()

# Upload results
await upload_results(benchmark_results)

# Close client
from benchwise.client import close_client
await close_client()
```

## See Also

- [API Integration](../advanced/api-integration.md)
- [Configuration](./config.md)
