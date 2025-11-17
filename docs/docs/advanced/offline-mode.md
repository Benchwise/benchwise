---
sidebar_position: 6
---

# Offline Mode

Run evaluations without internet connectivity.

## Enable Offline Mode

```python
from benchwise import get_offline_config, configure_benchwise

configure_benchwise(**get_offline_config())
```

## How It Works

- Results are cached locally
- No API uploads attempted
- Queue results for later sync

## Sync When Online

```python
from benchwise.client import sync_offline_results

# When back online
await sync_offline_results()
```

## Cache Management

```python
from benchwise import cache

# Clear cache
cache.clear_cache()

# List cached results
cached = cache.list_cached_results()
print(f"Cached: {len(cached)} evaluations")
```

## See Also

- [Configuration](./configuration.md)
- [API Integration](./api-integration.md)
