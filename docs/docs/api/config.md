---
sidebar_position: 7
---

# Configuration

Configuration management for Benchwise.

## BenchWiseConfig

```python
@dataclass
class BenchWiseConfig:
    api_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    upload_enabled: bool = False
    cache_enabled: bool = True
    debug: bool = False
```

## configure_benchwise

```python
def configure_benchwise(**kwargs) -> None
```

Configure Benchwise programmatically.

```python
from benchwise import configure_benchwise

configure_benchwise(
    api_url="https://api.benchwise.ai",
    upload_enabled=True,
    cache_enabled=True
)
```

## Preset Configurations

```python
from benchwise import get_development_config, get_production_config, get_offline_config

# Development
dev_config = get_development_config()

# Production
prod_config = get_production_config()

# Offline
offline_config = get_offline_config()
```

## Environment Variables

- `BENCHWISE_API_URL`
- `BENCHWISE_API_KEY`
- `BENCHWISE_UPLOAD`
- `BENCHWISE_CACHE`
- `BENCHWISE_DEBUG`

## See Also

- [Advanced Configuration](../advanced/configuration.md)
