---
sidebar_position: 1
---

# Configuration

Advanced configuration options for Benchwise.

## Configuration Sources

Benchwise loads configuration from multiple sources (in order of precedence):

1. Code-level configuration (`configure_benchwise()`)
2. Environment variables (`BENCHWISE_*`)
3. Config files (`.benchwise.json`, `~/.benchwise/config.json`)
4. Default values

## Programmatic Configuration

```python
from benchwise import configure_benchwise

configure_benchwise(
    api_url="https://api.benchwise.ai",
    api_key="your_key",
    upload_enabled=True,
    cache_enabled=True,
    debug=False
)
```

## Environment Variables

```bash
export BENCHWISE_API_URL="https://api.benchwise.ai"
export BENCHWISE_API_KEY="your_key"
export BENCHWISE_UPLOAD="true"
export BENCHWISE_CACHE="true"
export BENCHWISE_DEBUG="false"
```

## Config File

Create `.benchwise.json` in your project root:

```json
{
  "api_url": "https://api.benchwise.ai",
  "api_key": "your_key",
  "upload_enabled": true,
  "cache_enabled": true,
  "debug": false
}
```

## Preset Configurations

```python
from benchwise import get_development_config, get_production_config, get_offline_config

# Development mode
config = get_development_config()

# Production mode
config = get_production_config()

# Offline mode
config = get_offline_config()
```

## See Also

- [API Configuration](../api/config.md)
- [Offline Mode](./offline-mode.md)
