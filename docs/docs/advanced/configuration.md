---
sidebar_position: 1
---

# Configuration

Advanced configuration options for Benchwise.

## Configuration Sources

Understanding the order of precedence for Benchwise configuration settings.

Benchwise loads configuration from multiple sources (in order of precedence):

1. Code-level configuration (`configure_benchwise()`)
2. Environment variables (`BENCHWISE_*`)
3. Config files (`.benchwise.json`, `~/.benchwise/config.json`)
4. Default values


## Environment Variables

Set Benchwise configuration using environment variables.

```bash
export BENCHWISE_CACHE="true"
export BENCHWISE_DEBUG="false"
```

## Preset Configurations (Under development)

Quickly apply predefined configuration settings for common scenarios.

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
