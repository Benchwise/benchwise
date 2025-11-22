---
sidebar_position: 7
---

# Configuration API

Detailed API reference for Benchwise configuration management.

## BenchwiseConfig

```python
@dataclass
class BenchwiseConfig:
    api_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    upload_enabled: bool = False
    cache_enabled: bool = True
    offline_mode: bool = True
    timeout: float = 30.0
    max_retries: int = 3
    debug: bool = False
    verbose: bool = False
    default_models: list = field(default_factory=list)
    default_metrics: list = field(default_factory=list)
```

Configuration class for Benchwise SDK. Attributes can be set programmatically or via environment variables/config files.

## Functions

### configure_benchwise

```python
def configure_benchwise(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    upload_enabled: Optional[bool] = None,
    cache_enabled: Optional[bool] = None,
    debug: Optional[bool] = None,
    **kwargs,
) -> BenchwiseConfig:
    ...
```

Configure Benchwise settings programmatically.

### get_api_config

```python
def get_api_config() -> BenchwiseConfig:
    ...
```

Get the global Benchwise configuration instance.

### set_api_config

```python
def set_api_config(config: BenchwiseConfig):
    ...
```

Set the global Benchwise configuration instance.

### reset_config

```python
def reset_config():
    ...
```

Reset configuration to default values.

### is_api_available

```python
def is_api_available() -> bool:
    ...
```

Check if Benchwise API configuration is available.

### is_authenticated

```python
def is_authenticated() -> bool:
    ...
```

Check if API authentication is configured.

### get_cache_dir

```python
def get_cache_dir() -> Path:
    ...
```

Get the cache directory path.

### get_development_config

```python
def get_development_config() -> BenchwiseConfig:
    ...
```

Get configuration optimized for development.

### get_production_config

```python
def get_production_config(api_url: str, api_key: str) -> BenchwiseConfig:
    ...
```

Get configuration optimized for production.

### get_offline_config

```python
def get_offline_config() -> BenchwiseConfig:
    ...
```

Get configuration for offline usage.

### validate_api_connection

```python
def validate_api_connection(config: BenchwiseConfig) -> bool:
    ...
```

Validate API connection and configuration.

### validate_api_keys

```python
def validate_api_keys(config: BenchwiseConfig) -> Dict[str, bool]:
    ...
```

Validate external API keys by making test calls.

### print_configuration_status

```python
def print_configuration_status(config: BenchwiseConfig):
    ...
```

Print comprehensive configuration status.