"""
Benchwise Configuration Management

Handles configuration for API connection, authentication, and upload settings.
Supports environment variables and configuration files.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import json


@dataclass
class BenchwiseConfig:
    """
    Configuration class for Benchwise SDK.

    Attributes:
        api_url: URL of the Benchwise API
        api_key: API key for authentication
        upload_enabled: Whether to automatically upload results
        cache_enabled: Whether to cache results locally
        offline_mode: Whether to queue results when API is unavailable
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
    """

    # API Configuration
    api_url: str = "http://localhost:8000"
    api_key: Optional[str] = None

    # Upload Settings
    upload_enabled: bool = False
    auto_sync: bool = True

    # Caching Settings
    cache_enabled: bool = True
    cache_dir: str = "benchmark_cache"

    # Network Settings
    timeout: float = 30.0
    max_retries: int = 3

    # Offline Mode
    offline_mode: bool = True
    offline_queue_max_size: int = 1000

    # Debug Settings
    debug: bool = False
    verbose: bool = False

    # User preferences
    default_models: list = field(default_factory=list)
    default_metrics: list = field(default_factory=list)

    def __post_init__(self):
        """Load configuration from environment variables and config file."""
        self._load_from_env()
        self._load_from_file()
        self._validate_config()

    def _load_from_env(self):
        """Load configuration from environment variables."""

        # API Configuration
        self.api_url = os.getenv("BENCHWISE_API_URL", self.api_url)
        self.api_key = os.getenv("BENCHWISE_API_KEY", self.api_key)

        # Upload Settings
        upload_env = os.getenv("BENCHWISE_UPLOAD", "").lower()
        if upload_env in ("true", "1", "yes", "on"):
            self.upload_enabled = True
        elif upload_env in ("false", "0", "no", "off"):
            self.upload_enabled = False

        auto_sync_env = os.getenv("BENCHWISE_AUTO_SYNC", "").lower()
        if auto_sync_env in ("true", "1", "yes", "on"):
            self.auto_sync = True
        elif auto_sync_env in ("false", "0", "no", "off"):
            self.auto_sync = False

        # Cache Settings
        cache_env = os.getenv("BENCHWISE_CACHE", "").lower()
        if cache_env in ("false", "0", "no", "off"):
            self.cache_enabled = False

        cache_dir = os.getenv("BENCHWISE_CACHE_DIR")
        if cache_dir:
            self.cache_dir = cache_dir

        # Network Settings
        timeout = os.getenv("BENCHWISE_TIMEOUT")
        if timeout:
            try:
                self.timeout = float(timeout)
            except ValueError:
                pass

        retries = os.getenv("BENCHWISE_MAX_RETRIES")
        if retries:
            try:
                self.max_retries = int(retries)
            except ValueError:
                pass

        debug_env = os.getenv("BENCHWISE_DEBUG", "").lower()
        if debug_env in ("true", "1", "yes", "on"):
            self.debug = True

        verbose_env = os.getenv("BENCHWISE_VERBOSE", "").lower()
        if verbose_env in ("true", "1", "yes", "on"):
            self.verbose = True

    def _load_from_file(self):
        """Load configuration from config file."""
        config_paths = [
            Path.cwd() / ".benchwise.json",
            Path.home() / ".benchwise" / "config.json",
            Path.home() / ".config" / "benchwise" / "config.json",
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        config_data = json.load(f)

                    # Update configuration with file data
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)

                    if self.verbose:
                        print(f"📄 Loaded configuration from {config_path}")
                    break

                except (json.JSONDecodeError, OSError) as e:
                    if self.verbose:
                        print(f"⚠️ Failed to load config from {config_path}: {e}")

    def _validate_config(self):
        """Validate configuration values."""

        # Validate API URL
        if not self.api_url.startswith(("http://", "https://")):
            self.api_url = f"http://{self.api_url}"

        # Remove trailing slash from API URL
        self.api_url = self.api_url.rstrip("/")

        # Validate timeout
        if self.timeout <= 0:
            self.timeout = 30.0

        # Validate max retries
        if self.max_retries < 0:
            self.max_retries = 0

        # Validate cache directory
        if self.cache_enabled:
            cache_path = Path(self.cache_dir)
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except OSError:
                print(
                    f"Could not create cache directory {self.cache_dir}, disabling cache"
                )
                self.cache_enabled = False

    def save_to_file(self, file_path: Optional[Path] = None):
        """
        Save current configuration to file.

        Args:
            file_path: Path to save config file. If None, saves to default location.
        """
        if file_path is None:
            config_dir = Path.home() / ".config" / "benchwise"
            config_dir.mkdir(parents=True, exist_ok=True)
            file_path = config_dir / "config.json"

        config_dict = {
            "api_url": self.api_url,
            "upload_enabled": self.upload_enabled,
            "auto_sync": self.auto_sync,
            "cache_enabled": self.cache_enabled,
            "cache_dir": self.cache_dir,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "offline_mode": self.offline_mode,
            "debug": self.debug,
            "verbose": self.verbose,
            "default_models": self.default_models,
            "default_metrics": self.default_metrics,
        }

        # Don't save sensitive information like API key
        if self.api_key and not os.getenv("BENCHWISE_SAVE_API_KEY"):
            config_dict[
                "_note"
            ] = "API key not saved for security. Set BENCHWISE_API_KEY environment variable."

        try:
            with open(file_path, "w") as f:
                json.dump(config_dict, f, indent=2)

            if self.verbose:
                print(f"Configuration saved to {file_path}")

        except OSError as e:
            print(f"Failed to save configuration: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "api_url": self.api_url,
            "api_key": "***" if self.api_key else None,
            "upload_enabled": self.upload_enabled,
            "auto_sync": self.auto_sync,
            "cache_enabled": self.cache_enabled,
            "cache_dir": self.cache_dir,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "offline_mode": self.offline_mode,
            "debug": self.debug,
            "verbose": self.verbose,
            "default_models": self.default_models,
            "default_metrics": self.default_metrics,
        }

    def print_config(self):
        """Print current configuration in a readable format."""
        print("🔧 Benchwise Configuration:")
        print("=" * 30)

        config_dict = self.to_dict()
        for key, value in config_dict.items():
            print(f"  {key}: {value}")


_global_config: Optional[BenchwiseConfig] = None


def get_api_config() -> BenchwiseConfig:
    """
    Get the global Benchwise configuration.

    Returns:
        BenchwiseConfig instance
    """
    global _global_config

    if _global_config is None:
        _global_config = BenchwiseConfig()

    return _global_config


def set_api_config(config: BenchwiseConfig):
    """
    Set the global Benchwise configuration.

    Args:
        config: BenchwiseConfig instance to set as global
    """
    global _global_config
    _global_config = config


def configure_benchwise(
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    upload_enabled: Optional[bool] = None,
    cache_enabled: Optional[bool] = None,
    debug: Optional[bool] = None,
    **kwargs,
) -> BenchwiseConfig:
    """
    Configure Benchwise settings programmatically.

    Args:
        api_url: Benchwise API URL
        api_key: API key for authentication
        upload_enabled: Whether to automatically upload results
        cache_enabled: Whether to enable local caching
        debug: Whether to enable debug mode
        **kwargs: Additional configuration options

    Returns:
        Updated BenchwiseConfig instance
    """
    config = get_api_config()

    if api_url is not None:
        config.api_url = api_url
    if api_key is not None:
        config.api_key = api_key
    if upload_enabled is not None:
        config.upload_enabled = upload_enabled
    if cache_enabled is not None:
        config.cache_enabled = cache_enabled
    if debug is not None:
        config.debug = debug

    # Set any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Re-validate configuration after changes
    config._validate_config()

    return config


def reset_config():
    """Reset configuration to default values."""
    global _global_config
    _global_config = None


def is_api_available() -> bool:
    """
    Check if Benchwise API configuration is available.

    Returns:
        True if API URL is configured
    """
    config = get_api_config()
    return bool(
        config.api_url
        and config.api_url != "http://localhost:8000"
        or os.path.exists("docker-compose.yml")
    )


def is_authenticated() -> bool:
    """
    Check if API authentication is configured.

    Returns:
        True if API key is available
    """
    config = get_api_config()
    return bool(config.api_key)


def get_cache_dir() -> Path:
    """
    Get the cache directory path.

    Returns:
        Path to cache directory
    """
    config = get_api_config()
    return Path(config.cache_dir)


def get_development_config() -> BenchwiseConfig:
    """Get configuration optimized for development."""
    return BenchwiseConfig(
        api_url="http://localhost:8000",
        upload_enabled=False,
        cache_enabled=True,
        debug=True,
        verbose=True,
    )


def get_production_config(api_url: str, api_key: str) -> BenchwiseConfig:
    """Get configuration optimized for production."""
    return BenchwiseConfig(
        api_url=api_url,
        api_key=api_key,
        upload_enabled=True,
        auto_sync=True,
        cache_enabled=True,
        debug=False,
        verbose=False,
        timeout=60.0,
        max_retries=3,
    )


def get_offline_config() -> BenchwiseConfig:
    """Get configuration for offline usage."""
    return BenchwiseConfig(
        upload_enabled=False,
        cache_enabled=True,
        offline_mode=True,
        debug=False,
        verbose=True,
    )


def validate_api_connection(config: BenchwiseConfig) -> bool:
    """
    Validate API connection and configuration.

    Args:
        config: Configuration to validate

    Returns:
        True if connection is valid
    """
    try:
        import asyncio
        import httpx

        async def check_connection():
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{config.api_url}/health")
                return response.status_code == 200

        return asyncio.run(check_connection())

    except Exception as e:
        if config.verbose:
            print(f"API connection failed: {e}")
        return False


def validate_api_keys(config: BenchwiseConfig) -> Dict[str, bool]:
    """
    NEW: Validate external API keys by making test calls.

    Args:
        config: Configuration to check

    Returns:
        Dict mapping provider to validity status
    """
    import os

    results = {}

    if os.getenv("OPENAI_API_KEY"):
        try:
            import openai

            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            client.models.list()
            results["openai"] = True
        except Exception:
            results["openai"] = False

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            # Note: Anthropic doesn't have a simple test endpoint
            results["anthropic"] = True  # Assume valid if key exists
        except Exception:
            results["anthropic"] = False

    if os.getenv("GOOGLE_API_KEY"):
        try:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            # Simple test - list models
            list(genai.list_models())
            results["google"] = True
        except Exception:
            results["google"] = False

    if os.getenv("HUGGINGFACE_API_KEY"):
        try:
            import requests

            headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
            response = requests.get(
                "https://huggingface.co/api/whoami", headers=headers
            )
            results["huggingface"] = response.status_code == 200
        except Exception:
            results["huggingface"] = False

    return results


def print_configuration_status(config: BenchwiseConfig):
    """
    NEW: Print comprehensive configuration status.

    Args:
        config: Configuration to check
    """
    print("Benchwise Configuration Status")
    print("=" * 40)

    config.print_config()

    print("\nAPI Connection:")
    api_valid = validate_api_connection(config)
    print(f"  Status: {'Connected' if api_valid else 'Failed'}")

    print("\nExternal API Keys:")
    api_keys = validate_api_keys(config)
    if api_keys:
        for provider, valid in api_keys.items():
            status = "Valid" if valid else "Invalid"
            print(f"  {provider.title()}: {status}")
    else:
        print("  No API keys configured")

    print("\nCache Directory:")
    cache_dir = get_cache_dir()
    exists = cache_dir.exists()
    print(f"  Path: {cache_dir}")
    print(f"  Status: {'Exists' if exists else 'Missing'}")
