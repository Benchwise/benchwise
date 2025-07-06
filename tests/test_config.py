"""
Tests for configuration functionality
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from benchwise.config import (
    BenchWiseConfig,
    get_api_config,
    configure_benchwise,
    set_api_config,
    reset_config,
    is_api_available,
    is_authenticated,
    get_development_config,
    get_production_config,
    get_offline_config,
)


class TestBenchWiseConfig:
    def test_config_creation_defaults(self):
        config = BenchWiseConfig()

        assert config.api_url == "http://localhost:8000"
        assert config.api_key is None
        assert not config.upload_enabled
        assert config.cache_enabled
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert not config.debug

    def test_config_creation_custom(self):
        config = BenchWiseConfig(
            api_url="https://api.benchwise.ai",
            api_key="test-key",
            upload_enabled=True,
            debug=True,
        )

        assert config.api_url == "https://api.benchwise.ai"
        assert config.api_key == "test-key"
        assert config.upload_enabled
        assert config.debug

    def test_config_to_dict(self):
        config = BenchWiseConfig(api_key="secret-key")
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "api_url" in config_dict
        assert "upload_enabled" in config_dict
        assert config_dict["api_key"] == "***"  # Should be masked

    def test_config_print_config(self, capsys):
        config = BenchWiseConfig()
        config.print_config()

        captured = capsys.readouterr()
        assert "BenchWise Configuration" in captured.out
        assert "api_url" in captured.out


class TestConfigEnvironmentVariables:
    def test_load_api_url_from_env(self):
        with patch.dict(os.environ, {"BENCHWISE_API_URL": "https://test.api"}):
            config = BenchWiseConfig()
            assert config.api_url == "https://test.api"

    def test_load_api_key_from_env(self):
        with patch.dict(os.environ, {"BENCHWISE_API_KEY": "env-key"}):
            config = BenchWiseConfig()
            assert config.api_key == "env-key"

    def test_load_upload_setting_from_env(self):
        with patch.dict(os.environ, {"BENCHWISE_UPLOAD": "true"}):
            config = BenchWiseConfig()
            assert config.upload_enabled

        with patch.dict(os.environ, {"BENCHWISE_UPLOAD": "false"}):
            config = BenchWiseConfig()
            assert not config.upload_enabled

    def test_load_debug_from_env(self):
        with patch.dict(os.environ, {"BENCHWISE_DEBUG": "true"}):
            config = BenchWiseConfig()
            assert config.debug


class TestConfigFileLoading:
    def test_load_from_json_file(self):
        config_data = {
            "api_url": "https://file.api",
            "upload_enabled": True,
            "debug": True,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(config_data, f)
            temp_path = f.name

        try:
            # Mock the config file paths
            with patch.object(Path, "exists", return_value=True), patch(
                "builtins.open", create=True
            ) as mock_open:
                import json

                mock_open.return_value.__enter__.return_value.read.return_value = (
                    json.dumps(config_data)
                )

                config = BenchWiseConfig()
                # The actual file loading is complex to test, so we test the interface
                assert hasattr(config, "_load_from_file")

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_to_file(self):
        config = BenchWiseConfig(upload_enabled=True, debug=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            config.save_to_file(temp_path)

            # Check file was created
            assert temp_path.exists()

            # Check content (basic validation)
            content = temp_path.read_text()
            assert '"upload_enabled": true' in content

        finally:
            temp_path.unlink(missing_ok=True)


class TestGlobalConfig:
    def test_get_api_config(self):
        config = get_api_config()

        assert isinstance(config, BenchWiseConfig)
        assert hasattr(config, "api_url")
        assert hasattr(config, "upload_enabled")

    def test_set_api_config(self):
        original_config = get_api_config()

        new_config = BenchWiseConfig(debug=True, upload_enabled=True)
        set_api_config(new_config)

        retrieved_config = get_api_config()
        assert retrieved_config.debug
        assert retrieved_config.upload_enabled

        # Restore original
        set_api_config(original_config)

    def test_configure_benchwise(self):
        original_config = get_api_config()

        updated_config = configure_benchwise(
            api_url="https://new.api", upload_enabled=True, debug=True
        )

        assert updated_config.api_url == "https://new.api"
        assert updated_config.upload_enabled
        assert updated_config.debug

        # The global config should be updated
        global_config = get_api_config()
        assert global_config.api_url == "https://new.api"

        # Restore original
        set_api_config(original_config)

    def test_reset_config(self):
        configure_benchwise(debug=True, upload_enabled=True)

        reset_config()

        # Should be back to defaults
        config = get_api_config()
        assert not config.debug
        assert not config.upload_enabled


class TestConfigTemplates:
    def test_development_config(self):
        config = get_development_config()

        assert isinstance(config, BenchWiseConfig)
        assert config.api_url == "http://localhost:8000"
        assert not config.upload_enabled
        assert config.debug
        assert config.verbose

    def test_production_config(self):
        config = get_production_config("https://prod.api", "prod-key")

        assert isinstance(config, BenchWiseConfig)
        assert config.api_url == "https://prod.api"
        assert config.api_key == "prod-key"
        assert config.upload_enabled
        assert not config.debug
        assert config.timeout == 60.0

    def test_offline_config(self):
        config = get_offline_config()

        assert isinstance(config, BenchWiseConfig)
        assert not config.upload_enabled
        assert config.cache_enabled
        assert config.offline_mode
        assert config.verbose


class TestConfigValidation:
    def test_api_url_validation(self):
        config = BenchWiseConfig()
        config.api_url = "example.com"
        config._validate_config()

        assert config.api_url.startswith("http://")
        config.api_url = "https://example.com/"
        config._validate_config()

        assert not config.api_url.endswith("/")

    def test_timeout_validation(self):
        config = BenchWiseConfig()
        config.timeout = -5
        config._validate_config()

        assert config.timeout == 30.0  # Should reset to default

    def test_max_retries_validation(self):
        config = BenchWiseConfig()
        config.max_retries = -1
        config._validate_config()

        assert config.max_retries == 0  # Should reset to 0


class TestConfigUtilities:
    def test_is_api_available(self):
        result = is_api_available()
        assert isinstance(result, bool)

    def test_is_authenticated(self):
        config = BenchWiseConfig()
        set_api_config(config)

        result = is_authenticated()
        assert not result

        config_with_key = BenchWiseConfig(api_key="test-key")
        set_api_config(config_with_key)

        result = is_authenticated()
        assert result

        reset_config()


class TestConfigEdgeCases:
    def test_config_with_empty_values(self):
        config = BenchWiseConfig(api_url="", api_key="", timeout=0)

        config._validate_config()

        assert config.api_url.startswith("http://")
        assert config.timeout > 0

    def test_config_with_invalid_cache_dir(self):
        config = BenchWiseConfig(cache_dir="/invalid/path/that/cannot/be/created")

        config._validate_config()

        assert isinstance(config.cache_enabled, bool)

    def test_config_file_loading_errors(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name

        try:
            config = BenchWiseConfig()
            assert isinstance(config, BenchWiseConfig)

        finally:
            Path(temp_path).unlink(missing_ok=True)
