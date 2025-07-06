"""
Tests for model adapters
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from benchwise.models import (
    ModelAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    GoogleAdapter,
    HuggingFaceAdapter,
    MockAdapter,
    get_model_adapter,
)


class TestModelAdapterBase:
    def test_abstract_methods(self):
        with pytest.raises(TypeError):
            ModelAdapter("test-model")


class TestMockAdapter:
    def test_mock_adapter_creation(self):
        adapter = MockAdapter("mock-test-model")

        assert adapter.model_name == "mock-test-model"
        assert isinstance(adapter.config, dict)

    @pytest.mark.asyncio
    async def test_mock_generate(self):
        adapter = MockAdapter("mock-test")
        prompts = ["Hello", "World"]

        responses = await adapter.generate(prompts)

        assert len(responses) == 2
        assert all(isinstance(r, str) for r in responses)
        assert all("mock-test" in r.lower() for r in responses)

    def test_mock_token_count(self):
        adapter = MockAdapter("mock-test")

        count = adapter.get_token_count("Hello world")

        assert isinstance(count, int)
        assert count > 0

    def test_mock_cost_estimate(self):
        adapter = MockAdapter("mock-test")

        cost = adapter.get_cost_estimate(100, 50)

        assert isinstance(cost, float)
        assert cost >= 0


class TestGetModelAdapter:
    def test_get_gpt_adapter(self):
        adapter = get_model_adapter("gpt-3.5-turbo")
        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.model_name == "gpt-3.5-turbo"

    def test_get_claude_adapter(self):
        adapter = get_model_adapter("claude-3-haiku")
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter.model_name == "claude-3-haiku"

    def test_get_gemini_adapter(self):
        adapter = get_model_adapter("gemini-pro")
        assert isinstance(adapter, GoogleAdapter)
        assert adapter.model_name == "gemini-pro"

    def test_get_mock_adapter(self):
        adapter = get_model_adapter("mock-test-model")
        assert isinstance(adapter, MockAdapter)
        assert adapter.model_name == "mock-test-model"

    def test_get_huggingface_adapter_default(self):
        # Use a mock model name that won't trigger real HuggingFace download
        with patch("transformers.AutoTokenizer"), patch(
            "transformers.AutoModelForCausalLM"
        ):
            adapter = get_model_adapter("test/unknown-model-name")
            assert isinstance(adapter, HuggingFaceAdapter)
            assert adapter.model_name == "test/unknown-model-name"


class TestOpenAIAdapter:
    def test_openai_adapter_creation(self):
        with patch("openai.AsyncOpenAI"):
            adapter = OpenAIAdapter("gpt-3.5-turbo")
            assert adapter.model_name == "gpt-3.5-turbo"
            assert "gpt-3.5-turbo" in adapter.pricing

    def test_openai_token_count(self):
        with patch("openai.AsyncOpenAI"):
            adapter = OpenAIAdapter("gpt-3.5-turbo")
            count = adapter.get_token_count("Hello world")
            assert isinstance(count, int)
            assert count > 0

    def test_openai_cost_estimate(self):
        with patch("openai.AsyncOpenAI"):
            adapter = OpenAIAdapter("gpt-3.5-turbo")
            cost = adapter.get_cost_estimate(1000, 500)
            assert isinstance(cost, float)
            assert cost > 0

    @pytest.mark.asyncio
    async def test_openai_generate_mocked(self):
        with patch("openai.AsyncOpenAI") as mock_openai_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Mock response"

            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client

            adapter = OpenAIAdapter("gpt-3.5-turbo")
            responses = await adapter.generate(["Hello"])

            assert len(responses) == 1
            assert responses[0] == "Mock response"

    def test_openai_import_error(self):
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named 'openai'")
        ):
            with pytest.raises(ImportError, match="OpenAI package not installed"):
                OpenAIAdapter("gpt-3.5-turbo")


class TestAnthropicAdapter:
    def test_anthropic_adapter_creation(self):
        with patch("anthropic.AsyncAnthropic"):
            adapter = AnthropicAdapter("claude-3-haiku")
            assert adapter.model_name == "claude-3-haiku"
            assert "claude-3-haiku" in adapter.pricing

    def test_anthropic_token_count(self):
        with patch("anthropic.AsyncAnthropic"):
            adapter = AnthropicAdapter("claude-3-haiku")
            count = adapter.get_token_count("Hello world")
            assert isinstance(count, int)
            assert count > 0

    def test_anthropic_cost_estimate(self):
        with patch("anthropic.AsyncAnthropic"):
            adapter = AnthropicAdapter("claude-3-haiku")
            cost = adapter.get_cost_estimate(1000, 500)
            assert isinstance(cost, float)
            assert cost > 0

    def test_anthropic_import_error(self):
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'anthropic'"),
        ):
            with pytest.raises(ImportError, match="Anthropic package not installed"):
                AnthropicAdapter("claude-3-haiku")


class TestGoogleAdapter:
    def test_google_adapter_creation(self):
        with patch("google.generativeai.GenerativeModel"):
            adapter = GoogleAdapter("gemini-pro")
            assert adapter.model_name == "gemini-pro"

    def test_google_token_count(self):
        with patch("google.generativeai.GenerativeModel"):
            adapter = GoogleAdapter("gemini-pro")
            count = adapter.get_token_count("Hello world")
            assert isinstance(count, int)
            assert count > 0

    def test_google_cost_estimate(self):
        with patch("google.generativeai.GenerativeModel"):
            adapter = GoogleAdapter("gemini-pro")
            cost = adapter.get_cost_estimate(1000, 500)
            assert isinstance(cost, float)
            assert cost >= 0

    def test_google_import_error(self):
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'google.generativeai'"),
        ):
            with pytest.raises(
                ImportError, match="Google Generative AI package not installed"
            ):
                GoogleAdapter("gemini-pro")


class TestHuggingFaceAdapter:
    def test_huggingface_adapter_creation(self):
        with patch("transformers.AutoTokenizer"), patch(
            "transformers.AutoModelForCausalLM"
        ):
            adapter = HuggingFaceAdapter("gpt2")
            assert adapter.model_name == "gpt2"

    def test_huggingface_cost_estimate(self):
        with patch("transformers.AutoTokenizer"), patch(
            "transformers.AutoModelForCausalLM"
        ):
            adapter = HuggingFaceAdapter("gpt2")
            cost = adapter.get_cost_estimate(1000, 500)
            assert cost == 0.0

    def test_huggingface_import_error(self):
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'transformers'"),
        ):
            with pytest.raises(ImportError, match="Transformers package not installed"):
                HuggingFaceAdapter("gpt2")


class TestModelConfiguration:
    def test_adapter_with_config(self):
        config = {"temperature": 0.5, "max_tokens": 1000}
        adapter = MockAdapter("mock-test", config)

        assert adapter.config == config
        assert adapter.config["temperature"] == 0.5

    def test_adapter_without_config(self):
        adapter = MockAdapter("mock-test")

        assert isinstance(adapter.config, dict)
        assert len(adapter.config) == 0


class TestModelNaming:
    def test_gpt_variants(self):
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]

        for model in models:
            adapter = get_model_adapter(model)
            assert isinstance(adapter, OpenAIAdapter)
            assert adapter.model_name == model

    def test_claude_variants(self):
        models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]

        for model in models:
            adapter = get_model_adapter(model)
            assert isinstance(adapter, AnthropicAdapter)
            assert adapter.model_name == model

    def test_gemini_variants(self):
        models = ["gemini-pro", "gemini-1.5-pro"]

        for model in models:
            adapter = get_model_adapter(model)
            assert isinstance(adapter, GoogleAdapter)
            assert adapter.model_name == model

    def test_mock_variants(self):
        models = ["mock-test", "mock-gpt", "mock-claude"]

        for model in models:
            adapter = get_model_adapter(model)
            assert isinstance(adapter, MockAdapter)
            assert adapter.model_name == model
