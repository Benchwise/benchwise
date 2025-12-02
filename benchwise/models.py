from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from benchwise.types import ModelConfig, PricingInfo


class ModelAdapter(ABC):
    """Abstract base class for model adapters."""

    def __init__(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        self.model_name = model_name
        self.config: ModelConfig = config or {}

    @abstractmethod
    async def generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate responses for a list of prompts."""
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Get token count for text."""
        pass

    @abstractmethod
    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts."""
        pass


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models."""

    def __init__(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        super().__init__(model_name, config)
        try:
            import openai

            self.client = openai.AsyncOpenAI(
                api_key=config.get("api_key") if config else None
            )
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Please install it with: pip install 'benchwise[llm-apis]' or pip install openai"
            )

        # Model pricing (per 1K tokens)
        self.pricing: Dict[str, PricingInfo] = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "gpt-4o": {"input": 0.005, "output": 0.015},
        }

    async def generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate responses using OpenAI API."""
        responses: List[str] = []

        # Default parameters - exclude api_key from generation params
        generation_params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        # Add other config params but exclude api_key
        for key, value in self.config.items():
            if key != "api_key":  # Exclude api_key from generation params
                generation_params[key] = value

        for prompt in prompts:
            try:
                response = await self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}], **generation_params
                )
                responses.append(response.choices[0].message.content)
            except Exception as e:
                responses.append(f"Error: {str(e)}")

        return responses

    def get_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4  # Rough estimate: 1 token ≈ 4 characters

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost estimate."""
        model_pricing = self.pricing.get(
            self.model_name, {"input": 0.01, "output": 0.03}
        )
        input_cost = (input_tokens / 1000) * float(model_pricing["input"])
        output_cost = (output_tokens / 1000) * float(model_pricing["output"])
        return input_cost + output_cost


class AnthropicAdapter(ModelAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        super().__init__(model_name, config)
        try:
            import anthropic

            self.client = anthropic.AsyncAnthropic(
                api_key=config.get("api_key") if config else None
            )
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Please install it with: pip install 'benchwise[llm-apis]' or pip install anthropic"
            )

        # Model pricing (per 1K tokens)
        self.pricing: Dict[str, PricingInfo] = {
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
        }

    async def generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate responses using Anthropic API."""
        responses: List[str] = []

        # Default parameters - exclude api_key from generation params
        generation_params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0),
            "max_tokens": kwargs.get("max_tokens", 1000),
        }

        # Add other config params but exclude api_key
        for key, value in self.config.items():
            if key != "api_key":  # Exclude api_key from generation params
                generation_params[key] = value

        for prompt in prompts:
            try:
                response = await self.client.messages.create(
                    messages=[{"role": "user", "content": prompt}], **generation_params
                )
                responses.append(response.content[0].text)
            except Exception as e:
                responses.append(f"Error: {str(e)}")

        return responses

    def get_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        return len(text) // 4

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost estimate."""
        model_pricing = self.pricing.get(
            self.model_name, {"input": 0.003, "output": 0.015}
        )
        input_cost = (input_tokens / 1000) * float(model_pricing["input"])
        output_cost = (output_tokens / 1000) * float(model_pricing["output"])
        return input_cost + output_cost


class GoogleAdapter(ModelAdapter):
    """Adapter for Google Gemini models."""

    def __init__(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        super().__init__(model_name, config)
        try:
            import google.generativeai as genai

            if config and "api_key" in config:
                genai.configure(api_key=config["api_key"])
            self.model = genai.GenerativeModel(model_name)
            self.genai = genai
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. Please install it with: pip install 'benchwise[llm-apis]' or pip install google-generativeai"
            )

    async def generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate responses using Google Gemini API."""
        responses: List[str] = []

        for prompt in prompts:
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.genai.types.GenerationConfig(
                        temperature=kwargs.get("temperature", 0),
                        max_output_tokens=kwargs.get("max_tokens", 1000),
                    ),
                )
                responses.append(response.text)
            except Exception as e:
                responses.append(f"Error: {str(e)}")

        return responses

    def get_token_count(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost estimate (Google pricing varies)."""
        # Placeholder pricing
        input_cost = (input_tokens / 1000) * 0.001
        output_cost = (output_tokens / 1000) * 0.002
        return input_cost + output_cost


class HuggingFaceAdapter(ModelAdapter):
    """Adapter for Hugging Face models."""

    def __init__(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        super().__init__(model_name, config)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except ImportError:
            raise ImportError(
                "Transformers package not installed. Please install it with: pip install 'benchwise[transformers]' or pip install transformers torch"
            )

    async def generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate responses using Hugging Face models."""
        responses: List[str] = []

        for prompt in prompts:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=kwargs.get("max_tokens", 1000),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response[len(prompt) :].strip())
            except Exception as e:
                responses.append(f"Error: {str(e)}")

        return responses

    def get_token_count(self, text: str) -> int:
        """Get actual token count using tokenizer."""
        return len(self.tokenizer.encode(text))

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Hugging Face models are typically free when self-hosted."""
        return 0.0


class MockAdapter(ModelAdapter):
    """Mock adapter for testing without API dependencies."""

    def __init__(self, model_name: str, config: Optional[ModelConfig] = None) -> None:
        super().__init__(model_name, config)

    async def generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate mock responses."""
        return [
            f"Mock response from {self.model_name} for: {prompt[:50]}..."
            for prompt in prompts
        ]

    def get_token_count(self, text: str) -> int:
        """Mock token count."""
        return len(text) // 4

    def get_cost_estimate(self, input_tokens: int, output_tokens: int) -> float:
        """Mock cost estimate."""
        return 0.001  # Very cheap mock


def get_model_adapter(
    model_name: str, config: Optional[ModelConfig] = None
) -> ModelAdapter:
    """Factory function to get the appropriate model adapter."""

    # OpenAI models
    if model_name.startswith("gpt-"):
        return OpenAIAdapter(model_name, config)

    # Anthropic models
    elif model_name.startswith("claude-"):
        return AnthropicAdapter(model_name, config)

    # Google models
    elif model_name.startswith("gemini-"):
        return GoogleAdapter(model_name, config)

    # Mock models for testing
    elif model_name.startswith("mock-"):
        return MockAdapter(model_name, config)

    # Hugging Face models (default)
    else:
        return HuggingFaceAdapter(model_name, config)
